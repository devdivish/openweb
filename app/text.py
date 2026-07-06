from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def extract_text(path: Path, content_type: str | None = None) -> str:
    """Extract text from a file.

    When ``OCR_EXTRACTION_URL`` is set, every file is sent to the external OCR
    extraction service (the ``ocr_detection_worker_updated`` ``/extract``
    endpoint) instead of using native parsers. This routes PDFs, Office docs,
    images, etc. through the VLM OCR pipeline (dots-mocr).

    Env vars:
      - ``OCR_EXTRACTION_URL``     full endpoint, e.g. ``http://localhost:8200/extract``
      - ``OCR_EXTRACTION_FORMAT``  ``markdown`` (default) or ``text``
      - ``OCR_EXTRACTION_TIMEOUT`` request timeout in seconds (default 300)
      - ``OCR_FALLBACK_NATIVE``    ``true`` (default) falls back to native parsing
                                   if the OCR service errors; ``false`` re-raises
    """
    url = os.getenv("OCR_EXTRACTION_URL", "").strip()
    if url:
        try:
            return _extract_via_ocr(path, content_type, url)
        except Exception as exc:
            if _truthy(os.getenv("OCR_FALLBACK_NATIVE", "true")):
                log.warning(
                    "OCR extraction failed for %s (%s); falling back to native parser",
                    path.name, exc,
                )
                return _extract_native(path, content_type)
            raise RuntimeError(f"OCR extraction failed for {path}: {exc}") from exc
    return _extract_native(path, content_type)


def _extract_via_ocr(path: Path, content_type: str | None, url: str) -> str:
    """POST the file to the OCR extraction service and return its text."""
    import httpx

    timeout = float(os.getenv("OCR_EXTRACTION_TIMEOUT", "300"))
    fmt = os.getenv("OCR_EXTRACTION_FORMAT", "markdown").strip().lower()
    with path.open("rb") as handle:
        files = {"file": (path.name, handle, content_type or "application/octet-stream")}
        response = httpx.post(url, files=files, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data.get("markdown") if fmt == "markdown" else data.get("text")
    if not text:  # fall back to whichever field is populated
        text = data.get("text") or data.get("markdown") or ""
    return normalize_text(text)


def _extract_native(path: Path, content_type: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to read PDF files") from exc
        reader = PdfReader(str(path))
        return normalize_text("\n\n".join(page.extract_text() or "" for page in reader.pages))

    if suffix == ".docx":
        try:
            import docx2txt
        except ImportError as exc:
            raise RuntimeError("Install docx2txt to read DOCX files") from exc
        return normalize_text(docx2txt.process(str(path)) or "")

    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return normalize_text(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return normalize_text(raw.decode("utf-8", errors="ignore"))


def summarize_text(text: str, max_sentences: int = 3, max_chars: int = 600) -> tuple[str, list[str]]:
    normalized = normalize_text(text)
    keywords = extract_keywords(normalized)
    sentences = _sentences(normalized)
    if not sentences:
        return "", keywords

    keyword_set = set(keywords[:12])
    ranked = []
    for position, sentence in enumerate(sentences):
        terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9_]+", sentence)}
        score = len(terms & keyword_set)
        if position == 0:
            score += 1
        ranked.append((score, -position, position, sentence))
    ranked.sort(reverse=True)
    selected_positions = sorted(item[2] for item in ranked[:max_sentences])
    selected = [sentences[position] for position in selected_positions]
    summary = " ".join(selected).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return summary, keywords


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "before",
        "can",
        "from",
        "has",
        "have",
        "into",
        "is",
        "it",
        "its",
        "of",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    counts: dict[str, int] = {}
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()):
        if term in stopwords:
            continue
        counts[term] = counts.get(term, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]
    if len(sentences) == 1 and len(sentences[0]) > 700:
        return [sentences[0][:700].rsplit(" ", 1)[0].strip()]
    return sentences


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list[str]:
    return [chunk.text for chunk in chunk_text_with_spans(text, chunk_size, overlap)]


def chunk_text_with_spans(text: str, chunk_size: int = 900, overlap: int = 180) -> list[TextChunk]:
    text = normalize_text(text)
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = _paragraph_spans(text)
    chunks: list[TextChunk] = []
    current = ""
    current_start = 0
    current_end = 0

    def emit(value: str, start_char: int, end_char: int) -> None:
        value = value.strip()
        if value:
            chunks.append(TextChunk(text=value, start_char=start_char, end_char=end_char))

    for paragraph, paragraph_start, paragraph_end in paragraphs:
        if len(paragraph) > chunk_size:
            emit(current, current_start, current_end)
            current = ""
            current_start = 0
            current_end = 0
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                piece = paragraph[start:end].strip()
                leading = len(paragraph[start:end]) - len(paragraph[start:end].lstrip())
                trailing = len(paragraph[start:end]) - len(paragraph[start:end].rstrip())
                emit(
                    piece,
                    paragraph_start + start + leading,
                    paragraph_start + min(end, len(paragraph)) - trailing,
                )
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            if not current:
                current_start = paragraph_start
            current = candidate
            current_end = paragraph_end
        else:
            emit(current, current_start, current_end)
            current = paragraph
            current_start = paragraph_start
            current_end = paragraph_end

    emit(current, current_start, current_end)
    return chunks


def _paragraph_spans(text: str) -> list[tuple[str, int, int]]:
    spans = []
    for match in re.finditer(r".+?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        spans.append((stripped, start, end))
    return spans
