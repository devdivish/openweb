"""Plain-text extractor — no OCR, just decode + emit.

Handles the "already-text" family: plain .txt, logs, Markdown,
reStructuredText, config files, LaTeX source, SQL, and subtitle
formats. Every one of these is read byte-wise, decoded with
charset-normalizer, and emitted as a single text block per document.

Adding a new text-like extension
--------------------------------
Append to the `extensions` class attribute below. No other code changes
needed — the registry picks it up automatically.
"""
from __future__ import annotations

import logging
from pathlib import Path

from charset_normalizer import from_bytes

from .base import BaseExtractor, Block, ExtractedDoc, Page

log = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    extensions = [
        # Generic text
        "txt", "log",
        # Markdown / reST
        "md", "markdown", "rst",
        # Config files
        "ini", "conf", "cfg", "toml",
        # LaTeX source
        "tex", "bib",
        # Scripts / SQL (usually show up as "documents" in records)
        "sql",
        # Subtitles (occasionally ingested for transcripts)
        "srt", "vtt", "sbv",
    ]
    format_name = "text"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        fmt_cfg = self.cfg.extractors.get("text", {}) or {}
        max_chars = int(fmt_cfg.get("max_chars", 2_000_000))

        raw = path.read_bytes()
        best = from_bytes(raw).best()
        enc = best.encoding if best else "utf-8"
        try:
            content = raw.decode(enc, errors="replace")
        except LookupError:
            content = raw.decode("utf-8", errors="replace")

        warnings = []
        if len(content) > max_chars:
            content = content[:max_chars]
            warnings.append(f"truncated: source exceeded max_chars={max_chars}")

        # Strip BOM and trailing whitespace.
        content = content.lstrip("\ufeff").rstrip()

        page = Page(index=0, label=f"File: {path.name}")
        if content:
            page.blocks.append(Block(
                type="text",
                text=content,
                source=f"text:{path.suffix.lstrip('.').lower() or 'unknown'}",
            ))
        return ExtractedDoc(
            source_path=str(path),
            format="text",
            pages=[page],
            warnings=warnings,
        )
