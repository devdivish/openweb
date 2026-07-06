"""RTF extractor.

Strategy (in order):
    1. `striprtf` — pure Python, no native deps, easy to install offline
       (`pip install striprtf`). If present, used directly.
    2. LibreOffice fallback — `soffice --headless --convert-to txt` — the
       same tool already used for .doc/.ppt/.xls. Requires LibreOffice
       on the box.

If neither is available, the extractor raises with an actionable message
so the ES workflow can record it as `ExtractionStatus=10 / ErrorReason`.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from charset_normalizer import from_bytes

from .base import BaseExtractor, Block, ExtractedDoc, Page
from ..libreoffice import LibreOfficeError, convert as lo_convert

log = logging.getLogger(__name__)


class RtfExtractor(BaseExtractor):
    extensions = ["rtf"]
    format_name = "rtf"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        text = self._try_striprtf(path)
        if text is None:
            text = self._try_libreoffice(path)
        if text is None:
            raise RuntimeError(
                "RTF extraction unavailable: install `striprtf` "
                "(pip install striprtf) or ensure LibreOffice is on PATH."
            )

        page = Page(index=0, label=f"Document: {path.name}")
        if text.strip():
            page.blocks.append(Block(type="text", text=text.strip() + "\n",
                                     source="rtf"))
        return ExtractedDoc(source_path=str(path), format="rtf", pages=[page])

    # ------------------------------------------------------------------

    @staticmethod
    def _try_striprtf(path: Path) -> Optional[str]:
        try:
            from striprtf.striprtf import rtf_to_text  # type: ignore
        except ImportError:
            return None
        try:
            raw = path.read_bytes()
            # RTF is ASCII-with-escapes by spec, but files in the wild use
            # cp1252/latin-1. Try utf-8 first, fall back to latin-1.
            try:
                src = raw.decode("utf-8")
            except UnicodeDecodeError:
                src = raw.decode("latin-1", errors="replace")
            return rtf_to_text(src, errors="ignore")
        except Exception as e:
            log.warning("striprtf failed on %s: %s", path, e)
            return None

    def _try_libreoffice(self, path: Path) -> Optional[str]:
        fmt_cfg = self.cfg.extractors.get("rtf", {}) or {}
        if not fmt_cfg.get("legacy_via_libreoffice", True):
            return None
        lo_cfg = self.cfg.extractors.get("libreoffice", {}) or {}
        try:
            converted = lo_convert(
                path,
                target_ext="txt",
                timeout_s=float(lo_cfg.get("timeout_s", 300)),
                retries=int(lo_cfg.get("retries", 1)),
            )
        except LibreOfficeError as e:
            log.warning("LibreOffice fallback failed for %s: %s", path, e)
            return None

        try:
            raw = converted.read_bytes()
            best = from_bytes(raw).best()
            enc = best.encoding if best else "utf-8"
            try:
                return raw.decode(enc, errors="replace")
            except LookupError:
                return raw.decode("utf-8", errors="replace")
        finally:
            # The helper owns the output dir; clean up after ourselves.
            shutil.rmtree(converted.parent, ignore_errors=True)
