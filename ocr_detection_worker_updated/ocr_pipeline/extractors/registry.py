"""Extension -> extractor class mapping.

Keeps `orchestrator.py` free of format-specific conditionals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from .base import BaseExtractor
from .csv_extractor import CsvExtractor
from .docx_extractor import DocxExtractor
from .email_extractor import EmailExtractor
from .html_extractor import HtmlExtractor
from .image_extractor import ImageExtractor
from .pdf_extractor import PdfExtractor
from .pptx_extractor import PptxExtractor
from .rtf_extractor import RtfExtractor
from .structured_extractor import JsonExtractor, XmlExtractor, YamlExtractor
from .text_extractor import TextExtractor
from .xlsx_extractor import XlsxExtractor



_EXTRACTORS: Dict[str, Type[BaseExtractor]] = {}


def _register(cls: Type[BaseExtractor]) -> None:
    for ext in cls.extensions:
        _EXTRACTORS[ext.lower()] = cls


for _cls in (
    # "Heavy" document formats (OCR-capable)
    PdfExtractor, DocxExtractor, PptxExtractor,
    XlsxExtractor, CsvExtractor, ImageExtractor,
    # Text-native formats (no OCR, just parse)
    TextExtractor, HtmlExtractor, RtfExtractor,
    JsonExtractor, XmlExtractor, YamlExtractor,
    EmailExtractor
):
    _register(_cls)


def get_extractor(path: str | Path, cfg) -> BaseExtractor:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in _EXTRACTORS:
        raise ValueError(f"No extractor registered for .{ext}")
    cls = _EXTRACTORS[ext]
    # Honour per-format enable flag from config.
    fmt_cfg = cfg.extractors.get(cls.format_name, None)
    if fmt_cfg is not None and not fmt_cfg.get("enabled", True):
        raise RuntimeError(f"Extractor '{cls.format_name}' is disabled in config")
    return cls(cfg)


def supported_extensions() -> list[str]:
    return sorted(_EXTRACTORS.keys())
