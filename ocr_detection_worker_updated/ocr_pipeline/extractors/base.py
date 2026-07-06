"""Common extractor types.

Every extractor produces an `ExtractedDoc` which is a list of `Page`s, and
each page is a list of `Block`s. A block is either:
  - type="text"  : already-usable native text (Markdown)
  - type="image" : raw PIL image awaiting OCR
  - type="ocr"   : text produced by OCR (filled in by the orchestrator)
  - type="table" : pre-rendered Markdown table

The orchestrator walks the document, dispatches image blocks to the OCR
client, then the formatter serialises the final Markdown while preserving
the original block order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional

from PIL import Image


BlockType = Literal["text", "image", "ocr", "table"]


@dataclass
class Block:
    type: BlockType
    text: Optional[str] = None
    image: Optional[Image.Image] = None   # for type="image"
    # Bookkeeping: where the block came from (used for logs & sidecar).
    source: str = ""
    # When an image block is converted to an ocr block we keep the original
    # dimensions for the sidecar.
    meta: dict = field(default_factory=dict)
    # Dominant Unicode script of nearby native text (e.g. "Arabic",
    # "Devanagari", "Latin"). Populated by extractors for image blocks
    # when context is available; used by the orchestrator to pick which
    # OCR backend to route the image to. None = unknown -> fallback.
    script_hint: Optional[str] = None


@dataclass
class Page:
    index: int                            # 0-based
    label: str                            # e.g. "Page 1", "Slide 3", "Sheet: Sales"
    blocks: List[Block] = field(default_factory=list)
    # When True, the whole page was rendered and sent to OCR as one image
    # (native text extraction was insufficient). In that case `blocks`
    # holds a single image block that the orchestrator will replace.
    rendered_whole: bool = False


@dataclass
class ExtractedDoc:
    source_path: str
    format: str                           # "pdf", "docx", ...
    pages: List[Page] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BaseExtractor:
    """Subclasses implement `.extract()`.

    Extractors must NOT call the OCR client themselves — they only produce
    image blocks. Centralising OCR in the orchestrator lets us share one
    semaphore / batch across the whole document and all its pages.
    """

    #: list of lower-case extensions handled (without the dot)
    extensions: List[str] = []
    #: logical format name written into the sidecar
    format_name: str = "unknown"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def extract(self, path: str | Path) -> ExtractedDoc:  # pragma: no cover
        raise NotImplementedError
