"""DOCX extractor (python-docx).

Word documents have no intrinsic pages, so we emit ONE logical Page whose
blocks preserve the body order: paragraphs, tables, and inline/anchored
images (each image becomes an image block for OCR).

Legacy .doc binaries are converted to .docx via LibreOffice on demand.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List

from PIL import Image
from docx import Document
from docx.document import Document as _DocxDoc
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import BaseExtractor, Block, ExtractedDoc, Page
from ..libreoffice import LibreOfficeError, convert as lo_convert
from ..scripts import detect_script

log = logging.getLogger(__name__)


class DocxExtractor(BaseExtractor):
    extensions = ["docx", "doc"]
    format_name = "docx"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        src = path
        if path.suffix.lower() == ".doc":
            src = self._doc_to_docx(path)

        doc: _DocxDoc = Document(str(src))
        min_side = int(self.cfg.ocr.min_image_side_px)
        ocr_embedded = bool(self.cfg.extractors.docx.get("ocr_embedded_images", True))

        # First pass: collect all paragraph text to detect document-wide
        # script. .docx has no page concept so a document-level hint is
        # the best signal we have for routing image blocks.
        all_text = "\n".join(
            "".join(run.text for run in Paragraph(child, doc).runs)
            for child in doc.element.body.iterchildren()
            if child.tag == qn("w:p")
        )
        doc_script = detect_script(all_text)

        page = Page(index=0, label="Document")
        # Walk the body in document order; python-docx doesn't give us
        # paragraphs+tables+images together, so iterate over the XML children.
        for child in doc.element.body.iterchildren():
            tag = child.tag
            if tag == qn("w:p"):
                para = Paragraph(child, doc)
                md = self._paragraph_to_md(para)
                if md.strip():
                    page.blocks.append(Block(type="text", text=md, source="docx_paragraph"))
                if ocr_embedded:
                    for img in self._images_in_paragraph(doc, para, min_side):
                        page.blocks.append(
                            Block(type="image", image=img,
                                  source="docx_inline_image",
                                  meta={"w": img.width, "h": img.height},
                                  script_hint=doc_script)
                        )
            elif tag == qn("w:tbl"):
                tbl = Table(child, doc)
                md = self._table_to_md(tbl)
                if md.strip():
                    page.blocks.append(Block(type="table", text=md, source="docx_table"))

        # Also collect any floating images referenced elsewhere (headers,
        # shapes, drawings) via the document part rels.
        if ocr_embedded:
            for img in self._remaining_images(doc, min_side,
                                              already_seen=self._seen_blip_ids(doc)):
                page.blocks.append(
                    Block(type="image", image=img, source="docx_floating_image",
                          meta={"w": img.width, "h": img.height},
                          script_hint=doc_script)
                )

        return ExtractedDoc(source_path=str(path), format="docx", pages=[page])

    # ------------------------------------------------------------------
    # Paragraph / table -> Markdown
    # ------------------------------------------------------------------

    @staticmethod
    def _paragraph_to_md(p: Paragraph) -> str:
        style = (p.style.name or "").lower() if p.style is not None else ""
        text = "".join(run.text for run in p.runs)
        if not text:
            return ""
        if style.startswith("heading 1") or style == "title":
            return f"# {text}"
        if style.startswith("heading 2"):
            return f"## {text}"
        if style.startswith("heading 3"):
            return f"### {text}"
        if style.startswith("heading 4"):
            return f"#### {text}"
        if style.startswith("list") or style.startswith("bullet"):
            return f"- {text}"
        return text

    @staticmethod
    def _table_to_md(tbl: Table) -> str:
        rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells]
                for row in tbl.rows]
        if not rows:
            return ""
        header = rows[0]
        md = ["| " + " | ".join(header) + " |",
              "| " + " | ".join(["---"] * len(header)) + " |"]
        for r in rows[1:]:
            md.append("| " + " | ".join(r) + " |")
        return "\n".join(md)

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def _images_in_paragraph(self, doc, para: Paragraph, min_side: int) -> List[Image.Image]:
        imgs: List[Image.Image] = []
        for blip in para._element.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            if not rid:
                continue
            img = self._load_image_by_rid(doc, rid, min_side)
            if img is not None:
                imgs.append(img)
        return imgs

    @staticmethod
    def _seen_blip_ids(doc) -> set:
        seen = set()
        for blip in doc.element.body.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            if rid:
                seen.add(rid)
        return seen

    def _remaining_images(self, doc, min_side: int, already_seen: set) -> List[Image.Image]:
        """Images present in the package but not referenced inline in body."""
        imgs: List[Image.Image] = []
        part = doc.part
        for rid, rel in part.rels.items():
            if rid in already_seen:
                continue
            if "image" not in rel.reltype.lower():
                continue
            try:
                blob = rel.target_part.blob
                img = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception as e:
                log.debug("Skipping non-image relation %s: %s", rid, e)
                continue
            if img.width >= min_side and img.height >= min_side:
                imgs.append(img)
        return imgs

    @staticmethod
    def _load_image_by_rid(doc, rid: str, min_side: int):
        try:
            rel = doc.part.rels[rid]
            blob = rel.target_part.blob
            img = Image.open(io.BytesIO(blob)).convert("RGB")
        except Exception as e:
            log.debug("Failed to load image rid=%s: %s", rid, e)
            return None
        if img.width < min_side or img.height < min_side:
            return None
        return img

    # ------------------------------------------------------------------
    # .doc -> .docx via LibreOffice
    # ------------------------------------------------------------------

    def _doc_to_docx(self, path: Path) -> Path:
        if not self.cfg.extractors.docx.get("legacy_doc_via_libreoffice", True):
            raise RuntimeError(".doc support disabled via extractors.docx.legacy_doc_via_libreoffice")
        log.info("Converting legacy .doc -> .docx via LibreOffice: %s", path)
        lo_cfg = self.cfg.extractors.get("libreoffice", {}) or {}
        try:
            return lo_convert(
                path,
                target_ext="docx",
                timeout_s=float(lo_cfg.get("timeout_s", 300)),
                retries=int(lo_cfg.get("retries", 1)),
            )
        except LibreOfficeError as e:
            raise RuntimeError(str(e))
