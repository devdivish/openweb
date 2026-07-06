"""Reconstruct a final Markdown document + JSON sidecar from an ExtractedDoc."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .extractors.base import Block, ExtractedDoc, Page


def render_markdown(doc: ExtractedDoc, include_page_headings: bool) -> str:
    parts: List[str] = []
    title = Path(doc.source_path).name
    parts.append(f"# {title}\n")
    for page in doc.pages:
        if include_page_headings:
            parts.append(f"\n### {page.label}\n")
        for block in page.blocks:
            md = _block_to_md(block)
            if md:
                parts.append(md)
    return "\n\n".join(p.rstrip() for p in parts if p.strip()) + "\n"


def render_sidecar(doc: ExtractedDoc, markdown_path: str,
                   timings: Dict[str, Any] | None = None) -> str:
    payload = {
        "source_path": doc.source_path,
        "format": doc.format,
        "markdown_path": markdown_path,
        "warnings": doc.warnings,
        "timings": timings or {},
        "pages": [
            {
                "index": p.index,
                "label": p.label,
                "rendered_whole": p.rendered_whole,
                "blocks": [_block_to_sidecar(b) for b in p.blocks],
            }
            for p in doc.pages
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------

def _block_to_md(block: Block) -> str:
    if block.type in ("text", "ocr", "table"):
        return (block.text or "").strip()
    if block.type == "image":
        # Un-OCR'd image (shouldn't happen after orchestrator run, but just in case)
        return f"<!-- image not OCR'd: {block.source} -->"
    return ""


def _block_to_sidecar(block: Block) -> Dict[str, Any]:
    return {
        "type": block.type,
        "source": block.source,
        "char_count": len(block.text or ""),
        "script_hint": block.script_hint,
        "meta": block.meta,     # includes backend_used + script_detected for OCR blocks
    }
