"""Structured-data extractors — JSON, XML, YAML.

Each extractor parses the source file and emits a pretty-printed,
fenced code block so RAG chunkers can still index keys and values
naturally. If parsing fails (malformed input, partial file), we fall
back to the raw text by default so the doc is never silently dropped.

No new dependencies — stdlib + pyyaml (already in requirements).
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import yaml
from charset_normalizer import from_bytes

from .base import BaseExtractor, Block, ExtractedDoc, Page

log = logging.getLogger(__name__)


def _decode_file(path: Path) -> str:
    raw = path.read_bytes()
    best = from_bytes(raw).best()
    enc = best.encoding if best else "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _truncate(body: str, max_chars: int) -> Tuple[str, List[str]]:
    if len(body) > max_chars:
        return body[:max_chars] + "\n<!-- truncated -->", \
               [f"truncated: output exceeded max_chars={max_chars}"]
    return body, []


# ----------------------------------------------------------------------

class JsonExtractor(BaseExtractor):
    extensions = ["json"]
    format_name = "structured"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        content = _decode_file(path)
        fmt_cfg = self.cfg.extractors.get("structured", {}) or {}
        indent = int(fmt_cfg.get("indent", 2))
        max_chars = int(fmt_cfg.get("max_chars", 2_000_000))
        raw_on_err = bool(fmt_cfg.get("raw_on_parse_error", True))

        warnings: List[str] = []
        try:
            data = json.loads(content)
            pretty = json.dumps(data, indent=indent, ensure_ascii=False)
            body = f"```json\n{pretty}\n```"
        except Exception as e:
            if not raw_on_err:
                raise
            warnings.append(f"JSON parse failed: {e}; emitting raw content")
            body = f"```\n{content.strip()}\n```"

        body, trunc = _truncate(body, max_chars)
        warnings.extend(trunc)

        page = Page(index=0, label=f"Document: {path.name}")
        page.blocks.append(Block(type="text", text=body, source="json"))
        return ExtractedDoc(source_path=str(path), format="json",
                            pages=[page], warnings=warnings)


# ----------------------------------------------------------------------

class XmlExtractor(BaseExtractor):
    extensions = ["xml"]
    format_name = "structured"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        content = _decode_file(path)
        fmt_cfg = self.cfg.extractors.get("structured", {}) or {}
        max_chars = int(fmt_cfg.get("max_chars", 2_000_000))
        raw_on_err = bool(fmt_cfg.get("raw_on_parse_error", True))

        warnings: List[str] = []
        try:
            # Drop BOM if present so ElementTree accepts it.
            src = content.lstrip("\ufeff")
            root = ET.fromstring(src)
            # Pretty-print; ET.indent is 3.9+. If missing, ship as-is.
            try:
                ET.indent(root)
            except AttributeError:
                pass
            pretty = ET.tostring(root, encoding="unicode")
            body = f"```xml\n{pretty}\n```"
        except Exception as e:
            if not raw_on_err:
                raise
            warnings.append(f"XML parse failed: {e}; emitting raw content")
            body = f"```\n{content.strip()}\n```"

        body, trunc = _truncate(body, max_chars)
        warnings.extend(trunc)

        page = Page(index=0, label=f"Document: {path.name}")
        page.blocks.append(Block(type="text", text=body, source="xml"))
        return ExtractedDoc(source_path=str(path), format="xml",
                            pages=[page], warnings=warnings)


# ----------------------------------------------------------------------

class YamlExtractor(BaseExtractor):
    extensions = ["yml", "yaml"]
    format_name = "structured"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        content = _decode_file(path)
        fmt_cfg = self.cfg.extractors.get("structured", {}) or {}
        indent = int(fmt_cfg.get("indent", 2))
        max_chars = int(fmt_cfg.get("max_chars", 2_000_000))
        raw_on_err = bool(fmt_cfg.get("raw_on_parse_error", True))

        warnings: List[str] = []
        try:
            data = yaml.safe_load(content)
            pretty = yaml.dump(data, indent=indent, allow_unicode=True,
                               default_flow_style=False, sort_keys=False)
            body = f"```yaml\n{pretty}```"
        except Exception as e:
            if not raw_on_err:
                raise
            warnings.append(f"YAML parse failed: {e}; emitting raw content")
            body = f"```\n{content.strip()}\n```"

        body, trunc = _truncate(body, max_chars)
        warnings.extend(trunc)

        page = Page(index=0, label=f"Document: {path.name}")
        page.blocks.append(Block(type="text", text=body, source="yaml"))
        return ExtractedDoc(source_path=str(path), format="yaml",
                            pages=[page], warnings=warnings)
