"""HTML / XHTML extractor.

Uses the stdlib `html.parser` to walk the DOM and emit Markdown-flavoured
text. Preserves headings (h1..h6 -> #..######), paragraphs, lists
(ordered + nested), tables, links, bold/italic, and preformatted blocks.
Ignores `<script>`, `<style>`, `<noscript>`, `<head>`, `<meta>`, `<link>`,
`<svg>`, and `<iframe>` bodies.

No external dependencies — beautifulsoup4 is deliberately avoided to
keep the offline-install footprint small.
"""
from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

from charset_normalizer import from_bytes

from .base import BaseExtractor, Block, ExtractedDoc, Page

log = logging.getLogger(__name__)

_SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link",
              "svg", "iframe", "form", "button"}
_BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer",
               "nav", "aside", "main", "blockquote", "figure",
               "figcaption", "dt", "dd", "dl"}
_HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###",
                 "h4": "####", "h5": "#####", "h6": "######"}


class _Walker(HTMLParser):
    """Streaming HTML -> Markdown-ish text walker."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self._skip_depth: int = 0
        self._in_pre: int = 0
        self._list_stack: List[str] = []
        self._ol_counters: List[int] = []
        self._in_table: int = 0
        self._table_rows: List[List[str]] = []
        self._current_row: Optional[List[str]] = None
        self._current_cell: Optional[List[str]] = None
        self._link_href: Optional[str] = None
        self._link_text: List[str] = []

    # -- open tag --------------------------------------------------------
    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag in _HEADING_TAGS:
            self._emit("\n\n" + _HEADING_TAGS[tag] + " ")
        elif tag in _BLOCK_TAGS:
            self._emit("\n\n")
        elif tag == "br":
            self._emit("\n")
        elif tag == "hr":
            self._emit("\n\n---\n\n")
        elif tag == "ul":
            self._list_stack.append("ul")
            self._emit("\n")
        elif tag == "ol":
            self._list_stack.append("ol")
            self._ol_counters.append(0)
            self._emit("\n")
        elif tag == "li":
            indent = "  " * max(0, len(self._list_stack) - 1)
            if self._list_stack and self._list_stack[-1] == "ol":
                self._ol_counters[-1] += 1
                self._emit(f"\n{indent}{self._ol_counters[-1]}. ")
            else:
                self._emit(f"\n{indent}- ")
        elif tag == "pre":
            self._in_pre += 1
            self._emit("\n\n```\n")
        elif tag == "code" and not self._in_pre:
            self._emit("`")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("_")
        elif tag == "a":
            self._link_href = dict(attrs).get("href")
            self._link_text = []
        elif tag == "table":
            self._in_table += 1
            self._table_rows = []
            self._emit("\n\n")
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in ("td", "th") and self._in_table:
            self._current_cell = []

    # -- close tag -------------------------------------------------------
    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return

        if tag in _HEADING_TAGS:
            self._emit("\n\n")
        elif tag in ("ul", "ol"):
            if self._list_stack:
                if self._list_stack[-1] == "ol" and self._ol_counters:
                    self._ol_counters.pop()
                self._list_stack.pop()
            self._emit("\n")
        elif tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)
            self._emit("\n```\n\n")
        elif tag == "code" and not self._in_pre:
            self._emit("`")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("_")
        elif tag == "a":
            href = self._link_href
            text = "".join(self._link_text).strip()
            if text and href:
                self._emit(f"[{text}]({href})")
            elif text:
                self._emit(text)
            self._link_href = None
            self._link_text = []
        elif tag == "table" and self._in_table:
            self._in_table = max(0, self._in_table - 1)
            if self._table_rows:
                self._emit_table()
                self._table_rows = []
        elif tag == "tr" and self._in_table and self._current_row is not None:
            self._table_rows.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._in_table:
            if self._current_cell is not None and self._current_row is not None:
                self._current_row.append(" ".join("".join(self._current_cell).split()).strip())
            self._current_cell = None

    # -- text content ----------------------------------------------------
    def handle_data(self, data: str):
        if self._skip_depth:
            return
        if self._current_cell is not None:
            self._current_cell.append(data)
            return
        if self._link_href is not None:
            self._link_text.append(data)
            return
        self._emit(data)

    # -- helpers ---------------------------------------------------------
    def _emit(self, s: str) -> None:
        self.out.append(s)

    def _emit_table(self) -> None:
        if not self._table_rows:
            return
        header = self._table_rows[0]
        body = self._table_rows[1:]
        if not header:
            return
        self.out.append("\n| " + " | ".join(header) + " |\n")
        self.out.append("|" + "|".join("---" for _ in header) + "|\n")
        for row in body:
            padded = (row + [""] * len(header))[: len(header)]
            self.out.append("| " + " | ".join(padded) + " |\n")
        self.out.append("\n")

    def result(self) -> str:
        text = "".join(self.out)
        # Collapse 3+ newlines and trailing whitespace per line.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"


class HtmlExtractor(BaseExtractor):
    extensions = ["html", "htm", "xhtml"]
    format_name = "html"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        raw = path.read_bytes()
        best = from_bytes(raw).best()
        enc = best.encoding if best else "utf-8"
        try:
            content = raw.decode(enc, errors="replace")
        except LookupError:
            content = raw.decode("utf-8", errors="replace")

        warnings: List[str] = []
        walker = _Walker()
        try:
            walker.feed(content)
            walker.close()
            text = walker.result()
        except Exception as e:
            log.warning("HTML parse error on %s: %s — falling back to regex strip", path, e)
            text = _strip_tags_fallback(content)
            warnings.append(f"html parser failed: {e}; used regex fallback")

        page = Page(index=0, label=f"Document: {path.name}")
        if text.strip():
            page.blocks.append(Block(type="text", text=text, source="html"))
        return ExtractedDoc(source_path=str(path), format="html",
                            pages=[page], warnings=warnings)


def _strip_tags_fallback(content: str) -> str:
    """Last-resort tag stripper for malformed HTML."""
    content = re.sub(r"<(script|style)[^>]*>.*?</\1>", "",
                     content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", content)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
