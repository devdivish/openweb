"""RFC 822 email extractor — .eml files.

Uses the stdlib `email` package (no new dependencies). Emits a Markdown
document with the key headers followed by the body:

    **From:** ...
    **To:** ...
    **Subject:** ...
    **Date:** ...

    <body>

Body preference is configurable — `prefer_plain: true` (default) picks
text/plain first and falls back to text/html; `false` does the reverse.
text/html bodies are passed through the HTML walker so tags are stripped
and headings/lists/tables are preserved as Markdown.

Attachments are skipped for now — RAG embeddings for attachments should
come from ingesting those files separately via their own extractors.
"""
from __future__ import annotations

import logging
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import List

from .base import BaseExtractor, Block, ExtractedDoc, Page
from .html_extractor import _Walker, _strip_tags_fallback

log = logging.getLogger(__name__)


class EmailExtractor(BaseExtractor):
    extensions = ["eml"]
    format_name = "email"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)
        fmt_cfg = self.cfg.extractors.get("email", {}) or {}
        include_headers = bool(fmt_cfg.get("include_headers", True))
        prefer_plain = bool(fmt_cfg.get("prefer_plain", True))

        with path.open("rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        lines: List[str] = []
        if include_headers:
            for h in ("From", "To", "Cc", "Bcc", "Subject", "Date"):
                v = msg[h]
                if v:
                    lines.append(f"**{h}:** {v}")
            lines.append("")  # blank line between headers and body

        # Pick the best body part.
        pref = ("plain", "html") if prefer_plain else ("html", "plain")
        try:
            part = msg.get_body(preferencelist=pref)
        except Exception:
            part = None

        body = ""
        if part is not None:
            try:
                content = part.get_content()
            except Exception as e:
                content = ""
                log.warning("Failed to decode body of %s: %s", path, e)
            if part.get_content_type() == "text/html" and content:
                body = _html_to_text(content)
            else:
                body = content or ""

        lines.append(body.strip())
        text = "\n".join(lines).strip() + "\n"

        page = Page(index=0, label=f"Email: {path.name}")
        if text.strip():
            page.blocks.append(Block(type="text", text=text, source="email"))
        return ExtractedDoc(source_path=str(path), format="email", pages=[page])


def _html_to_text(html_content: str) -> str:
    walker = _Walker()
    try:
        walker.feed(html_content)
        walker.close()
        return walker.result()
    except Exception:
        return _strip_tags_fallback(html_content)
