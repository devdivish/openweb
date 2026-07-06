"""CSV / TSV extractor.

Auto-detects encoding and delimiter unless overridden in config. Very
large CSVs are truncated to `max_rows_per_sheet` rows for the markdown
table (RAG workflows rarely need the full raw sheet in context).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from charset_normalizer import from_bytes

from .base import BaseExtractor, Block, ExtractedDoc, Page

log = logging.getLogger(__name__)


class CsvExtractor(BaseExtractor):
    extensions = ["csv", "tsv"]
    format_name = "csv"

    def extract(self, path: str | Path) -> ExtractedDoc:
        path = Path(path)

        encoding = self.cfg.extractors.csv.get("encoding") or self._detect_encoding(path)
        delim: Optional[str] = self.cfg.extractors.csv.get("delimiter")
        if delim is None:
            delim = self._detect_delimiter(path, encoding)
        # Fall back to the xlsx row/col caps for consistency.
        max_rows = int(self.cfg.extractors.xlsx.get("max_rows_per_sheet", 5000))

        df = pd.read_csv(str(path), sep=delim, encoding=encoding,
                         dtype=str, keep_default_na=False,
                         nrows=max_rows, on_bad_lines="skip")

        page = Page(index=0, label="Table")
        if not df.empty:
            md = df.to_markdown(index=False)
            page.blocks.append(Block(type="table", text=md, source="csv"))
        return ExtractedDoc(source_path=str(path), format="csv", pages=[page])

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        with path.open("rb") as f:
            raw = f.read(64 * 1024)
        best = from_bytes(raw).best()
        enc = best.encoding if best else "utf-8"
        log.debug("Detected encoding %s for %s", enc, path)
        return enc

    @staticmethod
    def _detect_delimiter(path: Path, encoding: str) -> str:
        try:
            with path.open("r", encoding=encoding, errors="replace") as f:
                sample = f.read(8192)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            return ","
