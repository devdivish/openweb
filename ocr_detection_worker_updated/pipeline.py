"""Adapter that lets the Redis/MinIO worker use the more accurate OCR pipeline.

This file intentionally keeps the old worker-facing class name
`SmartDocumentPipeline`, so the worker structure does not change.  Internally it
uses the extractor/orchestrator/OCR client from OCR_Pipeline_2.0, but builds the
configuration from the existing worker settings/env values.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import shutil
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from logger import logger
from ocr_pipeline.config import Config, _validate
from ocr_pipeline.ocr_client import OCRClient
from ocr_pipeline.orchestrator import Pipeline as AccuratePipeline


OCR_PROMPT = """Extract the full textual content of the document image.
Preserve layout, reading order, headings, lists and tables.
Render tables as Markdown. Keep the original language.
Output clean Markdown only."""


@dataclass
class ExtractionResult:
    text: str
    markdown: str
    page_count: int
    ocr_block_count: int
    text_block_count: int
    source_file: str
    dominant_script: str
    script_counts: Dict[str, int] = field(default_factory=dict)
    bucket_counts: Dict[str, int] = field(default_factory=dict)
    ocr_model_names_used: List[str] = field(default_factory=list)


class SmartDocumentPipeline:
    """Worker-compatible wrapper around OCR_Pipeline_2.0.

    The constructor keeps the original worker parameters to avoid touching the
    Redis/MinIO worker code. Only the OCR core is replaced.
    """

    def __init__(
        self,
        *,
        detector_endpoint: str,
        detector_model_name: str,
        detector_api_key: str = "EMPTY",
        english_endpoint: str,
        english_model_name: str,
        english_api_key: str = "EMPTY",
        non_english_endpoint: str,
        non_english_model_name: str,
        non_english_api_key: str = "EMPTY",
        libreoffice_path: str = "/usr/bin/libreoffice",
        batch_size: int = 4,
        enable_preprocessing: bool = True,
        preprocessor: Optional[object] = None,
        image_block_parallelism: int = 4,
        script_detect_parallelism: int = 4,
        english_ocr_parallelism: int = 2,
        non_english_ocr_parallelism: int = 2,
        **_: object,
    ) -> None:
        # Kept for backwards compatibility with the old worker config. The
        # accurate pipeline does not need a separate detector when script
        # routing is disabled; it OCRs using the configured dots-mocr backend.
        self.detector_endpoint = detector_endpoint
        self.detector_model_name = detector_model_name
        self.detector_api_key = detector_api_key
        self.english_endpoint = english_endpoint
        self.english_model_name = english_model_name
        self.english_api_key = english_api_key
        self.non_english_endpoint = non_english_endpoint
        self.non_english_model_name = non_english_model_name
        self.non_english_api_key = non_english_api_key
        self.libreoffice_path = libreoffice_path
        self.batch_size = batch_size
        self.enable_preprocessing = enable_preprocessing
        self.preprocessor = preprocessor
        self.image_block_parallelism = max(1, int(image_block_parallelism))
        self.script_detect_parallelism = max(1, int(script_detect_parallelism))
        self.english_ocr_parallelism = max(1, int(english_ocr_parallelism))
        self.non_english_ocr_parallelism = max(1, int(non_english_ocr_parallelism))

        if self.libreoffice_path:
            os.environ["LIBREOFFICE_PATH"] = self.libreoffice_path

        self._base_config = self._build_base_config()

    def process(self, file_path: str) -> ExtractionResult:
        """Run accurate OCR on one local file and return the old result shape."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        output_dir = path.parent / f".ocr_pipeline_output_{uuid.uuid4().hex}"
        cfg_data = deepcopy(self._base_config)
        cfg_data["output"]["directory"] = str(output_dir)
        cfg = Config(cfg_data)
        _validate(cfg)

        logger.info("[pipeline.py] accurate OCR start file=%s output_dir=%s", path.name, output_dir)
        try:
            md_path, sidecar_path = self._run_coro_sync(self._process_async(path, cfg))
            markdown = md_path.read_text(encoding="utf-8")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            result = self._to_extraction_result(path, markdown, sidecar)
            logger.info(
                "[pipeline.py] accurate OCR done file=%s pages=%s ocr_blocks=%s text_blocks=%s",
                path.name,
                result.page_count,
                result.ocr_block_count,
                result.text_block_count,
            )
            return result
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


    
    @staticmethod
    def _run_coro_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # keep original exception semantics
                result["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def _process_async(self, path: Path, cfg: Config) -> tuple[Path, Path]:
        async with OCRClient(cfg) as ocr:
            if ocr.health_check_enabled:
                try:
                    await ocr.probe_models()
                except Exception as exc:
                    logger.warning("[pipeline.py] OCR health probe failed, continuing: %s", exc)
            pipeline = AccuratePipeline(cfg)
            return await pipeline.process_one(path, ocr)

    def _build_base_config(self) -> Dict:
        models = self._parse_models(self.english_model_name)
        ocr_concurrency = max(
            1,
            self.english_ocr_parallelism + self.non_english_ocr_parallelism,
            self.batch_size,
        )

        return {
            "ocr": {
                "active_backend": "dots_ocr",
                "min_native_chars_per_page": 40,
                "render_dpi": 150,
                "min_image_side_px": 64,
                "max_image_pixels": 4_000_000,
                "max_image_long_side_px": 2400,
                "max_output_tokens": 4096,
                "temperature": 0.0,
                "request_timeout_s": 180,
                "max_retries": 3,
                "retry_backoff_s": 2.0,
                # 1 keeps behaviour simple for a single worker model name. If
                # ENGLISH_MODEL_NAME contains comma-separated aliases, this can
                # still rotate aliases while avoiding extra control complexity.
                "alias_hop_attempts": 1,
                "health_check": {
                    "enabled": False,
                    "probe_timeout_s": 10,
                    "probe_before_batch": False,
                    "min_healthy": 1,
                },
                "script_routing": {
                    "enabled": False,
                    "detection_mode": "both",
                    "probe_max_tokens": 512,
                    "fallback_backend": "dots_ocr",
                    "by_script": {},
                },
            },
            "ocr_backends": {
                "dots_ocr": {
                    "base_url": self.english_endpoint.rstrip("/"),
                    "models": models,
                    "api_key": self.english_api_key or "EMPTY",
                    "prompt": OCR_PROMPT,
                    "image_encoding": "base64",
                    "max_image_pixels": 4_000_000,
                    "max_image_long_side_px": 2400,
                }
            },
            "parallelism": {
                # The worker already controls document/attachment concurrency.
                "document_workers": 1,
                "page_workers": self.image_block_parallelism,
                "ocr_concurrency": ocr_concurrency,
            },
            "extractors": {
                "libreoffice": {"timeout_s": 300, "retries": 1},
                "pdf": {"enabled": True, "ocr_embedded_images": True},
                "docx": {
                    "enabled": True,
                    "ocr_embedded_images": True,
                    "legacy_doc_via_libreoffice": True,
                },
                "pptx": {
                    "enabled": True,
                    "ocr_embedded_images": True,
                    "legacy_ppt_via_libreoffice": True,
                },
                "xlsx": {
                    "enabled": True,
                    "max_rows_per_sheet": 5000,
                    "max_cols_per_sheet": 200,
                    "legacy_xls_via_pandas": True,
                },
                "csv": {"enabled": True, "delimiter": None, "encoding": None},
                "image": {"enabled": True, "formats": ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]},
                "text": {"enabled": True, "max_chars": 2_000_000},
                "html": {"enabled": True},
                "rtf": {"enabled": True, "legacy_via_libreoffice": True},
                "structured": {"enabled": True, "raw_on_parse_error": True, "indent": 2, "max_chars": 2_000_000},
                "email": {"enabled": True, "include_headers": True, "prefer_plain": True},
            },
            "output": {
                "directory": "./extracted",
                "markdown_filename": "{stem}.md",
                "sidecar_filename": "{stem}.meta.json",
                "include_page_headings": True,
                "save_extracted_images": False,
                "images_subdir": "{stem}_images",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                "per_document_log": False,
                "run_log_file": None,
            },
            "workspace": {"tmp_dir": None, "keep_tmp": False},
        }

    @staticmethod
    def _parse_models(model_name: str) -> List[str]:
        # Supports either "dots-mocr" or "dots-mocr,dots-mocr_1" without
        # adding a new config file.
        models = [m.strip() for m in str(model_name or "").split(",") if m.strip()]
        return models or ["dots-mocr"]

    def _to_extraction_result(self, path: Path, markdown: str, sidecar: Dict) -> ExtractionResult:
        pages = sidecar.get("pages") or []
        script_counts: Dict[str, int] = {}
        bucket_counts: Dict[str, int] = {}
        model_names = set(self._parse_models(self.english_model_name))
        ocr_blocks = 0
        text_blocks = 0

        for page in pages:
            for block in page.get("blocks") or []:
                btype = block.get("type")
                if btype == "ocr":
                    ocr_blocks += 1
                    meta = block.get("meta") or {}
                    script = meta.get("script_detected") or block.get("script_hint") or self._detect_script_from_text(markdown)
                    backend = meta.get("backend_used") or "dots_ocr"
                    bucket_counts[backend] = bucket_counts.get(backend, 0) + 1
                    script_counts[script] = script_counts.get(script, 0) + 1
                elif btype in {"text", "table"}:
                    text_blocks += 1
                    script = block.get("script_hint") or self._detect_script_from_text(markdown)
                    script_counts[script] = script_counts.get(script, 0) + 1
                    bucket_counts["native_text"] = bucket_counts.get("native_text", 0) + 1

        plain_text = self._markdown_to_plain_text(markdown)
        dominant_script = max(script_counts, key=script_counts.get) if script_counts else self._detect_script_from_text(plain_text)
        return ExtractionResult(
            text=plain_text,
            markdown=markdown,
            page_count=len(pages),
            ocr_block_count=ocr_blocks,
            text_block_count=text_blocks,
            source_file=str(path),
            dominant_script=dominant_script or "Latin",
            script_counts=script_counts,
            bucket_counts=bucket_counts,
            ocr_model_names_used=sorted(model_names),
        )

    @staticmethod
    def _markdown_to_plain_text(markdown: str) -> str:
        lines: List[str] = []
        for line in markdown.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if clean.startswith("#"):
                clean = clean.lstrip("#").strip()
            lines.append(clean)
        return "\n".join(lines).strip()

    @staticmethod
    def _detect_script_from_text(text: str) -> str:
        text = text or ""
        if re.search(r"[\u0600-\u06FF]", text):
            return "Arabic"
        if re.search(r"[\u0900-\u097F]", text):
            return "Devanagari"
        if re.search(r"[\u0980-\u09FF]", text):
            return "Bengali"
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "Han"
        return "Latin"
