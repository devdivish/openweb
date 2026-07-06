"""HTTP extraction endpoint around the accurate OCR pipeline.

This exposes the SAME OCR core the Redis/MinIO worker uses
(`SmartDocumentPipeline`) as a simple synchronous HTTP service, so other
backends (e.g. the clone-openweb RAG backend) can extract text/markdown
from any supported file by POSTing it here — no Redis, no MinIO.

Run:
    pip install fastapi uvicorn python-multipart
    python extract_service.py
    # or: uvicorn extract_service:app --host 0.0.0.0 --port 8200

Endpoints:
    GET  /health           -> {"status": "ok", ...config snapshot...}
    POST /extract          -> multipart file field "file"
                              returns {text, markdown, page_count, ...}

The heavy work (extraction + VLM OCR) runs in a worker thread via
Starlette's run_in_threadpool, so the event loop stays responsive and
SmartDocumentPipeline's own asyncio.run() works cleanly (no running loop
in the worker thread).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from logger import logger
from settings import settings
from pipeline import SmartDocumentPipeline


def _build_pipeline() -> SmartDocumentPipeline:
    """Instantiate the pipeline exactly like engine/ocr_detection_engine.py."""
    return SmartDocumentPipeline(
        detector_endpoint=settings.detector_vllm_url,
        detector_model_name=settings.detector_model_name,
        detector_api_key=settings.detector_api_key,
        english_endpoint=settings.english_vllm_url,
        english_model_name=settings.english_model_name,
        english_api_key=settings.english_api_key,
        non_english_endpoint=settings.non_english_vllm_url,
        non_english_model_name=settings.non_english_model_name,
        non_english_api_key=settings.non_english_api_key,
        libreoffice_path=settings.libreoffice_path,
        batch_size=settings.batch_size,
        enable_preprocessing=settings.enable_preprocessing,
        image_block_parallelism=settings.image_block_parallelism,
        script_detect_parallelism=settings.script_detect_parallelism,
        english_ocr_parallelism=settings.english_ocr_parallelism,
        non_english_ocr_parallelism=settings.non_english_ocr_parallelism,
    )


app = FastAPI(title="OCR Extraction Service", version="1.0.0")

# Built once at import; the pipeline is stateless per-call and reusable.
PIPELINE = _build_pipeline()
TMP_ROOT = Path(settings.tmp_dir)
TMP_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "english_endpoint": settings.english_vllm_url,
        "english_model": settings.english_model_name,
        "non_english_endpoint": settings.non_english_vllm_url,
        "libreoffice_path": settings.libreoffice_path,
    }


def _run_extract(tmp_path: Path) -> dict:
    """Blocking extraction — called inside a worker thread."""
    result = PIPELINE.process(str(tmp_path))
    return {
        "text": result.text,
        "markdown": result.markdown,
        "page_count": result.page_count,
        "ocr_block_count": result.ocr_block_count,
        "text_block_count": result.text_block_count,
        "dominant_script": result.dominant_script,
        "script_counts": result.script_counts,
        "bucket_counts": result.bucket_counts,
        "ocr_model_names_used": result.ocr_model_names_used,
        "source_file": result.source_file,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> JSONResponse:
    """Extract text + markdown from one uploaded file using the OCR pipeline."""
    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix
    # Preserve the suffix: the extractor registry routes by file extension.
    tmp_path = TMP_ROOT / f"extract_{uuid.uuid4().hex}{suffix}"
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        tmp_path.write_bytes(contents)

        logger.info("[extract_service] extracting name=%s size=%d tmp=%s",
                    original_name, len(contents), tmp_path.name)
        try:
            payload = await run_in_threadpool(_run_extract, tmp_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # OCR/extraction failure
            logger.exception("[extract_service] extraction failed name=%s", original_name)
            raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}")

        payload["filename"] = original_name
        return JSONResponse(payload)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("EXTRACT_SERVICE_PORT", "8200"))
    host = os.getenv("EXTRACT_SERVICE_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
