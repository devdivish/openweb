# OCR Detection Worker - Accurate OCR Pipeline Integration

This package keeps the original `ocr_detection_worker` Redis/MinIO worker flow, but replaces the internal OCR core with the more accurate extraction/OCR logic from `OCR_Pipeline_2.0_with_ES_parallel_multimodel`.

## What changed

- Kept the worker entrypoint and structure:
  - `main.py`
  - `engine/ocr_detection_engine.py`
  - Redis input stream handling
  - MinIO source-file read and Markdown write-back
  - output stream / DLQ flow
- Replaced the old simplified OCR pipeline behind the same class name:
  - `pipeline.py` now acts as an adapter around the OCR Pipeline 2.0 orchestrator.
- Added the accurate OCR package under:
  - `ocr_pipeline/`
- Removed Elasticsearch-driven execution from the integration path.
  - The worker does not use the standalone pipeline's ES ingestion code.
- Updated `requirements.txt` for the accurate extractors and async OCR client.
- Updated `run.py` as a local test runner that uses the same OCR core without Redis/MinIO.

## Runtime behavior

The worker still:

1. Reads a job from the configured Redis input stream.
2. Reads each attachment from MinIO.
3. Saves the attachment to a temporary local path.
4. Runs the OCR Pipeline 2.0 extraction/OCR logic.
5. Writes the generated Markdown file back to MinIO.
6. Sends the next Redis stream message using the existing worker flow.

## Configuration

The worker still reads configuration from `.env` / `settings.py`.

Important existing values used by the new OCR core:

```env
ENGLISH_VLLM_URL=http://192.168.10.210:9000/v1
ENGLISH_MODEL_NAME=dots-mocr
ENGLISH_API_KEY=EMPTY

NON_ENGLISH_VLLM_URL=http://192.168.10.210:9000/v1
NON_ENGLISH_MODEL_NAME=dots-mocr
NON_ENGLISH_API_KEY=EMPTY

LIBREOFFICE_PATH=/usr/bin/libreoffice
BATCH_SIZE=4
IMAGE_BLOCK_PARALLELISM=2
ENGLISH_OCR_PARALLELISM=1
NON_ENGLISH_OCR_PARALLELISM=1
```

`ENGLISH_MODEL_NAME` can also contain comma-separated model aliases if you run multiple llama-swap aliases, for example:

```env
ENGLISH_MODEL_NAME=dots-mocr,dots-mocr_1,dots-mocr_2,dots-mocr_3
```

## Run the worker

```bash
pip install -r requirements.txt
python main.py
```

## Local test without Redis/MinIO

```bash
python run.py /path/to/file.pdf --output ./extracted
```

This writes Markdown output locally and uses the same OCR core as the worker.

## Notes

- OCR Pipeline 2.0 image downscaling, retry handling, markdown rendering, PDF/Office/image extractors, and table rendering are now used by the worker.
- The adapter disables script routing by default to keep behavior simple and use the configured `dots-mocr` backend directly.
- If OCR processing fails, the exception is re-raised so the outer worker can send the failed job to the DLQ as originally intended.
