
#<!----------------Testing --------------------------------------------------------->
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

from base_agent import BaseAgent
from engine.minio_apis import read_file_from_minio, write_bytes_to_local, write_text_to_minio
from logger import logger
from pipeline import SmartDocumentPipeline
from settings import settings
import hashlib
import os
from .cleaning import clean_ocr_markdown_text

OCR_CACHE_VERSION = 'native-text-preferred-v2'


def validate_task(fields: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any] | None]:
    data = fields.get('data')
    if not data:
        return False, 'Missing data field', None
    try:
        payload = json.loads(data)
    except Exception:
        return False, 'data field is not valid JSON', None
    if not payload.get('user_id'):
        return False, 'user_id is required', None
    attachments = payload.get('attachments')
    if not isinstance(attachments, list) or not attachments:
        return False, 'attachments must be a non-empty list', None
    return True, '', payload

class OCRDetectionEngine(BaseAgent):
    def __init__(self) -> None:
        self.pipeline = SmartDocumentPipeline(
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
        self.tmp_root = Path(settings.tmp_dir)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.attachment_sem = asyncio.Semaphore(settings.attachment_parallelism)
        self.pipeline_executor = ThreadPoolExecutor(max_workers=settings.pipeline_thread_workers)

    async def process(self, job_id: str, data: Dict[str, Any]):
        logger.info('[INFO] Starting process job_id=%s', job_id)
        is_valid, error, payload = validate_task(data)
        if not is_valid:
            raise RuntimeError(f'Payload validation failed: {error}')
        # print(payload)
        logger.info(
            '[INFO] Validated payload job_id=%s op=%s user_id=%s attachment_count=%s',
            job_id,
            payload.get('op'),
            payload.get('user_id'),
            len(payload.get('attachments', [])),
        )

        message_id = payload.get('message_id')
        task_key = f'task:{payload.get('message_id')}'

        await self.redis_client.set_key(task_key, 'status', 'Extracting')
        await self.redis_client.set_key(task_key, 'ui', f'Extracting files')

        results: List[Tuple[int, Dict[str, Any]]] = []
        task_tmp = self.tmp_root / str(uuid.uuid4())
        task_tmp.mkdir(parents=True, exist_ok=True)

        try:
            tasks = [
                asyncio.create_task(self._process_attachment_ordered(idx, payload, attachment, task_tmp))
                for idx, attachment in enumerate(payload['attachments'])
            ]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda item: item[0])
            ordered_results = [item[1] for item in results]
            print(ordered_results)

            output_payload = {
                'op': 'ocr_completed',
                'source_op': payload.get('op'),
                'message_id': message_id,
                'chat_id': payload.get('chat_id'),
                'user_id': payload.get('user_id'),
                'created_at': payload.get('created_at'),
                'attachments': ordered_results,
            }
            # logger.info(output_payload)
            logger.info('[INFO] Sending result to stream=%s job_id=%s', settings.output_stream_name, job_id)
    
            await self.redis_client.set_key(task_key, 'ui', '')       
            print("here")
            await self.redis_client.set_key(task_key, 'ui_detailed', '')       
            await self.redis_client.set_key(task_key, 'status', 'extraction_success')   

            await self.redis_client.send_to_stream(
            settings.output_stream_name,
                {'data': json.dumps(payload, ensure_ascii=False)},   
            )  
        
            shutil.rmtree(task_tmp, ignore_errors=True)

            logger.info('[INFO] Completed process job_id=%s', job_id)
        except Exception as e:
            # await self.redis_client.set_key(task_key, 'status', 'failed')
            await self.redis_client.set_key(task_key, 'status', 'failed') 
            # await self.redis_client.set_key(task_key, 'error', f'Extracting file failed: {str(e)}')
            await self.redis_client.set_key(task_key, 'ui_detailed', f'{str(e)}')
            await self.redis_client.set_key(task_key, 'error_reason', f'{str(e)}')
            await self.redis_client.set_key(task_key, 'current_stage', 'file_extractor')
            shutil.rmtree(task_tmp, ignore_errors=True)
            logger.info('[FAILED] Extraction process job_id=%s and exception = %s', job_id, e)
            raise RuntimeError(e)
            

    async def _process_attachment_ordered(
        self,
        index: int,
        payload: Dict[str, Any],
        attachment: Dict[str, Any],
        task_tmp: Path,
    ) -> Tuple[int, Dict[str, Any]]:
        async with self.attachment_sem:
            result = await self._process_attachment(payload, attachment, task_tmp)
            return index, result

    async def _run_pipeline(self, local_path: Path):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pipeline_executor, self.pipeline.process, str(local_path))

    async def _process_attachment(self, payload: Dict[str, Any], attachment: Dict[str, Any], task_tmp: Path) -> Dict[str, Any]:
            attachment_id = attachment.get('attachment_id')
            if attachment_id.startswith("temp_"):
                raise RuntimeError(f"Extraction Failed:{attachment.get('title')}. Invalid attachment id: {attachment_id}")
            title = attachment.get('title')   
            if not attachment_id or not title:
                raise RuntimeError(f'Extraction Failed: {attachment.get('title')} attachment_id/title missing in attachment: {attachment}')

            await self.redis_client.set_key(f'task:{payload["message_id"]}', 'ui_detailed', f'Extracting file {title}')
            source_object = f"{payload['user_id']}/{attachment_id}/{title}"
            logger.info('[INFO] Reading source from MinIO bucket=%s object=%s', settings.file_uploads_bucket, source_object)
            await self.redis_client.set_key(f'file_upload:{attachment_id}', 'status', 'ocr_processing')

            file_bytes = read_file_from_minio(self.mc, settings.file_uploads_bucket, source_object)
            
            if file_bytes is None:
                raise RuntimeError(f'Failed to read source object from MinIO: {source_object}') 

            # #Checking in cache
            hash_function = hashlib.new('sha256')
            hash_function.update(OCR_CACHE_VERSION.encode('utf-8'))
            hash_function.update(b'\0')
            hash_function.update(file_bytes)
            file_hash = hash_function.hexdigest()
            file_name=f"{file_hash}.md"
            cache_path="/mnt/data/OCR_cache"
            hash_path=os.path.join(cache_path,file_name)
            logger.info('[INFO] Hash function generated=%s',hash_function)
            #check if this file hash exisits--------------

            if os.path.exists(hash_path):
                logger.info('[CACHE] Found exisiting file in cache path=%s',hash_path)
                with open(hash_path,'r', encoding='utf-8') as f:
                    content=f.read()
                extraction_result= {"text":content,
                        "markdown":content
                }
                output_name = Path(title).stem + '.md'
                markdown_object = f"{payload['user_id']}/{attachment_id}/{settings.extracted_text_suffix}/{output_name}"
                logger.info('[INFO] Writing markdown to MinIO object=%s', markdown_object)
                write_text_to_minio(
                    self.mc,
                    settings.file_uploads_bucket,
                    markdown_object,
                    extraction_result["markdown"],
                    content_type='text/markdown; charset=utf-8',
                )

                logger.info(
                    '[INFO] OCR completed attachment_id=%s markdown_object=%s',
                    attachment_id,
                    markdown_object,
                )
                await self.redis_client.set_key(f'file_upload:{attachment_id}', 'status', 'ocr_completed')
                return {
                    'attachment_id': attachment_id,
                    'title': title
                }
            else:
                logger.info('[INFO] Read %s bytes from MinIO for attachment_id=%s', len(file_bytes), attachment_id)
                local_path = task_tmp / title
                write_bytes_to_local(file_bytes, str(local_path))
                logger.info('[INFO] Saved local temp file=%s', local_path)
                import zipfile
                logger.info(
                    "[INFO] Saved local temp file= %s exists=%s size=%s suffix=%s",
                    local_path,  
                    local_path.exists(),
                    local_path.stat().st_size if local_path.exists() else -1,
                    local_path.suffix.lower(),
                )

                if not local_path.exists():
                    logger.info("[WARNING] local temp file not created:%s", local_path)
                if local_path.exists() and local_path.stat().st_size == 0:
                    logger.info("[WARNING] local temp file is empty:%s", local_path)
                if local_path.suffix.lower() in {".docx",".pptx"} and not zipfile.is_zipfile(local_path):
                    logger.info(f"[WARNING] file extension says {local_path.suffix.lower()} but file is not a valid offi   ce package: {local_path}")
                
                logger.info('[INFO] Starting pipeline for attachment_id=%s', attachment_id)
            
                try:
                    extraction_result = await self._run_pipeline(local_path)
                except Exception as exc:
                    logger.error(
                        "[ERROR] Pipeline error for file %s (attachment_id=%s): %s",
                        title,
                        attachment_id,
                        exc,
                        exc_info=True,
                    )
                    raise RuntimeError(f'Extracting file failed: {title}. Please try to upload the file again.') from exc

                cleaned_markdown = clean_ocr_markdown_text(extraction_result.markdown)
                    
                logger.info(
                    '[INFO] Pipeline finished attachment_id=%s',
                    attachment_id,
                    
                )
                
                cache_path="/mnt/data/OCR_cache"
                hash_path=os.path.join(cache_path,file_name)
                os.makedirs(os.path.dirname(hash_path),exist_ok=True)
                with open(hash_path,'w', encoding='utf-8') as f:
                    f.write(cleaned_markdown)
                logger.info("[CACHE] New markdown content added to cache path :%s",hash_path)

                output_name = Path(title).stem + '.md'
                markdown_object = f"{payload['user_id']}/{attachment_id}/{settings.extracted_text_suffix}/{output_name}"
                logger.info('[INFO] Writing markdown to MinIO object=%s', markdown_object)
                write_text_to_minio(
                    self.mc,
                    settings.file_uploads_bucket,
                    markdown_object,
                    cleaned_markdown,
                    content_type='text/markdown; charset=utf-8',
                )

                logger.info(
                    '[INFO] OCR completed attachment_id=%s markdown_object=%s',
                    attachment_id,
                    markdown_object,
                )
                await self.redis_client.set_key(f'file_upload:{attachment_id}', 'status', 'ocr_completed')

                return {
                    'attachment_id': attachment_id,
                    'title': title
                }
            
