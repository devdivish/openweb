from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    redis_url: str = Field(default='redis://:Redis@123@192.168.10.35:6379/0')
    input_stream_name: str = Field(default='tasks.extraction')
    output_stream_name: str = Field(default='tasks.embed')
    dead_letter_queue: str = Field(default='tasks.dlq')
    group_name: str = Field(default='extraction')
    worker_name: str = Field(default='extraction')

    minio_endpoint: str = Field(default='192.168.10.35:9198')
    minio_access_key: str = Field(default='user1')
    minio_secret: str = Field(default='VMware@123')
    file_uploads_bucket: str = Field(default='app-uploads')
    extracted_text_suffix: str = Field(default='extracted_text')

    libreoffice_path: str = Field(default='/usr/bin/libreoffice')
    batch_size: int = Field(default=4)
    enable_preprocessing: bool = Field(default=True)
    language_sample_images: int = Field(default=3)

    detector_vllm_url: str = Field(default='http://192.168.10.210:9000/v1')
    detector_model_name: str = Field(default='dots-mocr')
    detector_api_key: str = Field(default='EMPTY')

    english_vllm_url: str = Field(default='http://192.168.10.210:9000/v1')
    english_model_name: str = Field(default='dots-mocr_1')
    english_api_key: str = Field(default='EMPTY')

    # urdu_vllm_url: str = Field(default='http://192.168.10.210:9000/v1')
    # urdu_model_name: str = Field(default='dots-mocr_2')
    # urdu_api_key: str = Field(default='EMPTY')

    # bengali_vllm_url: str = Field(default='http://192.168.10.210:9000/v1')
    # bengali_model_name: str = Field(default='dots-mocr_1')
    # bengali_api_key: str = Field(default='EMPTY')

    non_english_vllm_url: str = Field(default='http://192.168.10.210:9000/v1')
    non_english_model_name: str = Field(default='dots-mocr')
    non_english_api_key: str = Field(default='EMPTY')

    # audio_model_url: str = "http://192.168.10.210:9000"
    # audio_model_name: str = "whisper=large-v3"
    # audio_model_headers : dict = Field(default_factory=dict)
    # audio_chunk_length_sec : int = 60

    tmp_dir: str = Field(default='/mnt/data/ocr_detection_worker_updated/ocr_detection_worker_updated/tmp')

    worker_job_parallelism: int = Field(default=10) # redis messages a worker can process concurrently
    attachment_parallelism: int = Field(default=5) # if one redis job has 5 attachments, no of attachments it can process at a time
    pipeline_thread_workers: int = Field(default=2)  # number of internal threads ocr pipeline can use
    image_block_parallelism: int = Field(default=2) #number of image blocks processed concurrently
    script_detect_parallelism: int = Field(default=2)
    english_ocr_parallelism: int = Field(default=1)
    non_english_ocr_parallelism: int = Field(default=1)
    job_lock_ttl_seconds: int = Field(default=7200)
    completed_job_ttl_seconds: int = Field(default=604800)
    
    # OCR_MODELS = {
    #     "english":{
    #         "endpoint": 'http://192.168.10.210:3046/v1',
    #         "model": "dots-mocr"
    #     },
    #     "non-english":{
    #         "endpoint": 'http://192.168.10.210:3046/v1',
    #         "model": "dots-mocr"
    #     }
    # }


settings = Settings()
