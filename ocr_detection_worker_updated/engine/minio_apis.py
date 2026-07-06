# from __future__ import annotations

# from io import BytesIO
# from pathlib import Path

# from minio import Minio
# from minio.error import S3Error

# from logger import logger


# def read_file_from_minio(mc: Minio, bucket: str, object_name: str) -> bytes | None:
#     try:
#         data = mc.get_object(bucket, object_name)
#         try:
#             return data.read()
#         finally:
#             data.close()
#             data.release_conn()
#     except S3Error as exc:
#         logger.error('Error reading object from MinIO bucket=%s object=%s error=%s', bucket, object_name, exc)
#         return None


# def write_bytes_to_local(file_bytes: bytes, dest_path: str) -> None:
#     path = Path(dest_path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_bytes(file_bytes)


# def write_text_to_minio(mc: Minio, bucket: str, object_name: str, text: str) -> None:
#     payload = text.encode('utf-8')
#     stream = BytesIO(payload)
#     mc.put_object(
#         bucket_name=bucket,
#         object_name=object_name,
#         data=stream,
#         length=len(payload),
#         content_type='text/plain; charset=utf-8',
#     )


from __future__ import annotations

from io import BytesIO
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from logger import logger


def read_file_from_minio(mc: Minio, bucket: str, object_name: str) -> bytes | None:
    try:
        data = mc.get_object(bucket, object_name)
        try:
            return data.read()
        finally:
            data.close()
            data.release_conn()
    except S3Error as exc:
        logger.error('Error reading object from MinIO bucket=%s object=%s error=%s', bucket, object_name, exc)
        return None


def write_bytes_to_local(file_bytes: bytes, dest_path: str) -> None:
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)


def write_text_to_minio(mc: Minio, bucket: str, object_name: str, text: str, content_type: str = 'text/plain; charset=utf-8') -> None:
    payload = text.encode('utf-8')
    stream = BytesIO(payload)
    mc.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=stream,
        length=len(payload),
        content_type=content_type,
    )
