from __future__ import annotations

from abc import ABC, abstractmethod

from minio import Minio


class BaseAgent(ABC):
    def setup_redis(self, redis_client) -> None:
        self.redis_client = redis_client

    def setup_minio(self, minio_endpoint: str, minio_access_key: str, minio_password: str) -> None:
        self.mc = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_password,
            secure=False,
        )

    @abstractmethod
    async def process(self, job_id: str, data: dict):
        raise NotImplementedError
