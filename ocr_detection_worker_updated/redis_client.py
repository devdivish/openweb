# import json
# from typing import Any, Dict

# import redis.asyncio as redis


# def serialise(value: Any) -> str | bytes | int | float:
#     if isinstance(value, (str, bytes, int, float)):
#         return value
#     return json.dumps(value, ensure_ascii=False)


# class RedisClient:
#     def __init__(self, redis_url: str, group: str, stream: str, worker: str, decode_responses: bool = True):
#         self.group = group
#         self.stream = stream
#         self.worker = worker
#         self.ttl = 600
#         self.redis_client = redis.from_url(redis_url, decode_responses=decode_responses)

#     async def test_redis_connection(self):
#         if await self.redis_client.ping():
#             print('Redis is connected!')
#         else:
#             print('Redis is not connected')

#     async def ensure_group(self) -> None:
#         try:
#             await self.redis_client.xgroup_create(
#                 name=self.stream,
#                 groupname=self.group,
#                 id='0-0',
#                 mkstream=True,
#             )
#         except redis.ResponseError as exc:
#             if 'BUSYGROUP' not in str(exc):
#                 raise RuntimeError(f'Error creating group: {exc}')

#     async def xreadgroup(self, streams: Dict, count: int = 50, block_ms: int = 2000):
#         return await self.redis_client.xreadgroup(
#             groupname=self.group,
#             consumername=self.worker,
#             streams=streams,
#             count=count,
#             block=block_ms,
#         )

#     async def ack(self, job_id: str) -> None:
#         await self.redis_client.xack(self.stream, self.group, job_id)

#     async def send_to_stream(self, stream: str, data: Dict):
#         safe_data = {k: serialise(v) for k, v in data.items()}
#         await self.redis_client.xadd(stream, safe_data)

#     async def set_record(self, key: str, value: str):
#         await self.redis_client.setex(key, self.ttl, value)

#     async def hset_mapping(self, key: str, mapping: Dict[str, Any]) -> None:
#         safe_mapping = {k: serialise(v) for k, v in mapping.items()}
#         await self.redis_client.hset(key, mapping=safe_mapping)

#     async def get_full_record(self, key):
#         return await self.redis_client.hgetall(key)

#     async def set_key(self, redis_key: str, key: str, value):
#         await self.redis_client.hset(redis_key, key, serialise(value))

#     async def close(self) -> None:
#         await self.redis_client.close()


import json
from typing import Any, Dict

import redis.asyncio as redis


def serialise(value: Any) -> str | bytes | int | float:
    if isinstance(value, (str, bytes, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False)


class RedisClient:
    def __init__(self, redis_url: str, group: str, stream: str, worker: str, decode_responses: bool = True):
        self.group = group
        self.stream = stream
        self.worker = worker
        self.ttl = 600
        self.redis_client = redis.from_url(redis_url, decode_responses=decode_responses)

    async def test_redis_connection(self):
        if await self.redis_client.ping():
            print('Redis is connected!')
        else:
            print('Redis is not connected')

    async def ensure_group(self) -> None:
        try:
            await self.redis_client.xgroup_create(
                name=self.stream,
                groupname=self.group,
                id='0-0',
                mkstream=True,
            )
        except redis.ResponseError as exc:
            if 'BUSYGROUP' not in str(exc):
                raise RuntimeError(f'Error creating group: {exc}')

    async def xreadgroup(self, streams: Dict, count: int = 50, block_ms: int = 2000):
        return await self.redis_client.xreadgroup(
            groupname=self.group,
            consumername=self.worker,
            streams=streams,
            count=count,
            block=block_ms,
        )

    async def ack(self, job_id: str) -> None:
        await self.redis_client.xack(self.stream, self.group, job_id)

    async def send_to_stream(self, stream: str, data: Dict):
        safe_data = {k: serialise(v) for k, v in data.items()}
        await self.redis_client.xadd(stream, safe_data)

    async def set_record(self, key: str, value: str):
        await self.redis_client.setex(key, self.ttl, value)

    async def set_value(self, key: str, value: Any, ttl_seconds: int | None = None):
        if ttl_seconds:
            await self.redis_client.set(key, serialise(value), ex=ttl_seconds)
        else:
            await self.redis_client.set(key, serialise(value))

    async def key_exists(self, key: str) -> bool:
        return bool(await self.redis_client.exists(key))

    async def acquire_lock(self, key: str, ttl_seconds: int = 3600) -> bool:
        return bool(await self.redis_client.set(key, '1', ex=ttl_seconds, nx=True))

    async def release_lock(self, key: str) -> None:
        await self.redis_client.delete(key)

    async def hset_mapping(self, key: str, mapping: Dict[str, Any]) -> None:
        safe_mapping = {k: serialise(v) for k, v in mapping.items()}
        await self.redis_client.hset(key, mapping=safe_mapping)

    async def get_full_record(self, key):
        return await self.redis_client.hgetall(key)

    async def set_key(self, redis_key: str, key: str, value):
        await self.redis_client.hset(redis_key, key, serialise(value))

    async def close(self) -> None:
        await self.redis_client.close()
