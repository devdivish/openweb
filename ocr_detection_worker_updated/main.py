#<!-------------------Working code -------------------------------------->
'''
from __future__ import annotations

import asyncio
import time

from base_agent import BaseAgent
from engine.ocr_detection_engine import OCRDetectionEngine
from logger import logger
from redis_client import RedisClient
from settings import settings


class AIWorker:
    def __init__(self, redis_url: str, stream: str, group: str, worker: str, dlq: str, agent: BaseAgent):
        self.redis_url = redis_url
        self.stream = stream
        self.group = group
        self.worker = worker
        self.dlq = dlq
        self.agent = agent
        self.redis = None
        self.sem = asyncio.Semaphore(10)
        self.active_tasks: set[asyncio.Task] = set()

    async def initialize(self):
        self.redis = RedisClient(settings.redis_url, group=self.group, stream=self.stream, worker=self.worker)
        await self.redis.test_redis_connection()
        await self.redis.ensure_group()
        self.agent.setup_redis(self.redis)
        self.agent.setup_minio(settings.minio_endpoint, settings.minio_access_key, settings.minio_secret)

    async def _process_one(self, job_id, data):
        print(data)
        start = time.time()
        try:
            logger.info('Received job_id=%s', job_id)
            await self.agent.process(job_id, data)
            await self.redis.ack(job_id)
            logger.info('Job %s processed in %.3f seconds', job_id, time.time() - start)
        except Exception as exc:
            await self.redis.send_to_stream(self.dlq, data)
            await self.redis.ack(job_id)
            logger.error('FAILED job_id=%s pushed_to=%s error=%s', job_id, self.dlq, exc)
        finally:
            self.sem.release()

    async def _handle_batch(self, msgs):
        for job_id, data in msgs:
            await self.sem.acquire()
            task = asyncio.create_task(self._process_one(job_id, data))
            self.active_tasks.add(task)
            task.add_done_callback(lambda t: self.active_tasks.discard(t))

    async def run(self):
        logger.info('Starting worker=%s stream=%s group=%s', self.worker, self.stream, self.group)
        pending = await self.redis.xreadgroup({self.stream: '0'}, count=100)
        if pending:
            for _, msgs in pending:
                asyncio.create_task(self._handle_batch(msgs))

        try:
            while True:
                new_msgs = await self.redis.xreadgroup({self.stream: '>'})
                if new_msgs:
                    for _, msgs in new_msgs:
                        asyncio.create_task(self._handle_batch(msgs))
                await asyncio.sleep(0.05)
        finally:
            await self._shutdown()

    async def _shutdown(self):
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        if self.redis is not None:
            await self.redis.close()
        logger.info('Worker shutdown complete')


async def main():
    agent = OCRDetectionEngine()
    worker = AIWorker(
        settings.redis_url,
        settings.input_stream_name,
        settings.group_name,
        settings.worker_name,
        settings.dead_letter_queue,
        agent,
    )
    await worker.initialize()
    await worker.run()


if __name__ == '__main__':
    asyncio.run(main())

'''

#<!---------------------------Testing code--------------------------------->
from __future__ import annotations

import asyncio
import time

from base_agent import BaseAgent
from engine.ocr_detection_engine import OCRDetectionEngine
from logger import logger
from redis_client import RedisClient
from settings import settings


class AIWorker:
    def __init__(self, redis_url: str, stream: str, group: str, worker: str, dlq: str, agent: BaseAgent):
        self.redis_url = redis_url
        self.stream = stream
        self.group = group
        self.worker = worker
        self.dlq = dlq
        self.agent = agent
        self.redis = None
        self.sem = asyncio.Semaphore(settings.worker_job_parallelism)
        self.active_tasks: set[asyncio.Task] = set()

    async def initialize(self):
        self.redis = RedisClient(settings.redis_url, group=self.group, stream=self.stream, worker=self.worker)
        await self.redis.test_redis_connection()
        await self.redis.ensure_group()
        self.agent.setup_redis(self.redis)
        self.agent.setup_minio(settings.minio_endpoint, settings.minio_access_key, settings.minio_secret)
        logger.info(
            '[INFO] Worker concurrency configured job_parallelism=%s attachment_parallelism=%s pipeline_threads=%s',
            settings.worker_job_parallelism,
            settings.attachment_parallelism,
            settings.pipeline_thread_workers,
        )

    async def _process_one(self, job_id, data):
        start = time.time()
        try:
            logger.info('[INFO] Received job_id=%s', job_id)
            await self.agent.process(job_id, data)
            await self.redis.ack(job_id)
            logger.info('[INFO] Job %s processed in %.3f seconds', job_id, time.time() - start)
        except Exception as exc:
            await self.redis.send_to_stream(self.dlq, data)
            await self.redis.ack(job_id)
            logger.error('[INFO] FAILED job_id=%s pushed_to=%s error=%s', job_id, self.dlq, exc)
        finally:
            self.sem.release()

    async def _handle_batch(self, msgs):
        for job_id, data in msgs:
            await self.sem.acquire()
            task = asyncio.create_task(self._process_one(job_id, data))
            self.active_tasks.add(task)
            task.add_done_callback(lambda t: self.active_tasks.discard(t))

    async def run(self):
        logger.info('[INFO] Starting worker=%s stream=%s group=%s', self.worker, self.stream, self.group)
        pending = await self.redis.xreadgroup({self.stream: '0'}, count=100)
        if pending:
            for _, msgs in pending:
                asyncio.create_task(self._handle_batch(msgs))

        try:
            while True:
                new_msgs = await self.redis.xreadgroup({self.stream: '>'})
                if new_msgs:
                    for _, msgs in new_msgs:
                        asyncio.create_task(self._handle_batch(msgs))
                await asyncio.sleep(0.05)
        finally:
            await self._shutdown()

    async def _shutdown(self):
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        if self.redis is not None:
            await self.redis.close()
        logger.info('[INFO] Worker shutdown complete')


async def main():
    agent = OCRDetectionEngine()
    worker = AIWorker(
        settings.redis_url,
        settings.input_stream_name,
        settings.group_name,
        settings.worker_name,
        settings.dead_letter_queue,
        agent,
    )
    await worker.initialize()
    await worker.run()


if __name__ == '__main__':
    asyncio.run(main())
