from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import httpx

from app.config import Settings


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalHashEmbedder(Embedder):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.settings.openai_base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={"model": self.settings.openai_embedding_model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


class OllamaEmbedder(Embedder):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120) as client:
            vectors = []
            for text in texts:
                response = await client.post(
                    f"{self.settings.ollama_base_url.rstrip('/')}/api/embeddings",
                    json={"model": self.settings.ollama_embedding_model, "prompt": text},
                )
                response.raise_for_status()
                vectors.append(response.json()["embedding"])
            return vectors


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(settings)
    if provider == "ollama":
        return OllamaEmbedder(settings)
    return LocalHashEmbedder(settings.embedding_dimensions)
