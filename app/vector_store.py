from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.schemas import Source
from app.store import JsonStore

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class Chunk:
    id: str
    file_id: str
    filename: str
    text: str
    vector: list[float]
    index: int
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_id": self.file_id,
            "filename": self.filename,
            "text": self.text,
            "vector": self.vector,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(limit))
    norm_a = math.sqrt(sum(value * value for value in a[:limit]))
    norm_b = math.sqrt(sum(value * value for value in b[:limit]))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    def __init__(self, store: JsonStore):
        self.store = store

    def replace_file_chunks(self, file_id: str, chunks: list[Chunk]) -> None:
        self.store.replace_file_chunks(file_id, [chunk.to_dict() for chunk in chunks])

    def search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        file_ids: list[str] | None = None,
        relevance_threshold: float = 0.0,
        mode: str = "hybrid",
    ) -> list[Source]:
        allowed = set(file_ids or [])
        query_tokens = set(TOKEN_RE.findall(query_text.lower()))
        scored_by_id: dict[str, tuple[float, dict]] = {}
        if mode in {"hybrid", "vector"}:
            for chunk in self.store.chunks():
                if allowed and chunk["file_id"] not in allowed:
                    continue
                semantic_score = cosine(query_vector, chunk["vector"])
                chunk_tokens = set(TOKEN_RE.findall(chunk["text"].lower()))
                lexical_score = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
                score = (0.70 * semantic_score) + (0.20 * lexical_score)
                scored_by_id[chunk["id"]] = (score, chunk)

        if mode in {"hybrid", "keyword"} and hasattr(self.store, "search_text_chunks"):
            bm25_hits = self.store.search_text_chunks(query_text, top_k=max(top_k * 3, 10), file_ids=file_ids)
            total_hits = max(1, len(bm25_hits))
            for rank, hit in enumerate(bm25_hits):
                bm25_score = 1.0 - (rank / total_hits)
                existing_score, chunk = scored_by_id.get(hit["id"], (0.0, hit))
                keyword_weight = 1.0 if mode == "keyword" else 0.35
                scored_by_id[hit["id"]] = (existing_score + (keyword_weight * bm25_score), chunk)

        scored = [(score, chunk) for score, chunk in scored_by_id.values() if score >= relevance_threshold]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Source(
                file_id=chunk["file_id"],
                filename=chunk["filename"],
                chunk_id=chunk["id"],
                chunk_index=int(chunk.get("index", 0)),
                context_start_index=int(chunk.get("index", 0)),
                context_end_index=int(chunk.get("index", 0)),
                start_char=int(chunk.get("start_char", 0)),
                end_char=int(chunk.get("end_char", 0)),
                score=round(float(score), 5),
                text=chunk["text"],
            )
            for score, chunk in scored[:top_k]
        ]

    def full_context(self, file_ids: list[str] | None = None, max_chars: int = 24000) -> list[Source]:
        allowed = set(file_ids or [])
        grouped: dict[str, dict] = {}
        for chunk in sorted(self.store.chunks(), key=lambda item: (item["filename"], item["index"])):
            if allowed and chunk["file_id"] not in allowed:
                continue
            group = grouped.setdefault(
                chunk["file_id"],
                {
                    "file_id": chunk["file_id"],
                    "filename": chunk["filename"],
                    "chunk_id": f"{chunk['file_id']}:full",
                    "chunk_index": 0,
                    "context_start_index": 0,
                    "context_end_index": 0,
                    "start_char": int(chunk.get("start_char", 0)),
                    "end_char": int(chunk.get("end_char", 0)),
                    "score": 1.0,
                    "parts": [],
                },
            )
            group["parts"].append(chunk["text"])
            group["context_end_index"] = max(group["context_end_index"], int(chunk.get("index", 0)))
            group["end_char"] = max(group["end_char"], int(chunk.get("end_char", 0)))

        sources = []
        used = 0
        for group in grouped.values():
            remaining = max_chars - used
            if remaining <= 0:
                break
            text = "\n\n".join(group.pop("parts")).strip()
            text = text[:remaining]
            used += len(text)
            sources.append(Source(text=text, **group))
        return sources
