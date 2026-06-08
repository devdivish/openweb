from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from app.schemas import ChatSession, FileChunk, FileRecord, KnowledgeBase, Skill


class JsonStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"files": {}, "chats": {}, "chunks": [], "knowledge": {}, "skills": {}})

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(self.path)

    def data(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    def upsert_file(self, record: FileRecord) -> None:
        with self._lock:
            data = self._read()
            data["files"][record.id] = record.model_dump(mode="json")
            self._write(data)

    def list_files(self) -> list[FileRecord]:
        with self._lock:
            data = self._read()
            adapter = TypeAdapter(list[FileRecord])
            return adapter.validate_python(list(data["files"].values()))

    def get_file(self, file_id: str) -> FileRecord | None:
        with self._lock:
            raw = self._read()["files"].get(file_id)
            return FileRecord.model_validate(raw) if raw else None

    def replace_file_chunks(self, file_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._lock:
            data = self._read()
            data["chunks"] = [chunk for chunk in data["chunks"] if chunk["file_id"] != file_id]
            data["chunks"].extend(chunks)
            self._write(data)

    def chunks(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read()["chunks"])

    def file_chunks(self, file_id: str) -> list[FileChunk]:
        chunks = [
            FileChunk(
                id=chunk["id"],
                file_id=chunk["file_id"],
                filename=chunk["filename"],
                index=int(chunk.get("index", 0)),
                start_char=int(chunk.get("start_char", 0)),
                end_char=int(chunk.get("end_char", 0)),
                text=chunk["text"],
                text_chars=len(chunk["text"]),
            )
            for chunk in self.chunks()
            if chunk["file_id"] == file_id
        ]
        return sorted(chunks, key=lambda chunk: chunk.index)

    def upsert_chat(self, chat: ChatSession) -> None:
        with self._lock:
            data = self._read()
            data["chats"][chat.id] = chat.model_dump(mode="json")
            self._write(data)

    def list_chats(self) -> list[ChatSession]:
        with self._lock:
            data = self._read()
            adapter = TypeAdapter(list[ChatSession])
            return adapter.validate_python(list(data["chats"].values()))

    def get_chat(self, chat_id: str) -> ChatSession | None:
        with self._lock:
            raw = self._read()["chats"].get(chat_id)
            return ChatSession.model_validate(raw) if raw else None

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = chat_id in data["chats"]
            data["chats"].pop(chat_id, None)
            self._write(data)
            return existed

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = file_id in data["files"]
            data["files"].pop(file_id, None)
            data["chunks"] = [chunk for chunk in data["chunks"] if chunk["file_id"] != file_id]
            for knowledge in data.setdefault("knowledge", {}).values():
                knowledge["file_ids"] = [item for item in knowledge.get("file_ids", []) if item != file_id]
            self._write(data)
            return existed

    def create_knowledge(self, name: str, description: str = "", file_ids: list[str] | None = None) -> KnowledgeBase:
        with self._lock:
            data = self._read()
            knowledge = KnowledgeBase(
                id=str(uuid4()),
                name=name,
                description=description,
                file_ids=list(dict.fromkeys(file_ids or [])),
            )
            data.setdefault("knowledge", {})[knowledge.id] = knowledge.model_dump(mode="json")
            self._write(data)
            return knowledge

    def list_knowledge(self) -> list[KnowledgeBase]:
        with self._lock:
            data = self._read()
            adapter = TypeAdapter(list[KnowledgeBase])
            return adapter.validate_python(list(data.setdefault("knowledge", {}).values()))

    def get_knowledge(self, knowledge_id: str) -> KnowledgeBase | None:
        with self._lock:
            raw = self._read().setdefault("knowledge", {}).get(knowledge_id)
            return KnowledgeBase.model_validate(raw) if raw else None

    def delete_knowledge(self, knowledge_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = knowledge_id in data.setdefault("knowledge", {})
            data["knowledge"].pop(knowledge_id, None)
            self._write(data)
            return existed

    def set_knowledge_files(self, knowledge_id: str, file_ids: list[str]) -> KnowledgeBase | None:
        with self._lock:
            data = self._read()
            raw = data.setdefault("knowledge", {}).get(knowledge_id)
            if not raw:
                return None
            raw["file_ids"] = list(dict.fromkeys(file_ids))
            raw["updated_at"] = datetime.now(UTC).isoformat()
            self._write(data)
            return KnowledgeBase.model_validate(raw)

    def get_knowledge_file_ids(self, knowledge_ids: list[str]) -> list[str]:
        file_ids = []
        for knowledge_id in knowledge_ids:
            knowledge = self.get_knowledge(knowledge_id)
            if knowledge:
                file_ids.extend(knowledge.file_ids)
        return list(dict.fromkeys(file_ids))

    def create_skill(
        self,
        name: str,
        instruction: str,
        description: str = "",
        triggers: list[str] | None = None,
        tool_ids: list[str] | None = None,
        enabled: bool = True,
    ) -> Skill:
        with self._lock:
            data = self._read()
            skill = Skill(
                id=str(uuid4()),
                name=name,
                description=description,
                instruction=instruction,
                triggers=list(dict.fromkeys(triggers or [])),
                tool_ids=list(dict.fromkeys(tool_ids or [])),
                enabled=enabled,
            )
            data.setdefault("skills", {})[skill.id] = skill.model_dump(mode="json")
            self._write(data)
            return skill

    def list_skills(self) -> list[Skill]:
        with self._lock:
            data = self._read()
            adapter = TypeAdapter(list[Skill])
            return adapter.validate_python(list(data.setdefault("skills", {}).values()))

    def get_skill(self, skill_id: str) -> Skill | None:
        with self._lock:
            raw = self._read().setdefault("skills", {}).get(skill_id)
            return Skill.model_validate(raw) if raw else None

    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> Skill | None:
        with self._lock:
            data = self._read()
            raw = data.setdefault("skills", {}).get(skill_id)
            if not raw:
                return None
            for key, value in updates.items():
                if value is not None:
                    raw[key] = value
            raw["updated_at"] = datetime.now(UTC).isoformat()
            self._write(data)
            return Skill.model_validate(raw)

    def delete_skill(self, skill_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = skill_id in data.setdefault("skills", {})
            data["skills"].pop(skill_id, None)
            self._write(data)
            return existed


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS files (
                        id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        file_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        vector TEXT NOT NULL,
                        start_char INTEGER NOT NULL DEFAULT 0,
                        end_char INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
                    CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename);
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        file_id UNINDEXED,
                        filename UNINDEXED,
                        text,
                        tokenize='unicode61'
                    );
                    CREATE TABLE IF NOT EXISTS chats (
                        id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS knowledge_files (
                        knowledge_id TEXT NOT NULL,
                        file_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(knowledge_id, file_id),
                        FOREIGN KEY(knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_knowledge_files_file_id ON knowledge_files(file_id);
                    CREATE TABLE IF NOT EXISTS skills (
                        id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    INSERT INTO chunks_fts(chunk_id, file_id, filename, text)
                    SELECT id, file_id, filename, text
                    FROM chunks
                    WHERE id NOT IN (SELECT chunk_id FROM chunks_fts);
                    """
                )
                _ensure_column(conn, "chunks", "start_char", "INTEGER NOT NULL DEFAULT 0")
                _ensure_column(conn, "chunks", "end_char", "INTEGER NOT NULL DEFAULT 0")
                conn.commit()
            finally:
                conn.close()

    def data(self) -> dict[str, Any]:
        return {
            "files": {item.id: item.model_dump(mode="json") for item in self.list_files()},
            "chats": {item.id: item.model_dump(mode="json") for item in self.list_chats()},
            "chunks": self.chunks(),
            "knowledge": {item.id: item.model_dump(mode="json") for item in self.list_knowledge()},
            "skills": {item.id: item.model_dump(mode="json") for item in self.list_skills()},
        }

    def upsert_file(self, record: FileRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO files(id, payload, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at
                    """,
                    (record.id, record.model_dump_json(), record.created_at.isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

    def list_files(self) -> list[FileRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT payload FROM files ORDER BY created_at DESC").fetchall()
            finally:
                conn.close()
        return [FileRecord.model_validate_json(row["payload"]) for row in rows]

    def get_file(self, file_id: str) -> FileRecord | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT payload FROM files WHERE id = ?", (file_id,)).fetchone()
            finally:
                conn.close()
        return FileRecord.model_validate_json(row["payload"]) if row else None

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT id FROM files WHERE id = ?", (file_id,)).fetchone()
                conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM chunks_fts WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM knowledge_files WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                conn.commit()
                return row is not None
            finally:
                conn.close()

    def replace_file_chunks(self, file_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                conn.execute("DELETE FROM chunks_fts WHERE file_id = ?", (file_id,))
                conn.executemany(
                    """
                    INSERT INTO chunks(id, file_id, filename, chunk_index, text, vector, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk["id"],
                            chunk["file_id"],
                            chunk["filename"],
                            int(chunk["index"]),
                            chunk["text"],
                            json.dumps(chunk["vector"]),
                            int(chunk.get("start_char", 0)),
                            int(chunk.get("end_char", 0)),
                        )
                        for chunk in chunks
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO chunks_fts(chunk_id, file_id, filename, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(chunk["id"], chunk["file_id"], chunk["filename"], chunk["text"]) for chunk in chunks],
                )
                conn.commit()
            finally:
                conn.close()

    def chunks(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, file_id, filename, chunk_index, text, vector, start_char, end_char FROM chunks ORDER BY filename, chunk_index"
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "id": row["id"],
                "file_id": row["file_id"],
                "filename": row["filename"],
                "index": row["chunk_index"],
                "text": row["text"],
                "vector": json.loads(row["vector"]),
                "start_char": row["start_char"],
                "end_char": row["end_char"],
            }
            for row in rows
        ]

    def file_chunks(self, file_id: str) -> list[FileChunk]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, file_id, filename, chunk_index, text, start_char, end_char
                    FROM chunks
                    WHERE file_id = ?
                    ORDER BY chunk_index ASC
                    """,
                    (file_id,),
                ).fetchall()
            finally:
                conn.close()
        return [
            FileChunk(
                id=row["id"],
                file_id=row["file_id"],
                filename=row["filename"],
                index=row["chunk_index"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                text=row["text"],
                text_chars=len(row["text"]),
            )
            for row in rows
        ]

    def search_text_chunks(
        self,
        query: str,
        top_k: int,
        file_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        match_query = make_fts_query(query)
        if not match_query:
            return []

        allowed = list(dict.fromkeys(file_ids or []))
        with self._lock:
            conn = self._connect()
            try:
                where = "chunks_fts MATCH ?"
                params: list[Any] = [match_query]
                if allowed:
                    placeholders = ",".join("?" for _ in allowed)
                    where += f" AND chunks.file_id IN ({placeholders})"
                    params.extend(allowed)
                params.append(max(top_k * 4, top_k))
                rows = conn.execute(
                    f"""
                    SELECT
                        chunks.id,
                        chunks.file_id,
                        chunks.filename,
                        chunks.chunk_index,
                        chunks.text,
                        chunks.vector,
                        chunks.start_char,
                        chunks.end_char,
                        bm25(chunks_fts) AS bm25_score
                    FROM chunks_fts
                    JOIN chunks ON chunks.id = chunks_fts.chunk_id
                    WHERE {where}
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            finally:
                conn.close()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "file_id": row["file_id"],
                    "filename": row["filename"],
                    "index": row["chunk_index"],
                    "text": row["text"],
                    "vector": json.loads(row["vector"]),
                    "start_char": row["start_char"],
                    "end_char": row["end_char"],
                    "bm25_score": float(row["bm25_score"]),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def upsert_chat(self, chat: ChatSession) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO chats(id, payload, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                    """,
                    (chat.id, chat.model_dump_json(), chat.updated_at.isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

    def list_chats(self) -> list[ChatSession]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT payload FROM chats ORDER BY updated_at DESC").fetchall()
            finally:
                conn.close()
        return [ChatSession.model_validate_json(row["payload"]) for row in rows]

    def get_chat(self, chat_id: str) -> ChatSession | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT payload FROM chats WHERE id = ?", (chat_id,)).fetchone()
            finally:
                conn.close()
        return ChatSession.model_validate_json(row["payload"]) if row else None

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
                conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                conn.commit()
                return row is not None
            finally:
                conn.close()

    def create_knowledge(self, name: str, description: str = "", file_ids: list[str] | None = None) -> KnowledgeBase:
        now = datetime.now(UTC)
        knowledge = KnowledgeBase(
            id=str(uuid4()),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            file_ids=list(dict.fromkeys(file_ids or [])),
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO knowledge(id, name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (knowledge.id, knowledge.name, knowledge.description, now.isoformat(), now.isoformat()),
                )
                self._replace_knowledge_files(conn, knowledge.id, knowledge.file_ids)
                conn.commit()
            finally:
                conn.close()
        return knowledge

    def list_knowledge(self) -> list[KnowledgeBase]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, name, description, created_at, updated_at FROM knowledge ORDER BY updated_at DESC"
                ).fetchall()
                file_rows = conn.execute(
                    "SELECT knowledge_id, file_id FROM knowledge_files ORDER BY created_at ASC"
                ).fetchall()
            finally:
                conn.close()
        files_by_knowledge: dict[str, list[str]] = {}
        for row in file_rows:
            files_by_knowledge.setdefault(row["knowledge_id"], []).append(row["file_id"])
        return [
            KnowledgeBase(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                file_ids=files_by_knowledge.get(row["id"], []),
            )
            for row in rows
        ]

    def get_knowledge(self, knowledge_id: str) -> KnowledgeBase | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, name, description, created_at, updated_at FROM knowledge WHERE id = ?",
                    (knowledge_id,),
                ).fetchone()
                file_rows = conn.execute(
                    "SELECT file_id FROM knowledge_files WHERE knowledge_id = ? ORDER BY created_at ASC",
                    (knowledge_id,),
                ).fetchall()
            finally:
                conn.close()
        if not row:
            return None
        return KnowledgeBase(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            file_ids=[item["file_id"] for item in file_rows],
        )

    def delete_knowledge(self, knowledge_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT id FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
                conn.execute("DELETE FROM knowledge_files WHERE knowledge_id = ?", (knowledge_id,))
                conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
                conn.commit()
                return row is not None
            finally:
                conn.close()

    def set_knowledge_files(self, knowledge_id: str, file_ids: list[str]) -> KnowledgeBase | None:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT id FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
                if not row:
                    return None
                self._replace_knowledge_files(conn, knowledge_id, list(dict.fromkeys(file_ids)))
                conn.execute("UPDATE knowledge SET updated_at = ? WHERE id = ?", (now, knowledge_id))
                conn.commit()
            finally:
                conn.close()
        return self.get_knowledge(knowledge_id)

    def get_knowledge_file_ids(self, knowledge_ids: list[str]) -> list[str]:
        if not knowledge_ids:
            return []
        placeholders = ",".join("?" for _ in knowledge_ids)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT file_id FROM knowledge_files WHERE knowledge_id IN ({placeholders}) ORDER BY created_at ASC",
                    knowledge_ids,
                ).fetchall()
            finally:
                conn.close()
        return list(dict.fromkeys(row["file_id"] for row in rows))

    def create_skill(
        self,
        name: str,
        instruction: str,
        description: str = "",
        triggers: list[str] | None = None,
        tool_ids: list[str] | None = None,
        enabled: bool = True,
    ) -> Skill:
        now = datetime.now(UTC)
        skill = Skill(
            id=str(uuid4()),
            name=name,
            description=description,
            instruction=instruction,
            triggers=list(dict.fromkeys(triggers or [])),
            tool_ids=list(dict.fromkeys(tool_ids or [])),
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )
        self._upsert_skill(skill)
        return skill

    def list_skills(self) -> list[Skill]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT payload FROM skills ORDER BY updated_at DESC").fetchall()
            finally:
                conn.close()
        return [Skill.model_validate_json(row["payload"]) for row in rows]

    def get_skill(self, skill_id: str) -> Skill | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT payload FROM skills WHERE id = ?", (skill_id,)).fetchone()
            finally:
                conn.close()
        return Skill.model_validate_json(row["payload"]) if row else None

    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> Skill | None:
        skill = self.get_skill(skill_id)
        if not skill:
            return None
        data = skill.model_dump()
        for key, value in updates.items():
            if value is not None:
                data[key] = value
        data["updated_at"] = datetime.now(UTC)
        updated = Skill.model_validate(data)
        self._upsert_skill(updated)
        return updated

    def delete_skill(self, skill_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT id FROM skills WHERE id = ?", (skill_id,)).fetchone()
                conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
                conn.commit()
                return row is not None
            finally:
                conn.close()

    def _upsert_skill(self, skill: Skill) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO skills(id, payload, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                    """,
                    (skill.id, skill.model_dump_json(), skill.updated_at.isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

    def _replace_knowledge_files(self, conn: sqlite3.Connection, knowledge_id: str, file_ids: list[str]) -> None:
        now = datetime.now(UTC).isoformat()
        conn.execute("DELETE FROM knowledge_files WHERE knowledge_id = ?", (knowledge_id,))
        conn.executemany(
            "INSERT OR IGNORE INTO knowledge_files(knowledge_id, file_id, created_at) VALUES (?, ?, ?)",
            [(knowledge_id, file_id, now) for file_id in file_ids],
        )


def make_fts_query(query: str) -> str:
    tokens = []
    for token in query.replace('"', " ").replace("'", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum() or ch == "_")
        if len(cleaned) >= 2:
            tokens.append(cleaned)
    return " OR ".join(tokens[:16])


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
