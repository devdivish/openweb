from __future__ import annotations

import shutil
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re

from app.config import Settings
from app.embeddings import Embedder
from app.llm import AnswerGenerator, analyze_grounding, build_prompt_messages
from app.retrieval import (
    audit_answer,
    build_diagnostics,
    build_source_pack,
    fuse_candidate_sources,
    meaningful_terms,
    plan_queries,
    rerank_and_compress_sources,
)
from app.schemas import (
    AnswerGrounding,
    AnswerAuditResponse,
    AnswerQuality,
    BackendStatusResponse,
    ChatAnswerDefaults,
    ChatExportResponse,
    ChatAnswerPreviewResponse,
    ChatContextApplySuggestionsResponse,
    ChatContextPreviewResponse,
    ChatContextSuggestResponse,
    ChatMemoryItem,
    ChatMessage,
    ChatMessagePromptResponse,
    ChatMessageTraceResponse,
    ChatSession,
    FeedbackListItem,
    FeedbackListResponse,
    FileRecord,
    FileOverview,
    FileSearchItem,
    FileSearchResponse,
    MessageFeedback,
    ProviderStatus,
    PromptMessage,
    RetrievalSettingsResponse,
    StorageStatsResponse,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalExplainResponse,
    Skill,
    Source,
)
from app.store import JsonStore
from app.text import chunk_text_with_spans, extract_text, summarize_text
from app.tools import ToolRegistry
from app.vector_store import Chunk, VectorStore


@dataclass
class _PreparedAnswer:
    chat: ChatSession
    history: list[ChatMessage]
    sources: list[Source]
    tool_results: list
    skills: list[Skill]
    retrieval_query: str
    diagnostics: RetrievalDiagnostics
    system_prompt: str
    rag_template: str
    minimum_answerability: str


class RagService:
    def __init__(
        self,
        settings: Settings,
        store: JsonStore,
        embedder: Embedder,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        tool_registry: ToolRegistry | None = None,
    ):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store
        self.answer_generator = answer_generator
        self.tool_registry = tool_registry

    async def ingest_file(self, source_path: Path, filename: str, content_type: str | None, size: int) -> FileRecord:
        file_id = str(uuid.uuid4())
        safe_name = Path(filename).name
        stored_path = self.settings.uploads_dir / f"{file_id}_{safe_name}"
        shutil.copyfile(source_path, stored_path)

        text, chunks = await self._build_chunks(
            file_id=file_id,
            filename=safe_name,
            path=stored_path,
            content_type=content_type,
        )

        summary, keywords = summarize_text(text)
        record = FileRecord(
            id=file_id,
            filename=safe_name,
            content_type=content_type,
            path=str(stored_path),
            bytes=size,
            text_chars=len(text),
            chunk_count=len(chunks),
            summary=summary,
            keywords=keywords,
            created_at=datetime.now(UTC),
        )
        self.store.upsert_file(record)
        return record

    async def reindex_file(self, file_id: str) -> FileRecord:
        record = self.store.get_file(file_id)
        if not record:
            raise KeyError(f"File {file_id} was not found")
        path = Path(record.path)
        if not path.exists():
            raise FileNotFoundError(f"Stored file is missing: {record.path}")

        text, chunks = await self._build_chunks(
            file_id=record.id,
            filename=record.filename,
            path=path,
            content_type=record.content_type,
        )
        summary, keywords = summarize_text(text)
        updated = record.model_copy(
            update={
                "text_chars": len(text),
                "chunk_count": len(chunks),
                "summary": summary,
                "keywords": keywords,
            }
        )
        self.store.upsert_file(updated)
        return updated

    async def reindex_files(
        self,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
    ) -> tuple[list[FileRecord], list[dict[str, str]]]:
        resolved_file_ids = self._resolve_file_ids(file_ids, knowledge_ids)
        target_ids = resolved_file_ids or [record.id for record in self.store.list_files()]
        records: list[FileRecord] = []
        failures: list[dict[str, str]] = []
        for file_id in list(dict.fromkeys(target_ids)):
            try:
                records.append(await self.reindex_file(file_id))
            except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
                failures.append({"file_id": file_id, "error": str(exc)})
        return records, failures

    async def _build_chunks(
        self,
        file_id: str,
        filename: str,
        path: Path,
        content_type: str | None,
    ) -> tuple[str, list[Chunk]]:
        text = extract_text(path, content_type)
        pieces = chunk_text_with_spans(text, self.settings.chunk_size, self.settings.chunk_overlap)
        piece_texts = [piece.text for piece in pieces]
        vectors = await self.embedder.embed(piece_texts) if piece_texts else []
        chunks = [
            Chunk(
                id=f"{file_id}:{idx}",
                file_id=file_id,
                filename=filename,
                text=piece.text,
                vector=vectors[idx],
                index=idx,
                start_char=piece.start_char,
                end_char=piece.end_char,
            )
            for idx, piece in enumerate(pieces)
        ]
        self.vector_store.replace_file_chunks(file_id, chunks)
        return text, chunks

    def create_chat(
        self,
        title: str | None = None,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
    ) -> ChatSession:
        chat = ChatSession(
            id=str(uuid.uuid4()),
            title=title or "New chat",
            file_ids=list(dict.fromkeys(file_ids or [])),
            knowledge_ids=list(dict.fromkeys(knowledge_ids or [])),
        )
        self.store.upsert_chat(chat)
        return chat

    def backend_status(self) -> BackendStatusResponse:
        files = self.store.list_files()
        chats = self.store.list_chats()
        chunks = self.store.chunks()
        knowledge = self.store.list_knowledge()
        skills = self.store.list_skills()
        message_count = sum(len(chat.messages) for chat in chats)
        return BackendStatusResponse(
            embedding_provider=self._embedding_provider_status(),
            llm_provider=self._llm_provider_status(),
            retrieval=RetrievalSettingsResponse(
                top_k=self.settings.top_k,
                relevance_threshold=self.settings.relevance_threshold,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                full_context_max_chars=self.settings.full_context_max_chars,
                chat_context_messages=self.settings.chat_context_messages,
                chat_memory_items=self.settings.chat_memory_items,
                chat_memory_context_chars=self.settings.chat_memory_context_chars,
            ),
            storage=StorageStatsResponse(
                file_count=len(files),
                chunk_count=len(chunks),
                chat_count=len(chats),
                message_count=message_count,
                knowledge_count=len(knowledge),
                skill_count=len(skills),
                total_file_bytes=sum(file.bytes for file in files),
                total_file_text_chars=sum(file.text_chars for file in files),
                total_chunk_text_chars=sum(len(chunk.get("text", "")) for chunk in chunks),
            ),
            openai_api_key_configured=bool(self.settings.openai_api_key),
        )

    def search_files(self, query: str, limit: int = 20, knowledge_id: str | None = None) -> FileSearchResponse:
        query = " ".join(query.strip().split())
        allowed_ids: set[str] | None = None
        if knowledge_id:
            knowledge = self.store.get_knowledge(knowledge_id)
            if not knowledge:
                raise KeyError(f"Knowledge {knowledge_id} was not found")
            allowed_ids = set(knowledge.file_ids)
        all_items = self._search_files_in_scope(query, list(allowed_ids) if allowed_ids is not None else None)
        items = all_items[:limit]
        return FileSearchResponse(query=query, total_count=len(all_items), items=items)

    def suggest_chat_context(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        limit: int = 5,
    ) -> ChatContextSuggestResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        requested_file_ids = list(dict.fromkeys(file_ids or []))
        requested_knowledge_ids = list(dict.fromkeys(knowledge_ids or []))
        candidate_file_ids = self._resolve_file_ids(
            list(dict.fromkeys([*chat.file_ids, *requested_file_ids])) or None,
            list(dict.fromkeys([*chat.knowledge_ids, *requested_knowledge_ids])) or None,
        )
        suggestions = self._search_files_in_scope(message, candidate_file_ids)[:limit]
        suggested_file_ids = [item.file.id for item in suggestions]
        return ChatContextSuggestResponse(
            chat_id=chat.id,
            message=message,
            default_file_ids=chat.file_ids,
            default_knowledge_ids=chat.knowledge_ids,
            requested_file_ids=requested_file_ids,
            requested_knowledge_ids=requested_knowledge_ids,
            candidate_file_ids=candidate_file_ids or [],
            suggested_file_ids=suggested_file_ids,
            suggestions=suggestions,
            files=self._file_overviews(suggested_file_ids),
        )

    def apply_context_suggestions(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        limit: int = 5,
        replace: bool = False,
    ) -> ChatContextApplySuggestionsResponse:
        suggestion = self.suggest_chat_context(
            chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            limit=limit,
        )
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")

        suggested_file_ids = list(dict.fromkeys(suggestion.suggested_file_ids))
        if replace:
            applied_file_ids = suggested_file_ids
            next_file_ids = suggested_file_ids
        else:
            applied_file_ids = [file_id for file_id in suggested_file_ids if file_id not in chat.file_ids]
            next_file_ids = list(dict.fromkeys([*chat.file_ids, *suggested_file_ids]))

        chat.file_ids = next_file_ids
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return ChatContextApplySuggestionsResponse(
            chat=chat,
            suggestion=suggestion,
            applied_file_ids=applied_file_ids,
            replaced=replace,
        )

    def _embedding_provider_status(self) -> ProviderStatus:
        provider = self.settings.embedding_provider.lower()
        if provider == "openai":
            return ProviderStatus(
                provider="openai",
                configured=bool(self.settings.openai_api_key),
                model=self.settings.openai_embedding_model,
                base_url=self.settings.openai_base_url,
            )
        if provider == "ollama":
            return ProviderStatus(
                provider="ollama",
                configured=True,
                model=self.settings.ollama_embedding_model,
                base_url=self.settings.ollama_base_url,
            )
        return ProviderStatus(
            provider=provider,
            configured=True,
            model=f"local_hash:{self.settings.embedding_dimensions}",
        )

    def _llm_provider_status(self) -> ProviderStatus:
        provider = self.settings.llm_provider.lower()
        if provider == "openai":
            return ProviderStatus(
                provider="openai",
                configured=bool(self.settings.openai_api_key),
                model=self.settings.openai_chat_model,
                base_url=self.settings.openai_base_url,
            )
        if provider == "ollama":
            return ProviderStatus(
                provider="ollama",
                configured=True,
                model=self.settings.ollama_chat_model,
                base_url=self.settings.ollama_base_url,
            )
        return ProviderStatus(provider=provider, configured=True, model="extractive")

    def export_chat(self, chat_id: str) -> ChatExportResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        return ChatExportResponse(chat=chat)

    def import_chat(self, chat: ChatSession, title: str | None = None, preserve_ids: bool = False) -> ChatSession:
        imported = chat.model_copy(deep=True)
        now = datetime.now(UTC)
        if preserve_ids:
            if self.store.get_chat(imported.id):
                raise ValueError(f"Chat {imported.id} already exists")
        else:
            imported.id = str(uuid.uuid4())
            imported.messages = [
                message.model_copy(update={"id": str(uuid.uuid4())})
                for message in imported.messages
            ]
            imported.memories = [
                memory.model_copy(update={"id": str(uuid.uuid4()), "updated_at": now})
                for memory in imported.memories
            ]
        imported.title = title or imported.title
        imported.file_ids = list(dict.fromkeys(imported.file_ids))
        imported.knowledge_ids = list(dict.fromkeys(imported.knowledge_ids))
        imported.created_at = now if not preserve_ids else imported.created_at
        imported.updated_at = now
        self.store.upsert_chat(imported)
        return imported

    def update_chat_context(
        self,
        chat_id: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
    ) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        chat.file_ids = list(dict.fromkeys(file_ids or []))
        chat.knowledge_ids = list(dict.fromkeys(knowledge_ids or []))
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def update_chat_answer_defaults(self, chat_id: str, updates: dict) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        data = chat.answer_defaults.model_dump()
        for key, value in updates.items():
            data[key] = value
        chat.answer_defaults = ChatAnswerDefaults.model_validate(data)
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def compact_chat(self, chat_id: str, keep_last: int | None = None) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        keep = self.settings.chat_context_messages if keep_last is None else keep_last
        if self._compact_messages(chat, keep):
            chat.updated_at = datetime.now(UTC)
            self.store.upsert_chat(chat)
        return chat

    def prune_chat_messages(self, chat_id: str, message_id: str, include_selected: bool = True) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        target_index = next((index for index, item in enumerate(chat.messages) if item.id == message_id), None)
        if target_index is None:
            raise KeyError(f"Message {message_id} was not found")
        keep_count = target_index if include_selected else target_index + 1
        chat.messages = chat.messages[:keep_count]
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def add_chat_memory(self, chat_id: str, content: str, source_message: str = "") -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        self._upsert_memory(chat, content, source_message=source_message)
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def delete_chat_memory(self, chat_id: str, memory_id: str) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        original_count = len(chat.memories)
        chat.memories = [memory for memory in chat.memories if memory.id != memory_id]
        if len(chat.memories) == original_count:
            raise KeyError(f"Memory {memory_id} was not found")
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def audit_chat_message(self, chat_id: str, message_id: str) -> AnswerAuditResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        message = next((item for item in chat.messages if item.id == message_id), None)
        if not message:
            raise KeyError(f"Message {message_id} was not found")
        if message.role != "assistant":
            raise ValueError("Only assistant messages can be audited")
        grounding = message.grounding or analyze_grounding(message.content, message.sources)
        return audit_answer(message.content, message.sources, grounding, message_id=message.id)

    def get_message_prompt(self, chat_id: str, message_id: str) -> ChatMessagePromptResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        message = next((item for item in chat.messages if item.id == message_id), None)
        if not message:
            raise KeyError(f"Message {message_id} was not found")
        if message.role != "assistant":
            raise ValueError("Only assistant messages have answer prompt snapshots")
        if not message.prompt_messages:
            raise ValueError("Assistant message does not have a persisted prompt snapshot")
        return ChatMessagePromptResponse(
            chat_id=chat.id,
            message_id=message.id,
            retrieval_query=message.retrieval_query,
            prompt_messages=message.prompt_messages,
            prompt_chars=message.prompt_chars,
            system_prompt=message.system_prompt,
            rag_template=message.rag_template,
        )

    def get_message_trace(self, chat_id: str, message_id: str) -> ChatMessageTraceResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        message_index = next((index for index, item in enumerate(chat.messages) if item.id == message_id), None)
        if message_index is None:
            raise KeyError(f"Message {message_id} was not found")
        message = chat.messages[message_index]
        if message.role != "assistant":
            raise ValueError("Only assistant messages have answer traces")

        diagnostics = message.diagnostics
        source_pack = None
        if diagnostics:
            query = diagnostics.original_query or message.retrieval_query
            source_pack = build_source_pack(query, message.sources, diagnostics)

        grounding = message.grounding or analyze_grounding(message.content, message.sources)
        prompt = None
        if message.prompt_messages:
            prompt = ChatMessagePromptResponse(
                chat_id=chat.id,
                message_id=message.id,
                retrieval_query=message.retrieval_query,
                prompt_messages=message.prompt_messages,
                prompt_chars=message.prompt_chars,
                system_prompt=message.system_prompt,
                rag_template=message.rag_template,
            )
        skills = [self.store.get_skill(skill_id) for skill_id in message.skill_ids]
        return ChatMessageTraceResponse(
            chat_id=chat.id,
            chat_title=chat.title,
            message_id=message.id,
            question=_previous_user_question(chat.messages, message_index),
            answer=message,
            retrieval_query=message.retrieval_query,
            diagnostics=diagnostics,
            source_pack=source_pack,
            audit=audit_answer(message.content, message.sources, grounding, message_id=message.id),
            prompt=prompt,
            tool_results=message.tool_results,
            skills=[skill for skill in skills if skill],
            feedback=message.feedback,
        )

    def update_message_feedback(
        self,
        chat_id: str,
        message_id: str,
        rating: str | None = None,
        tags: list[str] | None = None,
        comment: str = "",
    ) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        message = next((item for item in chat.messages if item.id == message_id), None)
        if not message:
            raise KeyError(f"Message {message_id} was not found")
        if message.role != "assistant":
            raise ValueError("Only assistant messages can receive feedback")
        message.feedback = MessageFeedback(
            rating=rating,
            tags=_clean_feedback_tags(tags or []),
            comment=comment.strip(),
            updated_at=datetime.now(UTC),
        )
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def delete_message_feedback(self, chat_id: str, message_id: str) -> ChatSession:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        message = next((item for item in chat.messages if item.id == message_id), None)
        if not message:
            raise KeyError(f"Message {message_id} was not found")
        if message.role != "assistant":
            raise ValueError("Only assistant messages can have feedback")
        message.feedback = None
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)
        return chat

    def list_feedback(
        self,
        rating: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> FeedbackListResponse:
        normalized_tag = " ".join((tag or "").strip().lower().split())
        items: list[FeedbackListItem] = []
        for chat in self.store.list_chats():
            for index, message in enumerate(chat.messages):
                if message.role != "assistant" or not message.feedback:
                    continue
                if rating is not None and message.feedback.rating != rating:
                    continue
                if normalized_tag and normalized_tag not in message.feedback.tags:
                    continue
                items.append(
                    FeedbackListItem(
                        chat_id=chat.id,
                        chat_title=chat.title,
                        message_id=message.id,
                        message_created_at=message.created_at,
                        question=_previous_user_question(chat.messages, index),
                        answer=message.content,
                        feedback=message.feedback,
                        retrieval_query=message.retrieval_query,
                        diagnostics=message.diagnostics,
                        grounding=message.grounding,
                        source_count=len(message.sources),
                    )
                )
        items.sort(key=lambda item: item.feedback.updated_at, reverse=True)
        return FeedbackListResponse(total_count=len(items), items=items[:limit])

    def preview_context(
        self,
        chat_id: str,
        message: str = "",
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
    ) -> ChatContextPreviewResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        rolling_messages = self._rolling_context(chat)
        plan = plan_queries(rolling_messages, message) if message.strip() else None
        requested_file_ids = list(dict.fromkeys(file_ids or []))
        requested_knowledge_ids = list(dict.fromkeys(knowledge_ids or []))
        direct_file_ids = list(dict.fromkeys([*chat.file_ids, *requested_file_ids]))
        direct_knowledge_ids = list(dict.fromkeys([*chat.knowledge_ids, *requested_knowledge_ids]))
        resolved_file_ids = self._resolve_file_ids(direct_file_ids or None, direct_knowledge_ids or None) or []
        return ChatContextPreviewResponse(
            chat_id=chat.id,
            title=chat.title,
            summary=chat.summary,
            memories=chat.memories,
            memory_context=self._memory_context(chat),
            rolling_messages=rolling_messages,
            planned_queries=plan.queries if plan else [],
            retrieval_query="\n".join(plan.queries) if plan else "",
            default_file_ids=chat.file_ids,
            default_knowledge_ids=chat.knowledge_ids,
            requested_file_ids=requested_file_ids,
            requested_knowledge_ids=requested_knowledge_ids,
            resolved_file_ids=resolved_file_ids,
            files=self._file_overviews(resolved_file_ids),
            context_message_count=len(rolling_messages),
            context_chars=sum(len(message.content) for message in rolling_messages),
        )

    async def ask(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
    ) -> tuple[ChatMessage, list[Source], list, list[Skill], str, RetrievalDiagnostics, AnswerGrounding]:
        prepared = await self._prepare_answer(
            chat_id=chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=top_k,
            use_rag=use_rag,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            minimum_answerability=minimum_answerability,
            system_prompt=system_prompt,
            rag_template=rag_template,
            use_tools=use_tools,
            tool_ids=tool_ids,
            use_skills=use_skills,
            skill_ids=skill_ids,
        )
        answer = await self._generate_answer(message, prepared)
        return self._persist_answer(message, answer, prepared)

    async def preview_answer(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
    ) -> ChatAnswerPreviewResponse:
        prepared = await self._prepare_answer(
            chat_id=chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=top_k,
            use_rag=use_rag,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            minimum_answerability=minimum_answerability,
            system_prompt=system_prompt,
            rag_template=rag_template,
            use_tools=use_tools,
            tool_ids=tool_ids,
            use_skills=use_skills,
            skill_ids=skill_ids,
            learn_memories=False,
        )
        prompt_messages = [
            PromptMessage.model_validate(item)
            for item in build_prompt_messages(
                prepared.history,
                message,
                prepared.sources,
                prepared.tool_results,
                system_prompt=prepared.system_prompt,
                rag_template=prepared.rag_template,
            )
        ]
        return ChatAnswerPreviewResponse(
            chat_id=chat_id,
            message=message,
            retrieval_query=prepared.retrieval_query,
            diagnostics=prepared.diagnostics,
            source_pack=build_source_pack(message, prepared.sources, prepared.diagnostics),
            tool_results=prepared.tool_results,
            skills=prepared.skills,
            prompt_messages=prompt_messages,
            system_prompt=prepared.system_prompt,
            rag_template=prepared.rag_template,
            context_message_count=len(prepared.history),
            prompt_chars=sum(len(item.content) for item in prompt_messages),
            would_learn_memories=self._extract_memories(message),
        )

    async def stream_answer(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, object]]:
        prepared = await self._prepare_answer(
            chat_id=chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=top_k,
            use_rag=use_rag,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            minimum_answerability=minimum_answerability,
            system_prompt=system_prompt,
            rag_template=rag_template,
            use_tools=use_tools,
            tool_ids=tool_ids,
            use_skills=use_skills,
            skill_ids=skill_ids,
        )
        yield "retrieval", prepared
        if prepared.tool_results:
            yield "tools", prepared.tool_results

        chunks = []
        if not _meets_minimum_answerability(prepared.diagnostics.answerability, prepared.minimum_answerability):
            answer = self._guarded_answer(prepared.diagnostics, prepared.minimum_answerability)
            chunks.append(answer)
            yield "token", answer
        else:
            async for token in self.answer_generator.stream_answer(
                prepared.history,
                message,
                prepared.sources,
                prepared.tool_results,
                system_prompt=prepared.system_prompt,
                rag_template=prepared.rag_template,
            ):
                chunks.append(token)
                yield "token", token

        persisted = self._persist_answer(message, "".join(chunks), prepared)
        yield "done", persisted

    async def _prepare_answer(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
        learn_memories: bool = True,
    ) -> _PreparedAnswer:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")

        history = self._rolling_context(chat)
        plan = plan_queries(history, message)
        retrieval_query = "\n".join(plan.queries)
        resolved_file_ids = list(dict.fromkeys([*chat.file_ids, *(file_ids or [])]))
        resolved_knowledge_ids = list(dict.fromkeys([*chat.knowledge_ids, *(knowledge_ids or [])]))
        sources, diagnostics = await self.retrieve(
            query=message,
            file_ids=resolved_file_ids or None,
            knowledge_ids=resolved_knowledge_ids or None,
            top_k=top_k,
            context_mode=("none" if not use_rag else context_mode),
            retrieval_mode=retrieval_mode,
            planned_queries=plan.queries,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
        )

        skills = self._select_skills(message, skill_ids if use_skills else [])
        tool_results = []
        if use_tools and self.tool_registry:
            skill_tool_ids = _skill_tool_ids(skills) if use_skills else []
            tool_results = await self.tool_registry.run_for_message(
                message,
                tool_ids=tool_ids,
                force_tool_ids=skill_tool_ids,
            )
        effective_system_prompt = system_prompt or self.settings.system_prompt
        if use_skills and skills:
            effective_system_prompt = self._apply_skills_to_prompt(effective_system_prompt, skills)

        if learn_memories:
            learned_memories = self._extract_memories(message)
            for memory in learned_memories:
                self._upsert_memory(chat, memory, source_message=message)
            history = self._rolling_context(chat)
        return _PreparedAnswer(
            chat=chat,
            history=history,
            sources=sources,
            tool_results=tool_results,
            skills=skills,
            retrieval_query=retrieval_query,
            diagnostics=diagnostics,
            system_prompt=effective_system_prompt,
            rag_template=rag_template or self.settings.rag_template,
            minimum_answerability=minimum_answerability,
        )

    async def _generate_answer(self, message: str, prepared: _PreparedAnswer) -> str:
        if not _meets_minimum_answerability(prepared.diagnostics.answerability, prepared.minimum_answerability):
            return self._guarded_answer(prepared.diagnostics, prepared.minimum_answerability)
        return await self.answer_generator.answer(
            prepared.history,
            message,
            prepared.sources,
            prepared.tool_results,
            system_prompt=prepared.system_prompt,
            rag_template=prepared.rag_template,
        )

    def _guarded_answer(self, diagnostics: RetrievalDiagnostics, minimum_answerability: str) -> str:
        diagnostics.warnings.append(
            f"Minimum answerability guard blocked generation: required {minimum_answerability}, got {diagnostics.answerability}."
        )
        return (
            "I do not have enough retrieved file context to answer that confidently. "
            f"Required answerability: {minimum_answerability}; actual answerability: {diagnostics.answerability}."
        )

    def _persist_answer(
        self, message: str, answer: str, prepared: _PreparedAnswer
    ) -> tuple[ChatMessage, list[Source], list, list[Skill], str, RetrievalDiagnostics, AnswerGrounding]:
        grounding = analyze_grounding(answer, prepared.sources)
        prompt_messages = [
            PromptMessage.model_validate(item)
            for item in build_prompt_messages(
                prepared.history,
                message,
                prepared.sources,
                prepared.tool_results,
                system_prompt=prepared.system_prompt,
                rag_template=prepared.rag_template,
            )
        ]
        chat = prepared.chat
        chat.messages.append(ChatMessage(role="user", content=message))
        assistant_message = ChatMessage(
            role="assistant",
            content=answer,
            sources=prepared.sources,
            tool_results=prepared.tool_results,
            skill_ids=[skill.id for skill in prepared.skills],
            retrieval_query=prepared.retrieval_query,
            diagnostics=prepared.diagnostics,
            grounding=grounding,
            prompt_messages=prompt_messages,
            prompt_chars=sum(len(item.content) for item in prompt_messages),
            system_prompt=prepared.system_prompt,
            rag_template=prepared.rag_template,
        )
        audit = audit_answer(answer, prepared.sources, grounding, message_id=assistant_message.id)
        assistant_message.answer_quality = _quality_from_audit(audit)
        chat.messages.append(assistant_message)
        self._refresh_summary(chat)
        chat.updated_at = datetime.now(UTC)
        if chat.title == "New chat":
            chat.title = message[:60]
        self.store.upsert_chat(chat)
        return (
            assistant_message,
            prepared.sources,
            prepared.tool_results,
            prepared.skills,
            prepared.retrieval_query,
            prepared.diagnostics,
            grounding,
        )

    async def regenerate_last(
        self,
        chat_id: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
    ) -> tuple[ChatMessage, list[Source], list, list[Skill], str, RetrievalDiagnostics, AnswerGrounding]:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        if len(chat.messages) < 2:
            raise ValueError("Chat does not have an assistant answer to regenerate")

        user_message = chat.messages[-2]
        assistant_message = chat.messages[-1]
        if user_message.role != "user" or assistant_message.role != "assistant":
            raise ValueError("Last chat turn must be a user message followed by an assistant answer")

        message = user_message.content
        chat.messages = chat.messages[:-2]
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)

        return await self.ask(
            chat_id=chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=top_k,
            use_rag=use_rag,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            minimum_answerability=minimum_answerability,
            system_prompt=system_prompt,
            rag_template=rag_template,
            use_tools=use_tools,
            tool_ids=tool_ids,
            use_skills=use_skills,
            skill_ids=skill_ids,
        )

    async def rerun_from_message(
        self,
        chat_id: str,
        message_id: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
    ) -> tuple[ChatMessage, list[Source], list, list[Skill], str, RetrievalDiagnostics, AnswerGrounding]:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        target_index = next((index for index, item in enumerate(chat.messages) if item.id == message_id), None)
        if target_index is None:
            raise KeyError(f"Message {message_id} was not found")
        user_message = chat.messages[target_index]
        if user_message.role != "user":
            raise ValueError("Only user messages can be rerun")

        message = user_message.content
        chat.messages = chat.messages[:target_index]
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)

        return await self.ask(
            chat_id=chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=top_k,
            use_rag=use_rag,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            minimum_answerability=minimum_answerability,
            system_prompt=system_prompt,
            rag_template=rag_template,
            use_tools=use_tools,
            tool_ids=tool_ids,
            use_skills=use_skills,
            skill_ids=skill_ids,
        )

    async def edit_user_message(
        self,
        chat_id: str,
        message_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        minimum_answerability: str = "none",
        system_prompt: str | None = None,
        rag_template: str | None = None,
        use_tools: bool = True,
        tool_ids: list[str] | None = None,
        use_skills: bool = True,
        skill_ids: list[str] | None = None,
    ) -> tuple[ChatMessage, list[Source], list, list[Skill], str, RetrievalDiagnostics, AnswerGrounding]:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        target_index = next((index for index, item in enumerate(chat.messages) if item.id == message_id), None)
        if target_index is None:
            raise KeyError(f"Message {message_id} was not found")
        if chat.messages[target_index].role != "user":
            raise ValueError("Only user messages can be edited")
        if not message.strip():
            raise ValueError("Edited message cannot be empty")

        chat.messages = chat.messages[:target_index]
        chat.updated_at = datetime.now(UTC)
        self.store.upsert_chat(chat)

        return await self.ask(
            chat_id=chat_id,
            message=message,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=top_k,
            use_rag=use_rag,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            minimum_answerability=minimum_answerability,
            system_prompt=system_prompt,
            rag_template=rag_template,
            use_tools=use_tools,
            tool_ids=tool_ids,
            use_skills=use_skills,
            skill_ids=skill_ids,
        )

    async def retrieve(
        self,
        query: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        planned_queries: list[str] | None = None,
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
    ) -> tuple[list[Source], RetrievalDiagnostics]:
        planned = planned_queries or [query]
        resolved_file_ids = self._resolve_file_ids(file_ids, knowledge_ids)
        resolved_file_ids, routing_warnings = self._apply_query_file_filter(query, resolved_file_ids)
        resolved_file_ids, candidate_file_ids, routed_file_ids, file_routing_warnings = self._route_files_for_query(
            query,
            resolved_file_ids,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
        )
        routing_warnings.extend(file_routing_warnings)
        selected_top_k = top_k or self.settings.top_k
        max_chars = max_context_chars or self.settings.full_context_max_chars
        effective_context_mode = self._effective_context_mode(context_mode, resolved_file_ids, max_chars)
        if context_mode == "none":
            sources: list[Source] = []
            return sources, build_diagnostics(
                query,
                planned,
                context_mode,
                retrieval_mode,
                0,
                sources,
                effective_context_mode=effective_context_mode,
                file_selection_mode=file_selection_mode,
                candidate_file_ids=candidate_file_ids,
                routed_file_ids=routed_file_ids,
                source_window=source_window,
                diversity=diversity,
                mmr_lambda=mmr_lambda,
                warnings=routing_warnings,
            )
        if effective_context_mode == "full":
            sources = self.vector_store.full_context(file_ids=resolved_file_ids, max_chars=max_chars)
            return sources, build_diagnostics(
                query,
                planned,
                context_mode,
                retrieval_mode,
                len(sources),
                sources,
                effective_context_mode=effective_context_mode,
                file_selection_mode=file_selection_mode,
                candidate_file_ids=candidate_file_ids,
                routed_file_ids=routed_file_ids,
                source_window=source_window,
                diversity=diversity,
                mmr_lambda=mmr_lambda,
                warnings=routing_warnings,
            )

        candidates: list[Source] = []
        if effective_context_mode == "rag":
            vectors = await self.embedder.embed(planned)
            for planned_query, query_vector in zip(planned, vectors, strict=False):
                candidates.extend(
                    self.vector_store.search(
                        query_vector=query_vector,
                        query_text=planned_query,
                        top_k=max(selected_top_k * 3, selected_top_k),
                        file_ids=resolved_file_ids,
                        relevance_threshold=self.settings.relevance_threshold,
                        mode=retrieval_mode,
                    )
                )
        sources = rerank_and_compress_sources(
            query,
            candidates,
            selected_top_k,
            max_chars,
            max_sources_per_file=max_sources_per_file,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
        )
        sources = self._expand_source_windows(sources, source_window, max_chars)
        return sources, build_diagnostics(
            query,
            planned,
            context_mode,
            retrieval_mode,
            len(candidates),
            sources,
            effective_context_mode=effective_context_mode,
            file_selection_mode=file_selection_mode,
            candidate_file_ids=candidate_file_ids,
            routed_file_ids=routed_file_ids,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            warnings=routing_warnings,
        )

    async def explain_retrieval(
        self,
        query: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        candidate_limit: int = 20,
        planned_queries: list[str] | None = None,
    ) -> RetrievalExplainResponse:
        selected_top_k = top_k or self.settings.top_k
        planned = planned_queries or [query]
        retrieval_query = "\n".join(planned)
        sources, diagnostics = await self.retrieve(
            query=query,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=selected_top_k,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            planned_queries=planned,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
        )
        candidates = await self._retrieval_candidates(
            query=query,
            planned_queries=planned,
            file_ids=file_ids,
            knowledge_ids=knowledge_ids,
            top_k=selected_top_k,
            context_mode=context_mode,
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            candidate_limit=candidate_limit,
            selected_sources=sources,
        )
        return RetrievalExplainResponse(
            query=query,
            retrieval_query=retrieval_query,
            diagnostics=diagnostics,
            source_pack=build_source_pack(retrieval_query, sources, diagnostics),
            candidates=candidates,
        )

    async def explain_chat_retrieval(
        self,
        chat_id: str,
        message: str,
        file_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        top_k: int | None = None,
        use_rag: bool = True,
        context_mode: str = "rag",
        retrieval_mode: str = "hybrid",
        max_context_chars: int | None = None,
        max_sources_per_file: int | None = None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
        source_window: int = 0,
        diversity: str = "relevance",
        mmr_lambda: float = 0.7,
        candidate_limit: int = 20,
    ) -> RetrievalExplainResponse:
        chat = self.store.get_chat(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} was not found")
        history = self._rolling_context(chat)
        plan = plan_queries(history, message)
        resolved_file_ids = list(dict.fromkeys([*chat.file_ids, *(file_ids or [])]))
        resolved_knowledge_ids = list(dict.fromkeys([*chat.knowledge_ids, *(knowledge_ids or [])]))
        return await self.explain_retrieval(
            query=message,
            file_ids=resolved_file_ids or None,
            knowledge_ids=resolved_knowledge_ids or None,
            top_k=top_k,
            context_mode=("none" if not use_rag else context_mode),
            retrieval_mode=retrieval_mode,
            max_context_chars=max_context_chars,
            max_sources_per_file=max_sources_per_file,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
            source_window=source_window,
            diversity=diversity,
            mmr_lambda=mmr_lambda,
            candidate_limit=candidate_limit,
            planned_queries=plan.queries,
        )

    async def _retrieval_candidates(
        self,
        query: str,
        planned_queries: list[str],
        file_ids: list[str] | None,
        knowledge_ids: list[str] | None,
        top_k: int,
        context_mode: str,
        retrieval_mode: str,
        max_context_chars: int | None,
        file_selection_mode: str,
        file_selection_limit: int,
        candidate_limit: int,
        selected_sources: list[Source],
    ) -> list[RetrievalCandidate]:
        resolved_file_ids = self._resolve_file_ids(file_ids, knowledge_ids)
        resolved_file_ids, _routing_warnings = self._apply_query_file_filter(query, resolved_file_ids)
        resolved_file_ids, _candidate_file_ids, _routed_file_ids, _file_routing_warnings = self._route_files_for_query(
            query,
            resolved_file_ids,
            file_selection_mode=file_selection_mode,
            file_selection_limit=file_selection_limit,
        )
        max_chars = max_context_chars or self.settings.full_context_max_chars
        effective_context_mode = self._effective_context_mode(context_mode, resolved_file_ids, max_chars)
        if context_mode == "none":
            return []
        if effective_context_mode == "full":
            full_sources = self.vector_store.full_context(file_ids=resolved_file_ids, max_chars=max_chars)
            return self._explain_candidates(query, full_sources, selected_sources, candidate_limit)

        raw_candidates: list[Source] = []
        vectors = await self.embedder.embed(planned_queries)
        for planned_query, query_vector in zip(planned_queries, vectors, strict=False):
            raw_candidates.extend(
                self.vector_store.search(
                    query_vector=query_vector,
                    query_text=planned_query,
                    top_k=max(candidate_limit, top_k * 3, top_k),
                    file_ids=resolved_file_ids,
                    relevance_threshold=self.settings.relevance_threshold,
                    mode=retrieval_mode,
                )
            )
        return self._explain_candidates(query, raw_candidates, selected_sources, candidate_limit)

    def _explain_candidates(
        self,
        query: str,
        candidates: list[Source],
        selected_sources: list[Source],
        candidate_limit: int,
    ) -> list[RetrievalCandidate]:
        selected_ids = {source.chunk_id for source in selected_sources}
        query_terms = meaningful_terms(query)
        deduped: dict[str, Source] = {}
        for candidate in sorted(fuse_candidate_sources(candidates), key=lambda item: item.score, reverse=True):
            if candidate.chunk_id not in deduped:
                deduped[candidate.chunk_id] = candidate

        explained = []
        for rank, source in enumerate(list(deduped.values())[:candidate_limit], start=1):
            matched_terms = sorted(query_terms & meaningful_terms(source.text))
            reasons = [f"score={source.score}"]
            if source.query_hits > 1:
                reasons.append(f"query_hits={source.query_hits}")
            if matched_terms:
                reasons.append(f"matched_terms={len(matched_terms)}")
            reasons.append("selected_for_context" if source.chunk_id in selected_ids else "not_selected")
            explained.append(
                RetrievalCandidate(
                    rank=rank,
                    selected=source.chunk_id in selected_ids,
                    source=source,
                    matched_terms=matched_terms,
                    reasons=reasons,
                )
            )
        return explained

    def _resolve_file_ids(self, file_ids: list[str] | None, knowledge_ids: list[str] | None) -> list[str] | None:
        resolved = list(file_ids or [])
        if knowledge_ids and hasattr(self.store, "get_knowledge_file_ids"):
            resolved.extend(self.store.get_knowledge_file_ids(knowledge_ids))
        deduped = list(dict.fromkeys(resolved))
        return deduped or None

    def _file_overviews(self, file_ids: list[str]) -> list[FileOverview]:
        overviews = []
        for file_id in file_ids:
            record = self.store.get_file(file_id)
            if not record:
                continue
            overviews.append(
                FileOverview(
                    id=record.id,
                    filename=record.filename,
                    summary=record.summary,
                    keywords=record.keywords,
                    text_chars=record.text_chars,
                    chunk_count=record.chunk_count,
                )
            )
        return overviews

    def _search_files_in_scope(
        self,
        query: str,
        file_ids: list[str] | None = None,
    ) -> list[FileSearchItem]:
        query = " ".join(query.strip().split())
        allowed_ids = set(file_ids or [])
        query_terms = meaningful_terms(query)
        query_text = query.lower()
        items = []
        for record in self.store.list_files():
            if allowed_ids and record.id not in allowed_ids:
                continue
            score, matched_terms, reasons = _score_file_match(record, query_terms, query_text)
            if score <= 0:
                continue
            items.append(
                FileSearchItem(
                    file=record,
                    score=round(score, 5),
                    matched_terms=matched_terms,
                    reasons=reasons,
                )
            )
        items.sort(key=lambda item: (item.score, item.file.created_at), reverse=True)
        return items

    def _apply_query_file_filter(self, query: str, file_ids: list[str] | None) -> tuple[list[str] | None, list[str]]:
        query_text = _normalize_filename_text(query)
        if not query_text:
            return file_ids, []
        allowed = self.store.list_files()
        if file_ids:
            allowed_ids = set(file_ids)
            allowed = [record for record in allowed if record.id in allowed_ids]
        matches = []
        for record in allowed:
            if _filename_matches_query(record.filename, query_text):
                matches.append(record)
        if not matches or len(matches) == len(allowed):
            return file_ids, []
        matched_ids = [record.id for record in matches]
        matched_names = ", ".join(record.filename for record in matches[:3])
        if len(matches) > 3:
            matched_names += f", and {len(matches) - 3} more"
        return matched_ids, [f"Filename-aware retrieval narrowed context to: {matched_names}."]

    def _route_files_for_query(
        self,
        query: str,
        file_ids: list[str] | None,
        file_selection_mode: str = "all",
        file_selection_limit: int = 5,
    ) -> tuple[list[str] | None, list[str], list[str], list[str]]:
        candidate_file_ids = list(dict.fromkeys(file_ids or []))
        if file_selection_mode != "auto" or not candidate_file_ids:
            return file_ids, candidate_file_ids, candidate_file_ids, []

        limit = max(1, file_selection_limit)
        ranked = self._search_files_in_scope(query, candidate_file_ids)
        if not ranked:
            return (
                file_ids,
                candidate_file_ids,
                candidate_file_ids,
                ["Auto file routing found no file-level match, so retrieval kept all candidate files."],
            )

        routed = [item.file.id for item in ranked[:limit]]
        routed_names = ", ".join(item.file.filename for item in ranked[:3])
        if len(ranked[:limit]) > 3:
            routed_names += f", and {len(ranked[:limit]) - 3} more"
        warning = (
            f"Auto file routing selected {len(routed)} of {len(candidate_file_ids)} "
            f"candidate files: {routed_names}."
        )
        return routed, candidate_file_ids, routed, [warning]

    def _effective_context_mode(self, context_mode: str, file_ids: list[str] | None, max_chars: int) -> str:
        if context_mode != "auto":
            return context_mode
        total_chars = self._indexed_context_chars(file_ids)
        if total_chars and total_chars <= max_chars:
            return "full"
        return "rag"

    def _indexed_context_chars(self, file_ids: list[str] | None) -> int:
        allowed = set(file_ids or [])
        total = 0
        for chunk in self.store.chunks():
            if allowed and chunk["file_id"] not in allowed:
                continue
            total += len(chunk["text"])
        return total

    def _expand_source_windows(self, sources: list[Source], source_window: int, max_chars: int) -> list[Source]:
        if source_window <= 0 or not sources:
            return sources

        expanded = []
        used = 0
        chunks_by_file: dict[str, list] = {}
        for source in sources:
            file_chunks = chunks_by_file.setdefault(source.file_id, self.store.file_chunks(source.file_id))
            chunks_by_index = {chunk.index: chunk for chunk in file_chunks}
            start = max(0, source.chunk_index - source_window)
            end = min(max(chunks_by_index, default=source.chunk_index), source.chunk_index + source_window)
            parts = []
            for index in range(start, end + 1):
                chunk = chunks_by_index.get(index)
                if chunk:
                    parts.append(f"[chunk {index}]\n{chunk.text.strip()}")
            text = "\n\n".join(parts).strip() or source.text
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rsplit(" ", 1)[0].strip()
            if not text:
                continue
            selected_chunks = [chunks_by_index[index] for index in range(start, end + 1) if index in chunks_by_index]
            expanded.append(
                source.model_copy(
                    update={
                        "text": text,
                        "context_start_index": start,
                        "context_end_index": end,
                        "start_char": min((chunk.start_char for chunk in selected_chunks), default=source.start_char),
                        "end_char": max((chunk.end_char for chunk in selected_chunks), default=source.end_char),
                    }
                )
            )
            used += len(text)
        return expanded

    def _rolling_context(self, chat: ChatSession) -> list[ChatMessage]:
        messages = chat.messages[-self.settings.chat_context_messages :]
        system_parts = []
        memory_context = self._memory_context(chat)
        if memory_context:
            system_parts.append(f"Saved chat memory:\n{memory_context}")
        if chat.summary:
            system_parts.append(f"Earlier conversation summary:\n{chat.summary}")
        if system_parts:
            return [ChatMessage(role="system", content="\n\n".join(system_parts))] + messages
        return messages

    def _select_skills(self, message: str, skill_ids: list[str] | None = None) -> list[Skill]:
        if skill_ids is not None:
            skills = [self.store.get_skill(skill_id) for skill_id in skill_ids]
            return [skill for skill in skills if skill and skill.enabled]
        matches = []
        message_text = message.lower()
        for skill in self.store.list_skills():
            if not skill.enabled:
                continue
            triggers = [trigger.lower().strip() for trigger in skill.triggers if trigger.strip()]
            if not triggers:
                continue
            if any(trigger in message_text for trigger in triggers):
                matches.append(skill)
        return matches

    def _apply_skills_to_prompt(self, system_prompt: str, skills: list[Skill]) -> str:
        blocks = [
            f"<skill name='{skill.name}' id='{skill.id}'>\n{skill.instruction.strip()}\n</skill>"
            for skill in skills
            if skill.instruction.strip()
        ]
        if not blocks:
            return system_prompt
        return system_prompt + "\n\nActive skills:\n" + "\n\n".join(blocks)

    def _refresh_summary(self, chat: ChatSession) -> None:
        self._compact_messages(chat, self.settings.chat_context_messages)

    def _compact_messages(self, chat: ChatSession, keep: int) -> bool:
        keep = max(0, keep)
        if keep and len(chat.messages) <= keep:
            return False
        if not keep and not chat.messages:
            return False
        older = chat.messages[:-keep] if keep else chat.messages
        retained = chat.messages[-keep:] if keep else []
        if not older:
            return False
        lines = [chat.summary] if chat.summary else []
        for message in older:
            content = message.content.replace("\n", " ").strip()
            if len(content) > 240:
                content = content[:240].rsplit(" ", 1)[0] + "..."
            lines.append(f"{message.role}: {content}")
        summary = "\n".join(line for line in lines if line).strip()
        chat.summary = summary[-3000:]
        chat.messages = retained
        return True

    def _memory_context(self, chat: ChatSession) -> str:
        lines = []
        used = 0
        for memory in chat.memories[-self.settings.chat_memory_items :]:
            line = f"- {memory.content.strip()}"
            if used + len(line) > self.settings.chat_memory_context_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _extract_memories(self, message: str) -> list[str]:
        text = " ".join(message.strip().split())
        if not text:
            return []
        patterns = [
            r"\bremember(?: that)? (?P<fact>[^.?!]+)",
            r"\bmy (?P<key>[a-zA-Z][a-zA-Z0-9_ -]{1,40}) is (?P<value>[^.?!]+)",
            r"\bi prefer (?P<fact>[^.?!]+)",
            r"\bcall me (?P<fact>[^.?!]+)",
        ]
        memories = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                groups = match.groupdict()
                if "key" in groups and groups.get("key") and groups.get("value"):
                    memory = f"User's {groups['key'].strip()} is {groups['value'].strip()}"
                elif pattern.startswith("\\bi prefer"):
                    memory = f"User prefers {groups['fact'].strip()}"
                elif pattern.startswith("\\bcall me"):
                    memory = f"User wants to be called {groups['fact'].strip()}"
                else:
                    memory = _canonical_user_fact(groups["fact"].strip())
                memory = memory.strip(" ,;:")
                if len(memory) >= 4:
                    memories.append(memory[0].upper() + memory[1:])
        return list(dict.fromkeys(memories))

    def _upsert_memory(self, chat: ChatSession, content: str, source_message: str = "") -> None:
        normalized = _normalize_memory(content)
        if not normalized:
            return
        now = datetime.now(UTC)
        for memory in chat.memories:
            if _normalize_memory(memory.content) == normalized:
                memory.content = content.strip()
                memory.source_message = source_message or memory.source_message
                memory.updated_at = now
                break
        else:
            chat.memories.append(
                ChatMemoryItem(
                    id=str(uuid.uuid4()),
                    content=content.strip(),
                    source_message=source_message,
                    created_at=now,
                    updated_at=now,
                )
            )
        chat.memories = chat.memories[-self.settings.chat_memory_items :]


def _normalize_memory(content: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9_]+", content.lower()))


def _normalize_filename_text(value: str) -> str:
    value = value.lower().replace("\\", " ").replace("/", " ")
    return " ".join(re.findall(r"[a-zA-Z0-9_.-]+", value))


def _score_file_match(record: FileRecord, query_terms: set[str], query_text: str) -> tuple[float, list[str], list[str]]:
    filename_terms = meaningful_terms(Path(record.filename).stem.replace("-", " ").replace("_", " "))
    keyword_terms = set(record.keywords)
    summary_terms = meaningful_terms(record.summary)
    matched_terms = sorted(query_terms & (filename_terms | keyword_terms | summary_terms))
    reasons = []
    score = 0.0

    normalized_filename = _normalize_filename_text(record.filename)
    normalized_stem = _normalize_filename_text(Path(record.filename).stem)
    if normalized_filename and normalized_filename in query_text:
        score += 3.0
        reasons.append("filename_exact")
    elif normalized_stem and normalized_stem in query_text:
        score += 2.0
        reasons.append("filename_stem")

    filename_overlap = query_terms & filename_terms
    if filename_overlap:
        score += 1.5 * len(filename_overlap)
        reasons.append(f"filename_terms={len(filename_overlap)}")
    keyword_overlap = query_terms & keyword_terms
    if keyword_overlap:
        score += 1.25 * len(keyword_overlap)
        reasons.append(f"keywords={len(keyword_overlap)}")
    summary_overlap = query_terms & summary_terms
    if summary_overlap:
        score += 0.75 * len(summary_overlap)
        reasons.append(f"summary_terms={len(summary_overlap)}")

    return score, matched_terms, reasons


def _filename_matches_query(filename: str, normalized_query: str) -> bool:
    normalized_name = _normalize_filename_text(filename)
    if not normalized_name:
        return False
    stem = Path(filename).stem
    normalized_stem = _normalize_filename_text(stem)
    query_words = set(normalized_query.replace(".", " ").replace("-", " ").replace("_", " ").split())
    stem_words = [
        word
        for word in normalized_stem.replace(".", " ").replace("-", " ").replace("_", " ").split()
        if len(word) >= 3
    ]
    return (
        normalized_name in normalized_query
        or bool(normalized_stem and normalized_stem in normalized_query)
        or bool(len(stem_words) >= 2 and all(word in query_words for word in stem_words))
    )


def _canonical_user_fact(content: str) -> str:
    match = re.match(r"my ([a-zA-Z][a-zA-Z0-9_ -]{1,40}) is (.+)", content, flags=re.IGNORECASE)
    if match:
        return f"User's {match.group(1).strip()} is {match.group(2).strip()}"
    return content


def _clean_feedback_tags(tags: list[str]) -> list[str]:
    cleaned = []
    for tag in tags:
        normalized = " ".join(tag.strip().lower().split())
        if normalized and normalized not in cleaned:
            cleaned.append(normalized[:80])
    return cleaned[:20]


def _skill_tool_ids(skills: list[Skill]) -> list[str]:
    tool_ids = []
    for skill in skills:
        for tool_id in skill.tool_ids:
            normalized = tool_id.strip()
            if normalized and normalized not in tool_ids:
                tool_ids.append(normalized)
    return tool_ids


def _previous_user_question(messages: list[ChatMessage], assistant_index: int) -> str:
    for message in reversed(messages[:assistant_index]):
        if message.role == "user":
            return message.content
    return ""


def _meets_minimum_answerability(actual: str, minimum: str) -> bool:
    ranks = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return ranks.get(actual, 0) >= ranks.get(minimum, 0)


def _quality_from_audit(audit: AnswerAuditResponse) -> AnswerQuality:
    if audit.sentence_count == 0:
        status = "unknown"
    elif audit.answer_supported:
        status = "supported"
    elif audit.supported_count or audit.weak_count:
        status = "weak"
    else:
        status = "unsupported"
    return AnswerQuality(
        status=status,
        answer_supported=audit.answer_supported,
        support_score=audit.support_score,
        sentence_count=audit.sentence_count,
        supported_count=audit.supported_count,
        weak_count=audit.weak_count,
        unsupported_count=audit.unsupported_count,
        warnings=audit.warnings,
    )
