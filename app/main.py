from __future__ import annotations

import tempfile
from pathlib import Path

import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.embeddings import build_embedder
from app.llm import build_answer_generator
from app.rag import RagService
from app.retrieval import build_source_pack
from app.schemas import (
    AnswerAuditResponse,
    BatchFileUploadResponse,
    BackendStatusResponse,
    ChatAnswerPreviewResponse,
    ChatContextApplySuggestionsRequest,
    ChatContextApplySuggestionsResponse,
    ChatContextPreviewRequest,
    ChatContextPreviewResponse,
    ChatContextSuggestRequest,
    ChatContextSuggestResponse,
    ChatExportResponse,
    ChatMessagePromptResponse,
    ChatMessageTraceResponse,
    ChatRequest,
    ChatResponse,
    ChatRetrievalExplainRequest,
    ChatSession,
    CompactChatRequest,
    CreateChatMemoryRequest,
    CreateChatRequest,
    CreateChatResponse,
    CreateSkillRequest,
    EditChatMessageRequest,
    FeedbackListResponse,
    FileChunk,
    FileChunkWindowResponse,
    FileSummaryResponse,
    FileTextResponse,
    CreateKnowledgeRequest,
    FileRecord,
    FileSearchResponse,
    HealthResponse,
    ImportChatRequest,
    KnowledgeBase,
    KnowledgeBatchFileUploadResponse,
    KnowledgeFileUploadResponse,
    PruneChatMessagesRequest,
    RegenerateChatRequest,
    ReindexFailure,
    ReindexFilesRequest,
    ReindexFilesResponse,
    RetrievalExplainRequest,
    RetrievalExplainResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    SourcePackResponse,
    Skill,
    ToolSpec,
    UpdateChatAnswerDefaultsRequest,
    UpdateChatContextRequest,
    UpdateKnowledgeFilesRequest,
    UpdateMessageFeedbackRequest,
    UpdateSkillRequest,
)
from app.store import SQLiteStore
from app.text import extract_text
from app.tools import ToolRegistry
from app.vector_store import VectorStore

settings = get_settings()
store = SQLiteStore(settings.database_path)
embedder = build_embedder(settings)
vector_store = VectorStore(store)
answer_generator = build_answer_generator(settings)
tool_registry = ToolRegistry(store)
rag_service = RagService(settings, store, embedder, vector_store, answer_generator, tool_registry)

app = FastAPI(title="Lean OpenWebUI Backend", version="0.1.0")

ANSWER_OPTION_FIELDS = [
    "top_k",
    "use_rag",
    "context_mode",
    "retrieval_mode",
    "max_context_chars",
    "max_sources_per_file",
    "file_selection_mode",
    "file_selection_limit",
    "source_window",
    "diversity",
    "mmr_lambda",
    "minimum_answerability",
    "system_prompt",
    "rag_template",
    "use_tools",
    "tool_ids",
    "use_skills",
    "skill_ids",
]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
    )


@app.get("/api/status", response_model=BackendStatusResponse)
async def backend_status() -> BackendStatusResponse:
    return rag_service.backend_status()


@app.get("/api/feedback", response_model=FeedbackListResponse)
async def list_feedback(
    rating: str | None = None,
    tag: str | None = None,
    limit: int = 100,
) -> FeedbackListResponse:
    if rating not in {None, "up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return rag_service.list_feedback(rating=rating, tag=tag, limit=limit)


@app.post("/api/files", response_model=FileRecord)
async def upload_file(file: UploadFile = File(...)) -> FileRecord:
    suffix = ""
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1] if "." in file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        return await rag_service.ingest_file(
            source_path=Path(tmp_path),
            filename=file.filename or "upload",
            content_type=file.content_type,
            size=len(contents),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/files/batch", response_model=BatchFileUploadResponse)
async def upload_files(files: list[UploadFile] = File(...)) -> BatchFileUploadResponse:
    records = []
    for file in files:
        records.append(await upload_file(file))
    return BatchFileUploadResponse(files=records)


@app.get("/api/files", response_model=list[FileRecord])
async def list_files() -> list[FileRecord]:
    return store.list_files()


@app.get("/api/files/search", response_model=FileSearchResponse)
async def search_files(q: str, limit: int = 20, knowledge_id: str | None = None) -> FileSearchResponse:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    try:
        return rag_service.search_files(q, limit=limit, knowledge_id=knowledge_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/files/{file_id}/summary", response_model=FileSummaryResponse)
async def get_file_summary(file_id: str) -> FileSummaryResponse:
    record = store.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    return FileSummaryResponse(
        file_id=record.id,
        filename=record.filename,
        summary=record.summary,
        keywords=record.keywords,
        text_chars=record.text_chars,
        chunk_count=record.chunk_count,
    )


@app.post("/api/files/reindex", response_model=ReindexFilesResponse)
async def reindex_files(payload: ReindexFilesRequest) -> ReindexFilesResponse:
    records, failures = await rag_service.reindex_files(
        file_ids=payload.file_ids,
        knowledge_ids=payload.knowledge_ids,
    )
    return ReindexFilesResponse(
        requested_count=len(records) + len(failures),
        reindexed_count=len(records),
        files=records,
        failures=[ReindexFailure.model_validate(item) for item in failures],
    )


@app.get("/api/files/{file_id}/text", response_model=FileTextResponse)
async def get_file_text(file_id: str, start: int = 0, end: int | None = None) -> FileTextResponse:
    record = store.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    if start < 0:
        raise HTTPException(status_code=400, detail="start must be greater than or equal to 0")
    path = Path(record.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file not found")
    try:
        text = extract_text(path, record.content_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    total_chars = len(text)
    selected_end = total_chars if end is None else end
    if selected_end < start:
        raise HTTPException(status_code=400, detail="end must be greater than or equal to start")
    selected_start = min(start, total_chars)
    selected_end = min(selected_end, total_chars)
    return FileTextResponse(
        file_id=file_id,
        filename=record.filename,
        start_char=selected_start,
        end_char=selected_end,
        total_chars=total_chars,
        text=text[selected_start:selected_end],
    )


@app.get("/api/files/{file_id}/chunks", response_model=list[FileChunk])
async def list_file_chunks(file_id: str) -> list[FileChunk]:
    if not store.get_file(file_id):
        raise HTTPException(status_code=404, detail="File not found")
    return store.file_chunks(file_id)


@app.get("/api/files/{file_id}/chunks/{chunk_index}/window", response_model=FileChunkWindowResponse)
async def get_file_chunk_window(file_id: str, chunk_index: int, window: int = 1) -> FileChunkWindowResponse:
    if window < 0 or window > 10:
        raise HTTPException(status_code=400, detail="window must be between 0 and 10")
    record = store.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    chunks = store.file_chunks(file_id)
    target = next((chunk for chunk in chunks if chunk.index == chunk_index), None)
    if not target:
        raise HTTPException(status_code=404, detail="Chunk not found")
    start_index = max(0, chunk_index - window)
    end_index = chunk_index + window
    selected = [chunk for chunk in chunks if start_index <= chunk.index <= end_index]
    actual_start = selected[0].index if selected else chunk_index
    actual_end = selected[-1].index if selected else chunk_index
    return FileChunkWindowResponse(
        file_id=file_id,
        filename=record.filename,
        target_index=chunk_index,
        start_index=actual_start,
        end_index=actual_end,
        has_previous=any(chunk.index < actual_start for chunk in chunks),
        has_next=any(chunk.index > actual_end for chunk in chunks),
        chunks=selected,
        context_text="\n\n".join(f"[chunk {chunk.index}]\n{chunk.text}" for chunk in selected),
    )


@app.post("/api/knowledge", response_model=KnowledgeBase)
async def create_knowledge(payload: CreateKnowledgeRequest) -> KnowledgeBase:
    return store.create_knowledge(payload.name, payload.description, payload.file_ids)


@app.get("/api/knowledge", response_model=list[KnowledgeBase])
async def list_knowledge() -> list[KnowledgeBase]:
    return store.list_knowledge()


@app.get("/api/knowledge/{knowledge_id}", response_model=KnowledgeBase)
async def get_knowledge(knowledge_id: str) -> KnowledgeBase:
    knowledge = store.get_knowledge(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return knowledge


@app.put("/api/knowledge/{knowledge_id}/files", response_model=KnowledgeBase)
async def set_knowledge_files(knowledge_id: str, payload: UpdateKnowledgeFilesRequest) -> KnowledgeBase:
    knowledge = store.set_knowledge_files(knowledge_id, payload.file_ids)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return knowledge


@app.post("/api/knowledge/{knowledge_id}/files", response_model=KnowledgeBase)
async def add_knowledge_files(knowledge_id: str, payload: UpdateKnowledgeFilesRequest) -> KnowledgeBase:
    knowledge = store.get_knowledge(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    merged = list(dict.fromkeys([*knowledge.file_ids, *payload.file_ids]))
    updated = store.set_knowledge_files(knowledge_id, merged)
    assert updated is not None
    return updated


@app.post("/api/knowledge/{knowledge_id}/files/upload", response_model=KnowledgeFileUploadResponse)
async def upload_knowledge_file(knowledge_id: str, file: UploadFile = File(...)) -> KnowledgeFileUploadResponse:
    knowledge = store.get_knowledge(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    file_record = await upload_file(file)
    merged = list(dict.fromkeys([*knowledge.file_ids, file_record.id]))
    updated = store.set_knowledge_files(knowledge_id, merged)
    assert updated is not None
    return KnowledgeFileUploadResponse(file=file_record, knowledge=updated)


@app.post("/api/knowledge/{knowledge_id}/files/upload/batch", response_model=KnowledgeBatchFileUploadResponse)
async def upload_knowledge_files(
    knowledge_id: str, files: list[UploadFile] = File(...)
) -> KnowledgeBatchFileUploadResponse:
    knowledge = store.get_knowledge(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    records = []
    for file in files:
        records.append(await upload_file(file))
    merged = list(dict.fromkeys([*knowledge.file_ids, *[record.id for record in records]]))
    updated = store.set_knowledge_files(knowledge_id, merged)
    assert updated is not None
    return KnowledgeBatchFileUploadResponse(files=records, knowledge=updated)


@app.delete("/api/knowledge/{knowledge_id}/files/{file_id}", response_model=KnowledgeBase)
async def remove_knowledge_file(knowledge_id: str, file_id: str) -> KnowledgeBase:
    knowledge = store.get_knowledge(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    updated = store.set_knowledge_files(knowledge_id, [item for item in knowledge.file_ids if item != file_id])
    assert updated is not None
    return updated


@app.delete("/api/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: str) -> dict[str, bool]:
    return {"deleted": store.delete_knowledge(knowledge_id)}


@app.post("/api/knowledge/{knowledge_id}/reindex", response_model=ReindexFilesResponse)
async def reindex_knowledge(knowledge_id: str) -> ReindexFilesResponse:
    if not store.get_knowledge(knowledge_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    records, failures = await rag_service.reindex_files(knowledge_ids=[knowledge_id])
    return ReindexFilesResponse(
        requested_count=len(records) + len(failures),
        reindexed_count=len(records),
        files=records,
        failures=[ReindexFailure.model_validate(item) for item in failures],
    )


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str) -> dict[str, bool]:
    record = store.get_file(file_id)
    deleted = store.delete_file(file_id)
    if deleted and record:
        try:
            Path(record.path).unlink(missing_ok=True)
        except OSError:
            pass
    return {"deleted": deleted}


@app.post("/api/files/{file_id}/reindex", response_model=FileRecord)
async def reindex_file(file_id: str) -> FileRecord:
    try:
        return await rag_service.reindex_file(file_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc


@app.get("/api/tools", response_model=list[ToolSpec])
async def list_tools() -> list[ToolSpec]:
    return tool_registry.specs()


@app.post("/api/skills", response_model=Skill)
async def create_skill(payload: CreateSkillRequest) -> Skill:
    _validate_tool_ids(payload.tool_ids)
    return store.create_skill(
        payload.name,
        payload.instruction,
        description=payload.description,
        triggers=payload.triggers,
        tool_ids=payload.tool_ids,
        enabled=payload.enabled,
    )


@app.get("/api/skills", response_model=list[Skill])
async def list_skills() -> list[Skill]:
    return store.list_skills()


@app.get("/api/skills/{skill_id}", response_model=Skill)
async def get_skill(skill_id: str) -> Skill:
    skill = store.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@app.put("/api/skills/{skill_id}", response_model=Skill)
async def update_skill(skill_id: str, payload: UpdateSkillRequest) -> Skill:
    _validate_tool_ids(payload.tool_ids)
    skill = store.update_skill(skill_id, payload.model_dump())
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str) -> dict[str, bool]:
    return {"deleted": store.delete_skill(skill_id)}


def _validate_tool_ids(tool_ids: list[str] | None) -> None:
    unknown = sorted(set(tool_ids or []) - tool_registry.tool_ids())
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown tool IDs: {', '.join(unknown)}")


@app.post("/api/retrieval/search", response_model=RetrievalSearchResponse)
async def retrieval_search(payload: RetrievalSearchRequest) -> RetrievalSearchResponse:
    sources, diagnostics = await rag_service.retrieve(
        query=payload.query,
        file_ids=payload.file_ids,
        knowledge_ids=payload.knowledge_ids,
        top_k=payload.top_k,
        context_mode=payload.context_mode,
        retrieval_mode=payload.retrieval_mode,
        max_context_chars=payload.max_context_chars,
        max_sources_per_file=payload.max_sources_per_file,
        file_selection_mode=payload.file_selection_mode,
        file_selection_limit=payload.file_selection_limit,
        source_window=payload.source_window,
        diversity=payload.diversity,
        mmr_lambda=payload.mmr_lambda,
    )
    return RetrievalSearchResponse(query=payload.query, sources=sources, diagnostics=diagnostics)


@app.post("/api/retrieval/source-pack", response_model=SourcePackResponse)
async def retrieval_source_pack(payload: RetrievalSearchRequest) -> SourcePackResponse:
    sources, diagnostics = await rag_service.retrieve(
        query=payload.query,
        file_ids=payload.file_ids,
        knowledge_ids=payload.knowledge_ids,
        top_k=payload.top_k,
        context_mode=payload.context_mode,
        retrieval_mode=payload.retrieval_mode,
        max_context_chars=payload.max_context_chars,
        max_sources_per_file=payload.max_sources_per_file,
        file_selection_mode=payload.file_selection_mode,
        file_selection_limit=payload.file_selection_limit,
        source_window=payload.source_window,
        diversity=payload.diversity,
        mmr_lambda=payload.mmr_lambda,
    )
    return build_source_pack(payload.query, sources, diagnostics)


@app.post("/api/retrieval/explain", response_model=RetrievalExplainResponse)
async def explain_retrieval(payload: RetrievalExplainRequest) -> RetrievalExplainResponse:
    return await rag_service.explain_retrieval(
        query=payload.query,
        file_ids=payload.file_ids,
        knowledge_ids=payload.knowledge_ids,
        top_k=payload.top_k,
        context_mode=payload.context_mode,
        retrieval_mode=payload.retrieval_mode,
        max_context_chars=payload.max_context_chars,
        max_sources_per_file=payload.max_sources_per_file,
        file_selection_mode=payload.file_selection_mode,
        file_selection_limit=payload.file_selection_limit,
        source_window=payload.source_window,
        diversity=payload.diversity,
        mmr_lambda=payload.mmr_lambda,
        candidate_limit=payload.candidate_limit,
    )


@app.post("/api/chats", response_model=CreateChatResponse)
async def create_chat(payload: CreateChatRequest) -> CreateChatResponse:
    chat = rag_service.create_chat(payload.title, file_ids=payload.file_ids, knowledge_ids=payload.knowledge_ids)
    return CreateChatResponse(id=chat.id, title=chat.title)


@app.post("/api/chats/import", response_model=ChatSession)
async def import_chat(payload: ImportChatRequest) -> ChatSession:
    try:
        return rag_service.import_chat(payload.chat, title=payload.title, preserve_ids=payload.preserve_ids)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/chats", response_model=list[ChatSession])
async def list_chats() -> list[ChatSession]:
    return store.list_chats()


@app.get("/api/chats/{chat_id}", response_model=ChatSession)
async def get_chat(chat_id: str) -> ChatSession:
    chat = store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.get("/api/chats/{chat_id}/export", response_model=ChatExportResponse)
async def export_chat(chat_id: str) -> ChatExportResponse:
    try:
        return rag_service.export_chat(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/chats/{chat_id}/messages/{message_id}/source-pack", response_model=SourcePackResponse)
async def get_chat_message_source_pack(chat_id: str, message_id: str) -> SourcePackResponse:
    chat = store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    message = next((item for item in chat.messages if item.id == message_id), None)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Only assistant messages have answer source packs")
    if not message.diagnostics:
        raise HTTPException(status_code=400, detail="Assistant message does not have retrieval diagnostics")
    query = message.diagnostics.original_query or message.retrieval_query
    return build_source_pack(query, message.sources, message.diagnostics)


@app.get("/api/chats/{chat_id}/messages/{message_id}/audit", response_model=AnswerAuditResponse)
async def audit_chat_message(chat_id: str, message_id: str) -> AnswerAuditResponse:
    try:
        return rag_service.audit_chat_message(chat_id, message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/chats/{chat_id}/messages/{message_id}/prompt", response_model=ChatMessagePromptResponse)
async def get_chat_message_prompt(chat_id: str, message_id: str) -> ChatMessagePromptResponse:
    try:
        return rag_service.get_message_prompt(chat_id, message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/chats/{chat_id}/messages/{message_id}/trace", response_model=ChatMessageTraceResponse)
async def get_chat_message_trace(chat_id: str, message_id: str) -> ChatMessageTraceResponse:
    try:
        return rag_service.get_message_trace(chat_id, message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/chats/{chat_id}/messages/{message_id}/feedback", response_model=ChatSession)
async def update_message_feedback(
    chat_id: str,
    message_id: str,
    payload: UpdateMessageFeedbackRequest,
) -> ChatSession:
    try:
        return rag_service.update_message_feedback(
            chat_id,
            message_id,
            rating=payload.rating,
            tags=payload.tags,
            comment=payload.comment,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/chats/{chat_id}/messages/{message_id}/feedback", response_model=ChatSession)
async def delete_message_feedback(chat_id: str, message_id: str) -> ChatSession:
    try:
        return rag_service.delete_message_feedback(chat_id, message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/chats/{chat_id}/context", response_model=ChatSession)
async def update_chat_context(chat_id: str, payload: UpdateChatContextRequest) -> ChatSession:
    try:
        return rag_service.update_chat_context(chat_id, payload.file_ids, payload.knowledge_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/chats/{chat_id}/answer-defaults", response_model=ChatSession)
async def update_chat_answer_defaults(chat_id: str, payload: UpdateChatAnswerDefaultsRequest) -> ChatSession:
    try:
        return rag_service.update_chat_answer_defaults(chat_id, payload.model_dump(exclude_unset=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/compact", response_model=ChatSession)
async def compact_chat(chat_id: str, payload: CompactChatRequest) -> ChatSession:
    try:
        return rag_service.compact_chat(chat_id, payload.keep_last)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/context/preview", response_model=ChatContextPreviewResponse)
async def preview_chat_context(chat_id: str, payload: ChatContextPreviewRequest) -> ChatContextPreviewResponse:
    try:
        return rag_service.preview_context(
            chat_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/context/suggest", response_model=ChatContextSuggestResponse)
async def suggest_chat_context(chat_id: str, payload: ChatContextSuggestRequest) -> ChatContextSuggestResponse:
    try:
        return rag_service.suggest_chat_context(
            chat_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            limit=payload.limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/context/apply-suggestions", response_model=ChatContextApplySuggestionsResponse)
async def apply_chat_context_suggestions(
    chat_id: str, payload: ChatContextApplySuggestionsRequest
) -> ChatContextApplySuggestionsResponse:
    try:
        return rag_service.apply_context_suggestions(
            chat_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            limit=payload.limit,
            replace=payload.replace,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/retrieval/explain", response_model=RetrievalExplainResponse)
async def explain_chat_retrieval(chat_id: str, payload: ChatRetrievalExplainRequest) -> RetrievalExplainResponse:
    opts = answer_options(chat_id, payload)
    try:
        return await rag_service.explain_chat_retrieval(
            chat_id=chat_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            top_k=opts["top_k"],
            use_rag=opts["use_rag"],
            context_mode=opts["context_mode"],
            retrieval_mode=opts["retrieval_mode"],
            max_context_chars=opts["max_context_chars"],
            max_sources_per_file=opts["max_sources_per_file"],
            file_selection_mode=opts["file_selection_mode"],
            file_selection_limit=opts["file_selection_limit"],
            source_window=opts["source_window"],
            diversity=opts["diversity"],
            mmr_lambda=opts["mmr_lambda"],
            candidate_limit=payload.candidate_limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/messages/{message_id}/prune", response_model=ChatSession)
async def prune_chat_messages(chat_id: str, message_id: str, payload: PruneChatMessagesRequest) -> ChatSession:
    try:
        return rag_service.prune_chat_messages(chat_id, message_id, include_selected=payload.include_selected)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/messages/preview", response_model=ChatAnswerPreviewResponse)
async def preview_chat_answer(chat_id: str, payload: ChatRequest) -> ChatAnswerPreviewResponse:
    opts = answer_options(chat_id, payload)
    try:
        return await rag_service.preview_answer(
            chat_id=chat_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            top_k=opts["top_k"],
            use_rag=opts["use_rag"],
            context_mode=opts["context_mode"],
            retrieval_mode=opts["retrieval_mode"],
            max_context_chars=opts["max_context_chars"],
            max_sources_per_file=opts["max_sources_per_file"],
            file_selection_mode=opts["file_selection_mode"],
            file_selection_limit=opts["file_selection_limit"],
            source_window=opts["source_window"],
            diversity=opts["diversity"],
            mmr_lambda=opts["mmr_lambda"],
            minimum_answerability=opts["minimum_answerability"],
            system_prompt=opts["system_prompt"],
            rag_template=opts["rag_template"],
            use_tools=opts["use_tools"],
            tool_ids=opts["tool_ids"],
            use_skills=opts["use_skills"],
            skill_ids=opts["skill_ids"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/memories", response_model=ChatSession)
async def add_chat_memory(chat_id: str, payload: CreateChatMemoryRequest) -> ChatSession:
    try:
        return rag_service.add_chat_memory(chat_id, payload.content, payload.source_message)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/chats/{chat_id}/memories/{memory_id}", response_model=ChatSession)
async def delete_chat_memory(chat_id: str, memory_id: str) -> ChatSession:
    try:
        return rag_service.delete_chat_memory(chat_id, memory_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/messages", response_model=ChatResponse)
async def ask_chat(chat_id: str, payload: ChatRequest) -> ChatResponse:
    opts = answer_options(chat_id, payload)
    try:
        message, sources, tool_results, skills, retrieval_query, diagnostics, grounding = await rag_service.ask(
            chat_id=chat_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            top_k=opts["top_k"],
            use_rag=opts["use_rag"],
            context_mode=opts["context_mode"],
            retrieval_mode=opts["retrieval_mode"],
            max_context_chars=opts["max_context_chars"],
            max_sources_per_file=opts["max_sources_per_file"],
            file_selection_mode=opts["file_selection_mode"],
            file_selection_limit=opts["file_selection_limit"],
            source_window=opts["source_window"],
            diversity=opts["diversity"],
            mmr_lambda=opts["mmr_lambda"],
            minimum_answerability=opts["minimum_answerability"],
            system_prompt=opts["system_prompt"],
            rag_template=opts["rag_template"],
            use_tools=opts["use_tools"],
            tool_ids=opts["tool_ids"],
            use_skills=opts["use_skills"],
            skill_ids=opts["skill_ids"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatResponse(
        chat_id=chat_id,
        message=message,
        sources=sources,
        tool_results=tool_results,
        skills=skills,
        retrieval_query=retrieval_query,
        diagnostics=diagnostics,
        grounding=grounding,
    )


@app.post("/api/chats/{chat_id}/messages/regenerate", response_model=ChatResponse)
async def regenerate_chat_answer(chat_id: str, payload: RegenerateChatRequest) -> ChatResponse:
    opts = answer_options(chat_id, payload)
    try:
        message, sources, tool_results, skills, retrieval_query, diagnostics, grounding = await rag_service.regenerate_last(
            chat_id=chat_id,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            top_k=opts["top_k"],
            use_rag=opts["use_rag"],
            context_mode=opts["context_mode"],
            retrieval_mode=opts["retrieval_mode"],
            max_context_chars=opts["max_context_chars"],
            max_sources_per_file=opts["max_sources_per_file"],
            file_selection_mode=opts["file_selection_mode"],
            file_selection_limit=opts["file_selection_limit"],
            source_window=opts["source_window"],
            diversity=opts["diversity"],
            mmr_lambda=opts["mmr_lambda"],
            minimum_answerability=opts["minimum_answerability"],
            system_prompt=opts["system_prompt"],
            rag_template=opts["rag_template"],
            use_tools=opts["use_tools"],
            tool_ids=opts["tool_ids"],
            use_skills=opts["use_skills"],
            skill_ids=opts["skill_ids"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(
        chat_id=chat_id,
        message=message,
        sources=sources,
        tool_results=tool_results,
        skills=skills,
        retrieval_query=retrieval_query,
        diagnostics=diagnostics,
        grounding=grounding,
    )


@app.post("/api/chats/{chat_id}/messages/{message_id}/rerun", response_model=ChatResponse)
async def rerun_chat_from_message(chat_id: str, message_id: str, payload: RegenerateChatRequest) -> ChatResponse:
    opts = answer_options(chat_id, payload)
    try:
        message, sources, tool_results, skills, retrieval_query, diagnostics, grounding = await rag_service.rerun_from_message(
            chat_id=chat_id,
            message_id=message_id,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            top_k=opts["top_k"],
            use_rag=opts["use_rag"],
            context_mode=opts["context_mode"],
            retrieval_mode=opts["retrieval_mode"],
            max_context_chars=opts["max_context_chars"],
            max_sources_per_file=opts["max_sources_per_file"],
            file_selection_mode=opts["file_selection_mode"],
            file_selection_limit=opts["file_selection_limit"],
            source_window=opts["source_window"],
            diversity=opts["diversity"],
            mmr_lambda=opts["mmr_lambda"],
            minimum_answerability=opts["minimum_answerability"],
            system_prompt=opts["system_prompt"],
            rag_template=opts["rag_template"],
            use_tools=opts["use_tools"],
            tool_ids=opts["tool_ids"],
            use_skills=opts["use_skills"],
            skill_ids=opts["skill_ids"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(
        chat_id=chat_id,
        message=message,
        sources=sources,
        tool_results=tool_results,
        skills=skills,
        retrieval_query=retrieval_query,
        diagnostics=diagnostics,
        grounding=grounding,
    )


@app.post("/api/chats/{chat_id}/messages/{message_id}/edit", response_model=ChatResponse)
async def edit_chat_message(chat_id: str, message_id: str, payload: EditChatMessageRequest) -> ChatResponse:
    opts = answer_options(chat_id, payload)
    try:
        message, sources, tool_results, skills, retrieval_query, diagnostics, grounding = await rag_service.edit_user_message(
            chat_id=chat_id,
            message_id=message_id,
            message=payload.message,
            file_ids=payload.file_ids,
            knowledge_ids=payload.knowledge_ids,
            top_k=opts["top_k"],
            use_rag=opts["use_rag"],
            context_mode=opts["context_mode"],
            retrieval_mode=opts["retrieval_mode"],
            max_context_chars=opts["max_context_chars"],
            max_sources_per_file=opts["max_sources_per_file"],
            file_selection_mode=opts["file_selection_mode"],
            file_selection_limit=opts["file_selection_limit"],
            source_window=opts["source_window"],
            diversity=opts["diversity"],
            mmr_lambda=opts["mmr_lambda"],
            minimum_answerability=opts["minimum_answerability"],
            system_prompt=opts["system_prompt"],
            rag_template=opts["rag_template"],
            use_tools=opts["use_tools"],
            tool_ids=opts["tool_ids"],
            use_skills=opts["use_skills"],
            skill_ids=opts["skill_ids"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(
        chat_id=chat_id,
        message=message,
        sources=sources,
        tool_results=tool_results,
        skills=skills,
        retrieval_query=retrieval_query,
        diagnostics=diagnostics,
        grounding=grounding,
    )


@app.post("/api/chats/{chat_id}/messages/stream")
async def ask_chat_stream(chat_id: str, payload: ChatRequest) -> StreamingResponse:
    async def events():
        opts = answer_options(chat_id, payload)
        try:
            async for event, data in rag_service.stream_answer(
                chat_id=chat_id,
                message=payload.message,
                file_ids=payload.file_ids,
                knowledge_ids=payload.knowledge_ids,
                top_k=opts["top_k"],
                use_rag=opts["use_rag"],
                context_mode=opts["context_mode"],
                retrieval_mode=opts["retrieval_mode"],
                max_context_chars=opts["max_context_chars"],
                max_sources_per_file=opts["max_sources_per_file"],
                file_selection_mode=opts["file_selection_mode"],
                file_selection_limit=opts["file_selection_limit"],
                source_window=opts["source_window"],
                diversity=opts["diversity"],
                mmr_lambda=opts["mmr_lambda"],
                minimum_answerability=opts["minimum_answerability"],
                system_prompt=opts["system_prompt"],
                rag_template=opts["rag_template"],
                use_tools=opts["use_tools"],
                tool_ids=opts["tool_ids"],
                use_skills=opts["use_skills"],
                skill_ids=opts["skill_ids"],
            ):
                if event == "retrieval":
                    prepared = data
                    yield sse(
                        "retrieval",
                        {
                            "query": prepared.retrieval_query,
                            "sources": [source.model_dump() for source in prepared.sources],
                            "diagnostics": prepared.diagnostics.model_dump(),
                            "skills": [skill.model_dump(mode="json") for skill in prepared.skills],
                        },
                    )
                elif event == "tools":
                    yield sse("tools", {"results": [result.model_dump() for result in data]})
                elif event == "token":
                    yield sse("token", {"text": data})
                elif event == "done":
                    message, sources, tool_results, skills, retrieval_query, diagnostics, grounding = data
                    yield sse(
                        "done",
                        {
                            "message": message.model_dump(mode="json"),
                            "sources": [source.model_dump() for source in sources],
                            "tool_results": [result.model_dump() for result in tool_results],
                            "skills": [skill.model_dump(mode="json") for skill in skills],
                            "retrieval_query": retrieval_query,
                            "diagnostics": diagnostics.model_dump(),
                            "grounding": grounding.model_dump(),
                        },
                    )
        except KeyError as exc:
            yield sse("error", {"detail": str(exc)})
            return

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict[str, bool]:
    return {"deleted": store.delete_chat(chat_id)}


def answer_options(chat_id: str, payload) -> dict:
    chat = store.get_chat(chat_id)
    explicit = payload.model_fields_set
    defaults = chat.answer_defaults if chat else None
    options = {}
    for field in ANSWER_OPTION_FIELDS:
        value = getattr(payload, field)
        if field not in explicit and defaults is not None:
            default_value = getattr(defaults, field)
            if default_value is not None:
                value = default_value
        options[field] = value
    return options


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
