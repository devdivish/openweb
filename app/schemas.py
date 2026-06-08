from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Source(BaseModel):
    file_id: str
    filename: str
    chunk_id: str
    chunk_index: int = 0
    context_start_index: int = 0
    context_end_index: int = 0
    start_char: int = 0
    end_char: int = 0
    score: float
    query_hits: int = 1
    text: str


class FileChunk(BaseModel):
    id: str
    file_id: str
    filename: str
    index: int
    start_char: int = 0
    end_char: int = 0
    text: str
    text_chars: int


class FileChunkWindowResponse(BaseModel):
    file_id: str
    filename: str
    target_index: int
    start_index: int
    end_index: int
    has_previous: bool
    has_next: bool
    chunks: list[FileChunk]
    context_text: str


class FileTextResponse(BaseModel):
    file_id: str
    filename: str
    start_char: int
    end_char: int
    total_chars: int
    text: str


class FileSummaryResponse(BaseModel):
    file_id: str
    filename: str
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    text_chars: int
    chunk_count: int


class FileOverview(BaseModel):
    id: str
    filename: str
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    text_chars: int
    chunk_count: int


class FileSearchItem(BaseModel):
    file: FileRecord
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class FileSearchResponse(BaseModel):
    query: str
    total_count: int
    items: list[FileSearchItem]


class RetrievalDiagnostics(BaseModel):
    original_query: str
    planned_queries: list[str]
    context_mode: str
    effective_context_mode: str = ""
    retrieval_mode: str
    candidate_count: int
    selected_count: int
    total_context_chars: int
    file_selection_mode: str = "all"
    candidate_file_ids: list[str] = Field(default_factory=list)
    routed_file_ids: list[str] = Field(default_factory=list)
    source_window: int = 0
    diversity: str = "relevance"
    mmr_lambda: float = 0.7
    top_source_score: float = 0.0
    average_source_score: float = 0.0
    query_term_coverage: float = 0.0
    answerability: Literal["none", "low", "medium", "high"] = "none"
    warnings: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool_id: str
    name: str
    input: str
    output: str


class ToolSpec(BaseModel):
    id: str
    name: str
    description: str


class Skill(BaseModel):
    id: str
    name: str
    description: str = ""
    instruction: str
    triggers: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CreateSkillRequest(BaseModel):
    name: str
    instruction: str
    description: str = ""
    triggers: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    enabled: bool = True


class UpdateSkillRequest(BaseModel):
    name: str | None = None
    instruction: str | None = None
    description: str | None = None
    triggers: list[str] | None = None
    tool_ids: list[str] | None = None
    enabled: bool | None = None


class SourceCitation(BaseModel):
    marker: str
    source_index: int
    file_id: str
    filename: str
    chunk_id: str
    chunk_index: int = 0
    context_start_index: int = 0
    context_end_index: int = 0
    start_char: int = 0
    end_char: int = 0


class AnswerGrounding(BaseModel):
    has_sources: bool = False
    cited_source_count: int = 0
    uncited_source_count: int = 0
    missing_citation_count: int = 0
    citations: list[SourceCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MessageFeedback(BaseModel):
    rating: Literal["up", "down"] | None = None
    tags: list[str] = Field(default_factory=list)
    comment: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UpdateMessageFeedbackRequest(BaseModel):
    rating: Literal["up", "down"] | None = None
    tags: list[str] = Field(default_factory=list)
    comment: str = ""


class FeedbackListItem(BaseModel):
    chat_id: str
    chat_title: str
    message_id: str
    message_created_at: datetime
    question: str = ""
    answer: str
    feedback: MessageFeedback
    retrieval_query: str = ""
    diagnostics: RetrievalDiagnostics | None = None
    grounding: AnswerGrounding | None = None
    source_count: int = 0


class FeedbackListResponse(BaseModel):
    total_count: int
    items: list[FeedbackListItem]


class AnswerAuditSentence(BaseModel):
    index: int
    text: str
    cited_markers: list[str] = Field(default_factory=list)
    matched_source_indexes: list[int] = Field(default_factory=list)
    support_score: float = 0.0
    status: Literal["supported", "weak", "unsupported"] = "unsupported"


class AnswerAuditResponse(BaseModel):
    message_id: str = ""
    answer_supported: bool = False
    support_score: float = 0.0
    sentence_count: int = 0
    supported_count: int = 0
    weak_count: int = 0
    unsupported_count: int = 0
    grounding: AnswerGrounding
    sentences: list[AnswerAuditSentence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnswerQuality(BaseModel):
    status: Literal["supported", "weak", "unsupported", "unknown"] = "unknown"
    answer_supported: bool = False
    support_score: float = 0.0
    sentence_count: int = 0
    supported_count: int = 0
    weak_count: int = 0
    unsupported_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ProviderStatus(BaseModel):
    provider: str
    configured: bool
    model: str = ""
    base_url: str = ""


class RetrievalSettingsResponse(BaseModel):
    top_k: int
    relevance_threshold: float
    chunk_size: int
    chunk_overlap: int
    full_context_max_chars: int
    chat_context_messages: int
    chat_memory_items: int
    chat_memory_context_chars: int


class StorageStatsResponse(BaseModel):
    file_count: int
    chunk_count: int
    chat_count: int
    message_count: int
    knowledge_count: int
    skill_count: int
    total_file_bytes: int
    total_file_text_chars: int
    total_chunk_text_chars: int


class BackendStatusResponse(BaseModel):
    status: str = "ok"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding_provider: ProviderStatus
    llm_provider: ProviderStatus
    retrieval: RetrievalSettingsResponse
    storage: StorageStatsResponse
    openai_api_key_configured: bool = False


class FileRecord(BaseModel):
    id: str
    filename: str
    content_type: str | None = None
    path: str
    bytes: int
    text_chars: int
    chunk_count: int
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    created_at: datetime


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    file_ids: list[str] = Field(default_factory=list)


class CreateKnowledgeRequest(BaseModel):
    name: str
    description: str = ""
    file_ids: list[str] = Field(default_factory=list)


class UpdateKnowledgeFilesRequest(BaseModel):
    file_ids: list[str]


class KnowledgeFileUploadResponse(BaseModel):
    file: FileRecord
    knowledge: KnowledgeBase


class BatchFileUploadResponse(BaseModel):
    files: list[FileRecord]


class ReindexFailure(BaseModel):
    file_id: str
    error: str


class ReindexFilesRequest(BaseModel):
    file_ids: list[str] | None = None
    knowledge_ids: list[str] | None = None


class ReindexFilesResponse(BaseModel):
    requested_count: int
    reindexed_count: int
    files: list[FileRecord]
    failures: list[ReindexFailure] = Field(default_factory=list)


class KnowledgeBatchFileUploadResponse(BaseModel):
    files: list[FileRecord]
    knowledge: KnowledgeBase


class PromptMessage(BaseModel):
    role: str
    content: str


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sources: list[Source] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    diagnostics: RetrievalDiagnostics | None = None
    grounding: AnswerGrounding | None = None
    answer_quality: AnswerQuality = Field(default_factory=AnswerQuality)
    feedback: MessageFeedback | None = None
    prompt_messages: list[PromptMessage] = Field(default_factory=list)
    prompt_chars: int = 0
    system_prompt: str = ""
    rag_template: str = ""


class ChatMemoryItem(BaseModel):
    id: str
    content: str
    source_message: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatAnswerDefaults(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_rag: bool | None = None
    context_mode: Literal["auto", "rag", "full", "none"] | None = None
    retrieval_mode: Literal["hybrid", "vector", "keyword"] | None = None
    max_context_chars: int | None = Field(default=None, ge=500, le=50000)
    max_sources_per_file: int | None = Field(default=None, ge=1, le=20)
    file_selection_mode: Literal["all", "auto"] | None = None
    file_selection_limit: int | None = Field(default=None, ge=1, le=50)
    source_window: int | None = Field(default=None, ge=0, le=3)
    diversity: Literal["relevance", "mmr"] | None = None
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_answerability: Literal["none", "low", "medium", "high"] | None = None
    system_prompt: str | None = None
    rag_template: str | None = None
    use_tools: bool | None = None
    tool_ids: list[str] | None = None
    use_skills: bool | None = None
    skill_ids: list[str] | None = None


class ChatSession(BaseModel):
    id: str
    title: str = "New chat"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: str = ""
    memories: list[ChatMemoryItem] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    answer_defaults: ChatAnswerDefaults = Field(default_factory=ChatAnswerDefaults)
    messages: list[ChatMessage] = Field(default_factory=list)


class CreateChatRequest(BaseModel):
    title: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)


class CreateChatResponse(BaseModel):
    id: str
    title: str


class ChatExportResponse(BaseModel):
    version: str = "1"
    exported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chat: ChatSession


class ImportChatRequest(BaseModel):
    chat: ChatSession
    title: str | None = None
    preserve_ids: bool = False


class UpdateChatContextRequest(BaseModel):
    file_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)


class UpdateChatAnswerDefaultsRequest(ChatAnswerDefaults):
    pass


class CompactChatRequest(BaseModel):
    keep_last: int | None = Field(default=None, ge=0, le=100)


class PruneChatMessagesRequest(BaseModel):
    include_selected: bool = True


class CreateChatMemoryRequest(BaseModel):
    content: str
    source_message: str = ""


class ChatContextPreviewRequest(BaseModel):
    message: str = ""
    file_ids: list[str] | None = None
    knowledge_ids: list[str] | None = None


class ChatContextSuggestRequest(ChatContextPreviewRequest):
    limit: int = Field(default=5, ge=1, le=20)


class ChatContextApplySuggestionsRequest(ChatContextSuggestRequest):
    replace: bool = False


class ChatContextSuggestResponse(BaseModel):
    chat_id: str
    message: str
    default_file_ids: list[str]
    default_knowledge_ids: list[str]
    requested_file_ids: list[str]
    requested_knowledge_ids: list[str]
    candidate_file_ids: list[str]
    suggested_file_ids: list[str]
    suggestions: list[FileSearchItem]
    files: list[FileOverview] = Field(default_factory=list)


class ChatContextApplySuggestionsResponse(BaseModel):
    chat: ChatSession
    suggestion: ChatContextSuggestResponse
    applied_file_ids: list[str]
    replaced: bool = False


class ChatContextPreviewResponse(BaseModel):
    chat_id: str
    title: str
    summary: str
    memories: list[ChatMemoryItem]
    memory_context: str
    rolling_messages: list[ChatMessage]
    planned_queries: list[str]
    retrieval_query: str
    default_file_ids: list[str]
    default_knowledge_ids: list[str]
    requested_file_ids: list[str]
    requested_knowledge_ids: list[str]
    resolved_file_ids: list[str]
    files: list[FileOverview] = Field(default_factory=list)
    context_message_count: int
    context_chars: int


class ChatRequest(BaseModel):
    message: str
    file_ids: list[str] | None = None
    knowledge_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_rag: bool = True
    context_mode: Literal["auto", "rag", "full", "none"] = "rag"
    retrieval_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    max_context_chars: int | None = Field(default=None, ge=500, le=50000)
    max_sources_per_file: int | None = Field(default=None, ge=1, le=20)
    file_selection_mode: Literal["all", "auto"] = "all"
    file_selection_limit: int = Field(default=5, ge=1, le=50)
    source_window: int = Field(default=0, ge=0, le=3)
    diversity: Literal["relevance", "mmr"] = "relevance"
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    minimum_answerability: Literal["none", "low", "medium", "high"] = "none"
    system_prompt: str | None = None
    rag_template: str | None = None
    use_tools: bool = True
    tool_ids: list[str] | None = None
    use_skills: bool = True
    skill_ids: list[str] | None = None


class ChatRetrievalExplainRequest(ChatRequest):
    candidate_limit: int = Field(default=20, ge=1, le=100)


class RegenerateChatRequest(BaseModel):
    file_ids: list[str] | None = None
    knowledge_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_rag: bool = True
    context_mode: Literal["auto", "rag", "full", "none"] = "rag"
    retrieval_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    max_context_chars: int | None = Field(default=None, ge=500, le=50000)
    max_sources_per_file: int | None = Field(default=None, ge=1, le=20)
    file_selection_mode: Literal["all", "auto"] = "all"
    file_selection_limit: int = Field(default=5, ge=1, le=50)
    source_window: int = Field(default=0, ge=0, le=3)
    diversity: Literal["relevance", "mmr"] = "relevance"
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    minimum_answerability: Literal["none", "low", "medium", "high"] = "none"
    system_prompt: str | None = None
    rag_template: str | None = None
    use_tools: bool = True
    tool_ids: list[str] | None = None
    use_skills: bool = True
    skill_ids: list[str] | None = None


class EditChatMessageRequest(RegenerateChatRequest):
    message: str


class RetrievalSearchRequest(BaseModel):
    query: str
    file_ids: list[str] | None = None
    knowledge_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    context_mode: Literal["auto", "rag", "full"] = "rag"
    retrieval_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    max_context_chars: int | None = Field(default=None, ge=500, le=50000)
    max_sources_per_file: int | None = Field(default=None, ge=1, le=50)
    file_selection_mode: Literal["all", "auto"] = "all"
    file_selection_limit: int = Field(default=5, ge=1, le=50)
    source_window: int = Field(default=0, ge=0, le=3)
    diversity: Literal["relevance", "mmr"] = "relevance"
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)


class RetrievalExplainRequest(RetrievalSearchRequest):
    candidate_limit: int = Field(default=20, ge=1, le=100)


class RetrievalSearchResponse(BaseModel):
    query: str
    sources: list[Source]
    diagnostics: RetrievalDiagnostics


class RetrievalCandidate(BaseModel):
    rank: int
    selected: bool
    source: Source
    matched_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SourcePackItem(BaseModel):
    marker: str
    source: Source
    excerpt: str
    matched_terms: list[str] = Field(default_factory=list)


class SourcePackFile(BaseModel):
    file_id: str
    filename: str
    source_count: int
    top_score: float
    markers: list[str]
    sources: list[SourcePackItem]


class SourcePackResponse(BaseModel):
    query: str
    diagnostics: RetrievalDiagnostics
    sources: list[Source]
    files: list[SourcePackFile]
    context_text: str


class RetrievalExplainResponse(BaseModel):
    query: str
    retrieval_query: str = ""
    diagnostics: RetrievalDiagnostics
    source_pack: SourcePackResponse
    candidates: list[RetrievalCandidate]


class ChatAnswerPreviewResponse(BaseModel):
    chat_id: str
    message: str
    retrieval_query: str
    diagnostics: RetrievalDiagnostics
    source_pack: SourcePackResponse
    tool_results: list[ToolResult]
    skills: list[Skill]
    prompt_messages: list[PromptMessage]
    system_prompt: str
    rag_template: str
    context_message_count: int
    prompt_chars: int
    would_learn_memories: list[str] = Field(default_factory=list)


class ChatMessagePromptResponse(BaseModel):
    chat_id: str
    message_id: str
    retrieval_query: str = ""
    prompt_messages: list[PromptMessage]
    prompt_chars: int
    system_prompt: str = ""
    rag_template: str = ""


class ChatMessageTraceResponse(BaseModel):
    chat_id: str
    chat_title: str
    message_id: str
    question: str = ""
    answer: ChatMessage
    retrieval_query: str = ""
    diagnostics: RetrievalDiagnostics | None = None
    source_pack: SourcePackResponse | None = None
    audit: AnswerAuditResponse | None = None
    prompt: ChatMessagePromptResponse | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    feedback: MessageFeedback | None = None


class ChatResponse(BaseModel):
    chat_id: str
    message: ChatMessage
    sources: list[Source]
    tool_results: list[ToolResult]
    skills: list[Skill]
    retrieval_query: str
    diagnostics: RetrievalDiagnostics
    grounding: AnswerGrounding


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    llm_provider: str
