# OpenWebUI-Style Lean Backend

This backend recreates the core behavior you asked for without copying Open WebUI's whole system:

- file upload and text extraction
- chunking and file-based RAG
- local persistent vector search
- SQLite-backed persistence for files, chunks, chats, and messages
- hybrid retrieval with SQLite FTS5/BM25 plus vector scoring
- follow-up aware query planning that uses recent turns, compact summaries, and saved memory
- optional file-level auto-routing before chunk retrieval for large chat or knowledge contexts
- source reranking, deduplication, and context-budget compression
- knowledge bases for grouping files and querying collections
- configurable system prompts and RAG prompt templates
- chat sessions with rolling context, compact summaries, and durable memories
- source-aware prompt assembly
- citation grounding metadata for mapping answer citations back to chunks
- persisted answer quality verdicts derived from source-audit checks
- persisted per-answer retrieval traces for later audit/debugging
- answer generation through OpenAI, Ollama, or a local extractive fallback
- simple tool calling with safe built-ins
- persistent prompt skills with trigger-based activation
- provider-aware streaming response events over server-sent events

## Run

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

## Main Endpoints

- `GET /api/status` returns provider configuration, retrieval settings, and storage/index counts without exposing secrets.
- `GET /api/feedback` lists assistant-answer feedback across chats with optional `rating`, `tag`, and `limit` filters.
- `POST /api/files` uploads and indexes a file.
- `POST /api/files/batch` uploads and indexes multiple files.
- `GET /api/files` lists indexed files.
- `GET /api/files/search?q=...` ranks indexed files by filename, keywords, and summary, optionally scoped with `knowledge_id`.
- `GET /api/files/{file_id}/summary` returns the persisted local summary and keywords for one file.
- `GET /api/files/{file_id}/text` returns the extracted document text, optionally sliced with `start` and `end` character offsets.
- `GET /api/files/{file_id}/chunks` lists indexed chunks for one file in document order.
- `GET /api/files/{file_id}/chunks/{chunk_index}/window` returns a small chunk neighborhood around a citation target.
- `DELETE /api/files/{file_id}` removes an indexed file, its stored upload, and its chunks.
- `POST /api/files/{file_id}/reindex` rebuilds text chunks and embeddings for a stored file.
- `POST /api/files/reindex` rebuilds chunks and embeddings for all files, selected `file_ids`, or files resolved from `knowledge_ids`.
- `POST /api/knowledge` creates a named knowledge base.
- `GET /api/knowledge` lists knowledge bases.
- `GET /api/knowledge/{knowledge_id}` returns one knowledge base.
- `PUT /api/knowledge/{knowledge_id}/files` replaces file membership.
- `POST /api/knowledge/{knowledge_id}/files` adds files.
- `POST /api/knowledge/{knowledge_id}/files/upload` uploads, indexes, and attaches a file in one call.
- `POST /api/knowledge/{knowledge_id}/files/upload/batch` uploads, indexes, and attaches multiple files in one call.
- `DELETE /api/knowledge/{knowledge_id}/files/{file_id}` removes one file from a knowledge base.
- `POST /api/knowledge/{knowledge_id}/reindex` rebuilds chunks and embeddings for every file in a knowledge base.
- `DELETE /api/knowledge/{knowledge_id}` deletes a knowledge base.
- `POST /api/chats` creates a chat session.
- `POST /api/chats/import` imports an exported chat session, duplicating chat/message/memory IDs by default.
- `PUT /api/chats/{chat_id}/context` sets default `file_ids` and `knowledge_ids` for a chat.
- `PUT /api/chats/{chat_id}/answer-defaults` sets durable RAG/generation defaults for a chat.
- `POST /api/chats/{chat_id}/compact` summarizes older chat messages now and keeps the last `keep_last` messages.
- `POST /api/chats/{chat_id}/context/preview` shows rolling context, memory, summary, planned retrieval queries, resolved files, and file overviews for a draft message.
- `POST /api/chats/{chat_id}/context/suggest` suggests relevant files for a draft message without mutating chat context.
- `POST /api/chats/{chat_id}/context/apply-suggestions` saves suggested files onto the chat context, merging by default or replacing when `replace=true`.
- `POST /api/chats/{chat_id}/retrieval/explain` explains retrieval for a draft chat message using chat context and saved defaults without saving a message.
- `POST /api/chats/{chat_id}/memories` adds a durable chat memory.
- `DELETE /api/chats/{chat_id}/memories/{memory_id}` removes a durable chat memory.
- `POST /api/chats/{chat_id}/messages` asks a question, retrieves relevant file chunks, maintains chat context, and returns an answer with sources.
- `POST /api/chats/{chat_id}/messages/preview` dry-runs retrieval, tools, skills, and prompt assembly without saving a message.
- `POST /api/chats/{chat_id}/messages/{message_id}/prune` removes a retained message branch from chat context.
- `POST /api/chats/{chat_id}/messages/regenerate` replaces the last assistant answer using the previous user message and fresh retrieval/generation options.
- `POST /api/chats/{chat_id}/messages/{message_id}/rerun` truncates after a selected user message and reruns it with fresh retrieval/generation options.
- `POST /api/chats/{chat_id}/messages/{message_id}/edit` truncates after a selected user message, replaces its text, and answers the edited text.
- `PUT /api/chats/{chat_id}/messages/{message_id}/feedback` records rating, tags, and a comment on an assistant message.
- `DELETE /api/chats/{chat_id}/messages/{message_id}/feedback` clears feedback from an assistant message.
- `POST /api/chats/{chat_id}/messages/stream` streams SSE events: `retrieval`, `tools`, provider `token` chunks, and `done`.
- `GET /api/chats/{chat_id}` returns session history.
- `GET /api/chats/{chat_id}/export` exports a chat with messages, memories, answer defaults, and persisted answer traces.
- `GET /api/chats/{chat_id}/messages/{message_id}/source-pack` returns citation-ready evidence stored on a past assistant answer.
- `GET /api/chats/{chat_id}/messages/{message_id}/audit` scores a past assistant answer against its persisted sources and citation grounding.
- `GET /api/chats/{chat_id}/messages/{message_id}/prompt` returns the persisted prompt snapshot for a past assistant answer.
- `GET /api/chats/{chat_id}/messages/{message_id}/trace` returns a combined saved-answer trace with question, answer, evidence, audit, prompt, tools, skills, and feedback.
- `DELETE /api/chats/{chat_id}` deletes a session.
- `POST /api/retrieval/search` runs retrieval without adding a chat message.
- `POST /api/retrieval/source-pack` runs retrieval and returns citation-ready grouped evidence.
- `POST /api/retrieval/explain` runs retrieval and returns selected sources plus ranked candidate chunks for debugging.
- `GET /api/tools` lists available built-in tools.
- `POST /api/skills` creates a reusable prompt skill, optionally with tool bindings.
- `GET /api/skills` lists skills.
- `GET /api/skills/{skill_id}` returns one skill.
- `PUT /api/skills/{skill_id}` updates one skill.
- `DELETE /api/skills/{skill_id}` deletes one skill.

## RAG Modes

Use `GET /api/status` to inspect the running backend. It reports the configured LLM and embedding providers, model names, whether provider credentials are configured, retrieval defaults such as chunk size and `top_k`, and storage stats such as file, chunk, chat, message, knowledge, and skill counts. API keys are never returned.

`ChatRequest.context_mode` controls file context:

- `auto`: use full context when the selected indexed files fit under `max_context_chars`; otherwise use RAG.
- `rag`: embed the query, search chunks, and inject top passages.
- `full`: concatenate indexed file chunks up to `FULL_CONTEXT_MAX_CHARS`.
- `none`: skip file context.

`use_rag=false` is kept as a compatibility shortcut for `context_mode=none`.

`retrieval_mode` controls how `context_mode=rag` searches chunks:

- `hybrid`: vector score plus SQLite FTS5/BM25 keyword rank.
- `vector`: vector and token-overlap scoring only.
- `keyword`: SQLite FTS5/BM25 only.

`max_context_chars` caps how much source text is selected for prompt context. Responses include `diagnostics` with:

- `original_query`
- `planned_queries`
- `context_mode`
- `effective_context_mode`
- `retrieval_mode`
- `candidate_count`
- `selected_count`
- `total_context_chars`
- `file_selection_mode`
- `candidate_file_ids`
- `routed_file_ids`
- `source_window`
- `diversity`
- `mmr_lambda`
- `top_source_score`
- `average_source_score`
- `query_term_coverage`
- `answerability`
- `warnings`

Chat responses also include `grounding`, which maps answer citation markers such as `[1]` back to retrieved file chunks. It reports:

- `has_sources`
- `cited_source_count`
- `uncited_source_count`
- `missing_citation_count`
- `citations` with file IDs, chunk IDs, chunk ranges, and `start_char`/`end_char` offsets
- `warnings`

Assistant messages also persist `answer_quality`, a compact verdict derived from the deterministic answer audit. It includes a `status` (`supported`, `weak`, `unsupported`, or `unknown`), support score, sentence counts, and warnings. Use `GET /api/chats/{chat_id}/messages/{message_id}/audit` when you need the full sentence-by-sentence breakdown.

Retrieved `sources` include `chunk_id`, `chunk_index`, `start_char`, and `end_char`, so a client can jump from a citation to the exact indexed chunk and highlight the source range in the extracted document text.

Use `GET /api/files/{file_id}/text?start=0&end=500` to fetch the normalized extracted text behind those offsets. Omit `start` and `end` to return the full extracted text for the stored file.

Each indexed file stores a local extractive `summary` and `keywords`. These are generated during upload and refreshed during reindex, so file lists and `GET /api/files/{file_id}/summary` can show a quick document overview without loading full text or calling an LLM.

Use `GET /api/files/search?q=refund+exceptions` to find likely documents before choosing chat context. File search ranks filename matches, keyword overlap, and summary overlap, and can be scoped to one knowledge base with `knowledge_id=...`.

Use `GET /api/files/{file_id}/chunks/{chunk_index}/window?window=1` to fetch only the citation target chunk plus nearby chunks. The response includes `start_index`, `end_index`, `has_previous`, `has_next`, the selected chunks with character offsets, and a formatted `context_text` block for source previews.

Use `POST /api/files/reindex` after changing chunk settings, embedding settings, or stored source files. Omit the body filters to rebuild every indexed file, send `file_ids` for a selected subset, or send `knowledge_ids` to rebuild the files attached to one or more knowledge bases. The response includes `requested_count`, `reindexed_count`, refreshed file records, and per-file failures, so one missing stored upload does not stop the whole batch. `POST /api/knowledge/{knowledge_id}/reindex` is the same workflow scoped to one knowledge base.

Use `POST /api/retrieval/source-pack` when a client needs inspectable evidence before generation or wants to render citations directly. It accepts the same body as `/api/retrieval/search` and returns raw `sources`, grouped `files`, per-source citation markers, matched query terms, excerpts, diagnostics, and a `context_text` block suitable for prompt injection or preview. The context text includes score, chunk range, character range, and chunk ID metadata for each source.

Use `POST /api/retrieval/explain` while tuning file QA. It accepts the same retrieval fields plus `candidate_limit`, returns the selected source pack, and lists ranked candidate chunks with `selected`, `matched_terms`, and short `reasons` so you can see which chunks were considered but did not make the final context budget.

Follow-up chat retrieval plans multiple queries from the current message, recent user turns, memories, and compact summaries. When the same chunk is found by multiple planned queries, retrieval fuses those hits, gives the chunk a small score boost, and exposes `query_hits` on the returned source. Source-pack context also includes `query_hits` so clients can see which evidence survived because several query variants agreed on it.

Set `source_window` on chat or retrieval requests to include neighboring chunks around each selected hit. For example, `source_window=1` keeps the citation anchored to the matched chunk while expanding the prompt text with one chunk before and one chunk after it. Expanded sources report `context_start_index`, `context_end_index`, and the expanded `start_char`/`end_char` range.

Set `diversity="mmr"` to use maximal marginal relevance when selecting sources. This helps broad multi-file questions by keeping highly relevant chunks while penalizing repeated near-duplicates. `mmr_lambda` controls the relevance/diversity tradeoff: higher values favor relevance, lower values favor variety.

`answerability` is a lightweight retrieval confidence signal derived from selected source scores and query-term coverage. It can be `none`, `low`, `medium`, or `high`. Use it to decide when the UI should warn that retrieved evidence may be weak.

Set `minimum_answerability` on chat requests to require a minimum retrieval confidence before generation. For example, `minimum_answerability="medium"` returns an insufficient-evidence answer when retrieval is `none` or `low`, while still preserving diagnostics and sources for inspection.

Pass `knowledge_ids` in chat or retrieval requests to search every file attached to those knowledge bases. `file_ids` and `knowledge_ids` can be combined; the backend resolves and deduplicates them before retrieval.

When the user query mentions an attached filename, retrieval automatically narrows context to matching files before vector/keyword search. For example, a query that names `beta-guide.txt` searches that file within the allowed chat, file, or knowledge context and records a diagnostics warning describing the filename-aware routing decision.

Use `max_sources_per_file` when asking across many files and you want broader coverage instead of several chunks from the same file occupying the whole context.

Set `file_selection_mode="auto"` when a chat or knowledge base has many attached files and you want the backend to choose likely documents before chunk retrieval. The file router ranks the allowed files using filenames, summaries, and keywords, then searches chunks only inside the top `file_selection_limit` files. Diagnostics include the original `candidate_file_ids`, the narrowed `routed_file_ids`, and a warning describing the routing decision. The default `file_selection_mode="all"` preserves the full allowed file set.

Chats can store default `file_ids` and `knowledge_ids`. Once attached, follow-up messages can omit those IDs and still retrieve from the saved chat context. Per-message IDs are merged with the chat defaults.

Chats can also store answer defaults with `PUT /api/chats/{chat_id}/answer-defaults`, including `top_k`, `use_rag`, `context_mode`, `retrieval_mode`, `max_context_chars`, `max_sources_per_file`, `file_selection_mode`, `file_selection_limit`, `source_window`, `diversity`, `mmr_lambda`, `minimum_answerability`, prompt overrides, tool settings, and skill settings. Message, preview, stream, regenerate, rerun, and edit requests inherit these defaults unless the request explicitly sends that field. Send `null` for a field in the defaults update to clear that saved default.

Use `GET /api/chats/{chat_id}/export` to serialize a chat with its messages, compact summary, durable memories, default file/knowledge context, answer defaults, and persisted retrieval traces. Use `POST /api/chats/import` with the exported `chat` object to restore it. Imports duplicate chat, message, and memory IDs by default so they do not overwrite local chats; set `preserve_ids=true` only when intentionally replaying an export into an empty matching environment.

Use `POST /api/chats/{chat_id}/context/preview` to inspect what the backend will use for a draft message before generating. It returns the compact summary, durable memory block, rolling messages, planned retrieval queries, default and requested context IDs, resolved file IDs after knowledge-base expansion, file summaries/keywords, and context character counts.

Use `POST /api/chats/{chat_id}/context/suggest` to choose likely files before answering. It ranks files within the chat's saved context plus optional requested files/knowledge bases, returns suggested file IDs and file overviews, and does not mutate the chat.

Use `POST /api/chats/{chat_id}/context/apply-suggestions` when the client wants to turn those ranked suggestions into saved chat context. It uses the same request shape as suggest plus `replace`; by default it appends new suggested files without duplicates, while `replace=true` makes the chat's `file_ids` exactly the suggested set. Knowledge-base defaults are left unchanged.

Use `POST /api/chats/{chat_id}/retrieval/explain` to debug retrieval for the next message in a real chat. It inherits the chat's saved answer defaults, saved file and knowledge context, rolling messages, compact summary, and durable memories, then returns the effective `retrieval_query`, selected source pack, and ranked candidate chunks without appending any messages.

Chats also keep durable memory items for facts and preferences that should survive rolling context compaction. The service auto-learns simple user facts such as "remember that ...", "my ... is ...", "I prefer ...", and "call me ...". You can also add or delete memories through the memory endpoints. Saved memories are injected into future prompt context alongside the compact conversation summary.

Use `POST /api/chats/{chat_id}/compact` to manually move older retained messages into the chat summary. Omit `keep_last` to use the backend rolling-window size, set it to a positive number to keep only that many most recent messages, or set `keep_last=0` to summarize the full retained history.

Assistant messages persist their answer trace: `retrieval_query`, `diagnostics`, `sources`, `grounding`, `answer_quality`, `prompt_messages`, prompt character count, system prompt, and RAG template. This lets `GET /api/chats/{chat_id}` show why an older answer was produced, not just the final text.

Use `POST /api/chats/{chat_id}/messages/preview` to dry-run the full answer setup for a draft message. It returns retrieval diagnostics, source pack, tool results, active skills, exact provider prompt messages, prompt character count, and memories that would be learned. It does not append chat messages or save auto-learned memories.

Each chat message has a stable `id`. Use `GET /api/chats/{chat_id}/messages/{message_id}/source-pack` for an assistant message to reconstruct the grouped evidence and prompt-ready context that supported that past answer.

Use `GET /api/chats/{chat_id}/messages/{message_id}/prompt` for an assistant message to inspect the exact provider prompt snapshot saved at generation time. This remains available even after later compaction, edits, or retrieval setting changes.

Use `GET /api/chats/{chat_id}/messages/{message_id}/trace` when a client needs the full replay bundle for a saved assistant answer. It combines the previous user question, assistant message, source pack, audit result, persisted prompt snapshot, tool outputs, active skills, and feedback into one response.

Use `GET /api/chats/{chat_id}/messages/{message_id}/audit` for a lightweight evidence check on a past assistant answer. It splits the answer into sentences, checks each sentence against cited sources when citations are present and all persisted sources otherwise, and returns `supported`, `weak`, or `unsupported` sentence statuses plus grounding warnings. This is deterministic and local; it does not call another LLM.

Use `PUT /api/chats/{chat_id}/messages/{message_id}/feedback` to attach human feedback to an assistant answer. The payload accepts `rating` (`up`, `down`, or `null`), `tags`, and `comment`; tags are normalized and deduplicated. Use `DELETE /api/chats/{chat_id}/messages/{message_id}/feedback` to clear it. Feedback is persisted on the assistant message and included in chat export/import.

Use `GET /api/feedback` to review rated answers across chats. It returns the chat ID/title, assistant message ID, previous user question, answer text, feedback, retrieval query, diagnostics, grounding, and source count. Filter with `rating=up` or `rating=down`, `tag=...`, and `limit=...` when building an evaluation or tuning workflow.

Use `POST /api/chats/{chat_id}/messages/{message_id}/prune` to remove a retained branch from context without generating a new answer. By default it removes the selected message and everything after it. Send `include_selected=false` to keep the selected message and remove only later messages.

Use `POST /api/chats/{chat_id}/messages/regenerate` after a chat answer to rerun the previous user message with different retrieval or generation settings, such as `retrieval_mode`, `source_window`, `minimum_answerability`, prompt overrides, tools, or skills. The endpoint replaces the last user/assistant turn instead of appending a duplicate follow-up.

Use `POST /api/chats/{chat_id}/messages/{message_id}/rerun` when an earlier retained user turn should become the new branch point. The backend keeps messages before that user turn, discards the selected turn and all later messages, then asks the same user text again with the supplied retrieval/generation settings.

Use `POST /api/chats/{chat_id}/messages/{message_id}/edit` to edit an earlier retained user turn. The backend keeps messages before that turn, discards the selected turn and all later messages, then asks the edited text with the supplied retrieval/generation settings.

The streaming endpoint performs retrieval, tool execution, and skill selection first, emits those metadata events, then streams answer tokens from the configured provider. OpenAI and Ollama use their native streaming APIs; the local extractive fallback emits its generated answer through the same `token` event shape. The final `done` event includes the persisted assistant message plus sources, diagnostics, grounding, tool results, and active skills.

For short follow-up questions, retrieval planning uses recent user turns plus the compact summary and saved memory system context. That helps questions like "And exceptions?" stay attached to the earlier topic even after older chat turns have been compacted.

## Prompting

Global prompt defaults can be set with:

- `SYSTEM_PROMPT`
- `RAG_TEMPLATE`

`RAG_TEMPLATE` supports:

- `{context}`
- `{question}`

Chat requests can override both with `system_prompt` and `rag_template`. Prompt source blocks include file ID, chunk ID, chunk index, chunk range, character range, and score attributes so providers can keep answers tied to inspectable evidence.

## Built-In Tools

The backend can automatically run small safe tools before answer generation:

- `calculator`
- `list_files`
- `file_stats`
- `time`

Pass `tool_ids` to explicitly run tools, or set `use_tools=false` to disable tools for a request.

## Skills

Skills are persistent prompt modules. Each skill has a name, instruction text, optional description, trigger phrases, optional `tool_ids`, and an enabled flag. When `use_skills=true`, matching enabled skills are injected into the system prompt before answer generation. Pass `skill_ids` on a chat request to explicitly activate specific skills.

When a matching skill has `tool_ids`, those built-in tools run even if the user message would not normally trigger them. This lets a skill behave like a small executable workflow, not just prompt text. The skill API rejects unknown tool IDs so saved skill configs remain runnable.

Chat responses include the active `skills`, and assistant messages persist `skill_ids` so older answers can be audited.

## Model Providers

The default is `LLM_PROVIDER=extractive`, which lets the backend answer from retrieved passages without a paid model. For better answers:

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="..."
```

or:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_CHAT_MODEL="llama3.1"
```

Embedding also defaults to a deterministic local hashing embedder. It is not as smart as a transformer model, but it is fast, dependency-light, persistent, and good enough for the backend architecture. You can switch later to OpenAI or Ollama embeddings with:

```powershell
$env:EMBEDDING_PROVIDER="openai"
```

or:

```powershell
$env:EMBEDDING_PROVIDER="ollama"
```
