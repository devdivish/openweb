# Backend Iterations

This log records the concrete changes made during each goal-continuation iteration.

## 2026-06-08 - Chat export/import

- Added chat export with `GET /api/chats/{chat_id}/export`.
- Added chat import with `POST /api/chats/import`.
- Import duplicates chat, message, and memory IDs by default to avoid overwriting local sessions.
- Added `preserve_ids=true` for deliberate replay into an environment where the original chat ID does not already exist.
- Preserved summaries, memories, default file/knowledge context, answer defaults, messages, retrieval traces, sources, grounding, tool results, and skill IDs.
- Added core and API tests for duplicate imports and ID collision handling.
- Documented the new endpoints in `README.md`.

## 2026-06-08 - Backend status/config endpoint

- Added `GET /api/status` for a safe operational snapshot of the backend.
- Reports configured LLM and embedding providers, model names, base URLs, and whether provider credentials are configured.
- Reports retrieval defaults including `top_k`, relevance threshold, chunk size/overlap, full-context budget, and chat memory/context limits.
- Reports storage/index stats including files, chunks, chats, messages, knowledge bases, skills, and total indexed text sizes.
- Avoids returning secret values such as API keys.
- Added core and API tests for status counts and response shape.
- Documented the endpoint in `README.md`.

## 2026-06-08 - Assistant message feedback

- Added persisted feedback metadata on chat assistant messages.
- Added `PUT /api/chats/{chat_id}/messages/{message_id}/feedback` for rating, tags, and comments.
- Added `DELETE /api/chats/{chat_id}/messages/{message_id}/feedback` to clear feedback.
- Normalizes and deduplicates feedback tags.
- Restricts feedback to assistant messages so user turns remain plain context.
- Feedback is included in chat history and chat export/import because it lives on the message record.
- Added core and API tests for feedback persistence, validation, tag cleanup, and deletion.
- Documented the endpoints in `README.md`.

## 2026-06-08 - Feedback review endpoint

- Added `GET /api/feedback` to list feedback across chats.
- Supports `rating`, `tag`, and `limit` filters.
- Returns the prior user question, assistant answer, chat ID/title, message ID, retrieval query, diagnostics, grounding, source count, and feedback metadata.
- Makes the human feedback loop useful for RAG evaluation and answer-quality tuning.
- Added core and API tests for filtering, context fields, limits, and validation.
- Documented the endpoint in `README.md`.

## 2026-06-08 - Skill-bound tool execution

- Added `tool_ids` to skills so a matched skill can force specific built-in tools to run.
- Tool execution now merges automatic message-triggered tools with skill-bound tools.
- Explicit request `tool_ids` still work, and skill-bound tools are added on top when skills are enabled.
- Skill create/update API calls reject unknown tool IDs.
- Persisted assistant messages keep the resulting tool outputs and matched skill IDs for replay and audit.
- Added core and API tests for skill-triggered tool execution and invalid tool validation.
- Documented tool-bound skills in `README.md`.

## 2026-06-08 - Multi-query retrieval fusion

- Added `query_hits` metadata to returned sources.
- Fused duplicate chunk candidates found by multiple planned retrieval queries.
- Added a small transparent score boost for chunks that appear across several query variants.
- Included `query_hits` in source-pack context metadata for retrieval debugging.
- Added unit coverage proving repeated planned-query hits outrank a single near-tie candidate.
- Documented query fusion behavior in `README.md`.

## 2026-06-08 - Persisted prompt snapshots

- Added prompt snapshot fields to assistant messages: `prompt_messages`, `prompt_chars`, `system_prompt`, and `rag_template`.
- Saved the exact assembled provider prompt when an answer is persisted.
- Added `GET /api/chats/{chat_id}/messages/{message_id}/prompt` to replay the saved prompt for an assistant answer.
- Kept old messages backward-compatible with empty prompt snapshot defaults.
- Added core and API tests for prompt snapshot persistence and endpoint validation.
- Documented the prompt replay endpoint in `README.md`.

## 2026-06-08 - Filename-aware retrieval routing

- Added automatic narrowing when a query mentions an allowed filename or filename stem.
- Applies routing inside chat defaults, explicit `file_ids`, and knowledge-base resolved file context.
- Emits a retrieval diagnostics warning describing the matched filename route.
- Keeps the public request/response shape stable while improving multi-file QA precision.
- Fixed SQLite FTS search so file scope is applied before ranking and limiting, preventing busy indexes from dropping allowed-file matches.
- Added core and API tests proving filename mentions route retrieval to the intended document.
- Documented filename-aware routing in `README.md`.

## 2026-06-08 - Persisted file summaries

- Added local extractive summaries and keywords to `FileRecord`.
- Summaries are generated on upload and refreshed during file reindex.
- Added `GET /api/files/{file_id}/summary` for lightweight document overview.
- Kept generation deterministic and dependency-light; no LLM call is required.
- Added core tests for summary persistence/reindex refresh and API tests for the summary endpoint.
- Documented file summaries in `README.md`.

## 2026-06-08 - Context preview file overviews

- Added `FileOverview` to expose filename, summary, keywords, text size, and chunk count.
- Added resolved file overviews to `POST /api/chats/{chat_id}/context/preview`.
- Lets clients inspect available document context before generation without fetching full text.
- Reuses persisted file summaries, so preview remains fast and local.
- Added core and API tests for context preview file overview data.
- Documented the richer context preview response in `README.md`.

## 2026-06-08 - Combined answer trace endpoint

- Added `ChatMessageTraceResponse` for one-call answer replay.
- Added `GET /api/chats/{chat_id}/messages/{message_id}/trace`.
- Trace includes prior user question, assistant answer, source pack, audit, prompt snapshot, tool results, active skills, and feedback.
- Reuses persisted traces and deterministic audit logic so it does not call an LLM.
- Added core and API coverage for trace payloads and user-message validation.
- Documented the trace endpoint in `README.md`.

## 2026-06-08 - File-level search endpoint

- Added `FileSearchResponse` and ranked file search results.
- Added `GET /api/files/search?q=...` with optional `limit` and `knowledge_id`.
- Ranks files using filename, persisted keywords, and persisted summary overlap.
- Supports knowledge-base scoping for choosing context from a specific collection.
- Added core and API tests for ranking, limits, and validation.
- Documented file-level search in `README.md`.

## 2026-06-08 - Chat context suggestions

- Added `ChatContextSuggestRequest` and `ChatContextSuggestResponse`.
- Added `POST /api/chats/{chat_id}/context/suggest` for non-mutating file suggestions.
- Suggestions rank files inside the chat's saved file/knowledge context plus optional request context.
- Response includes candidate IDs, suggested IDs, file search reasons, and file overviews.
- Added core and API tests proving suggestions do not mutate chat defaults.
- Documented context suggestions in `README.md`.

## 2026-06-08 - Apply suggested chat context

- Added `ChatContextApplySuggestionsRequest` and `ChatContextApplySuggestionsResponse`.
- Added `POST /api/chats/{chat_id}/context/apply-suggestions` to persist suggested files onto chat context.
- Merge mode appends new suggested files without duplicates; `replace=true` swaps the saved file context to the suggestion set.
- Kept knowledge-base defaults unchanged so file suggestions remain a focused mutation.
- Added core and API tests proving saved suggestions are used by later answers.
- Documented the apply-suggestions workflow in `README.md`.

## 2026-06-08 - Automatic file routing for retrieval

- Added `file_selection_mode` and `file_selection_limit` to retrieval and chat answer request models.
- Added matching chat answer defaults so file routing can be saved per chat.
- Added diagnostics fields for `file_selection_mode`, `candidate_file_ids`, and `routed_file_ids`.
- Implemented auto routing that ranks allowed files by filename, summary, and keywords before chunk retrieval.
- Kept `file_selection_mode="all"` as the default to preserve existing retrieval behavior.
- Added core and API tests for routed retrieval and saved chat defaults.
- Documented auto file routing in `README.md`.

## 2026-06-08 - Persisted answer quality verdicts

- Added `AnswerQuality` to assistant messages.
- Persisted a compact quality verdict whenever an answer is saved, derived from the deterministic source audit.
- Quality includes status, support score, supported/weak/unsupported sentence counts, and audit warnings.
- Kept the full sentence-level audit available through the existing audit and trace endpoints.
- Added core and API assertions proving quality is returned immediately and retained in chat history.
- Documented answer quality metadata in `README.md`.
