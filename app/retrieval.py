from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import (
    AnswerAuditResponse,
    AnswerAuditSentence,
    AnswerGrounding,
    ChatMessage,
    RetrievalDiagnostics,
    Source,
    SourcePackFile,
    SourcePackItem,
    SourcePackResponse,
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class RetrievalPlan:
    original_query: str
    queries: list[str]


def plan_queries(history: list[ChatMessage], message: str, max_queries: int = 3) -> RetrievalPlan:
    recent_user_turns = [item.content for item in history if item.role == "user"]
    system_hint = compact_system_hint([item.content for item in history if item.role == "system"])
    queries = [message]

    if is_follow_up(message):
        contextual_queries = []
        if system_hint:
            contextual_queries.append(f"{system_hint}\n{message}")
        if recent_user_turns:
            previous = recent_user_turns[-1]
            contextual_queries.append(f"{previous}\n{message}")
        queries = contextual_queries + queries

    if recent_user_turns:
        compact = "\n".join(recent_user_turns[-2:] + [message])
        if compact not in queries:
            queries.append(compact)
    if system_hint:
        compact = "\n".join([system_hint, message])
        if compact not in queries:
            queries.append(compact)

    deduped = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return RetrievalPlan(original_query=message, queries=deduped[:max_queries])


def is_follow_up(message: str) -> bool:
    text = message.strip().lower()
    if len(text.split()) <= 5:
        return True
    return text.startswith(("and ", "what about", "how about", "then ", "also ", "compare", "same for"))


def compact_system_hint(system_messages: list[str], limit: int = 500) -> str:
    raw = "\n\n".join(message.strip() for message in system_messages if message.strip())
    if "Earlier conversation summary:" in raw:
        memory_part, summary_part = raw.split("Earlier conversation summary:", 1)
        text = f"Earlier conversation summary: {summary_part} {memory_part}"
    else:
        text = raw
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[-limit:].split(" ", 1)[-1]


def rerank_and_compress_sources(
    query: str,
    candidates: list[Source],
    top_k: int,
    max_context_chars: int,
    max_sources_per_file: int | None = None,
    diversity: str = "relevance",
    mmr_lambda: float = 0.7,
) -> list[Source]:
    candidates = fuse_candidate_sources(candidates)
    query_terms = set(TOKEN_RE.findall(query.lower()))
    reranked = []
    for source in candidates:
        source_terms = set(TOKEN_RE.findall(source.text.lower()))
        overlap = len(query_terms & source_terms) / max(1, len(query_terms))
        score = float(source.score) + (0.15 * overlap)
        reranked.append((score, source))

    reranked.sort(key=lambda item: item[0], reverse=True)
    if diversity == "mmr":
        reranked = mmr_order(reranked, mmr_lambda=mmr_lambda)
    selected: list[Source] = []
    used = 0
    seen = set()
    selected_by_file: dict[str, int] = {}
    for score, source in reranked:
        if source.chunk_id in seen:
            continue
        if max_sources_per_file is not None and selected_by_file.get(source.file_id, 0) >= max_sources_per_file:
            continue
        remaining = max_context_chars - used
        if remaining <= 0 or len(selected) >= top_k:
            break
        text = source.text.strip()
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0].strip()
        if not text:
            continue
        selected.append(source.model_copy(update={"score": round(score, 5), "text": text}))
        seen.add(source.chunk_id)
        selected_by_file[source.file_id] = selected_by_file.get(source.file_id, 0) + 1
        used += len(text)
    return selected


def fuse_candidate_sources(candidates: list[Source]) -> list[Source]:
    fused: dict[str, Source] = {}
    hit_counts: dict[str, int] = {}
    for source in candidates:
        existing = fused.get(source.chunk_id)
        hit_counts[source.chunk_id] = hit_counts.get(source.chunk_id, 0) + max(1, source.query_hits)
        if existing is None or source.score > existing.score:
            fused[source.chunk_id] = source

    merged = []
    for chunk_id, source in fused.items():
        query_hits = hit_counts[chunk_id]
        fusion_boost = min(0.2, 0.04 * max(0, query_hits - 1))
        merged.append(
            source.model_copy(
                update={
                    "score": round(float(source.score) + fusion_boost, 5),
                    "query_hits": query_hits,
                }
            )
        )
    return merged


def build_diagnostics(
    original_query: str,
    planned_queries: list[str],
    context_mode: str,
    retrieval_mode: str,
    candidate_count: int,
    sources: list[Source],
    effective_context_mode: str | None = None,
    file_selection_mode: str = "all",
    candidate_file_ids: list[str] | None = None,
    routed_file_ids: list[str] | None = None,
    source_window: int = 0,
    diversity: str = "relevance",
    mmr_lambda: float = 0.7,
    warnings: list[str] | None = None,
) -> RetrievalDiagnostics:
    confidence = assess_answerability(original_query, sources)
    combined_warnings = list(dict.fromkeys([*confidence["warnings"], *(warnings or [])]))
    return RetrievalDiagnostics(
        original_query=original_query,
        planned_queries=planned_queries,
        context_mode=context_mode,
        effective_context_mode=effective_context_mode or context_mode,
        retrieval_mode=retrieval_mode,
        candidate_count=candidate_count,
        selected_count=len(sources),
        total_context_chars=sum(len(source.text) for source in sources),
        file_selection_mode=file_selection_mode,
        candidate_file_ids=candidate_file_ids or [],
        routed_file_ids=routed_file_ids or [],
        source_window=source_window,
        diversity=diversity,
        mmr_lambda=mmr_lambda,
        top_source_score=confidence["top_source_score"],
        average_source_score=confidence["average_source_score"],
        query_term_coverage=confidence["query_term_coverage"],
        answerability=confidence["answerability"],
        warnings=combined_warnings,
    )


def build_source_pack(query: str, sources: list[Source], diagnostics: RetrievalDiagnostics) -> SourcePackResponse:
    query_terms = meaningful_terms(query)
    items = []
    files: dict[str, dict] = {}
    context_parts = []
    for index, source in enumerate(sources, start=1):
        marker = f"[{index}]"
        source_terms = meaningful_terms(source.text)
        matched_terms = sorted(query_terms & source_terms)
        item = SourcePackItem(
            marker=marker,
            source=source,
            excerpt=citation_excerpt(source.text, query_terms),
            matched_terms=matched_terms,
        )
        items.append(item)
        group = files.setdefault(
            source.file_id,
            {
                "file_id": source.file_id,
                "filename": source.filename,
                "source_count": 0,
                "top_score": 0.0,
                "markers": [],
                "sources": [],
            },
        )
        group["source_count"] += 1
        group["top_score"] = max(float(group["top_score"]), source.score)
        group["markers"].append(marker)
        group["sources"].append(item)
        context_parts.append(
            "\n".join(
                [
                    f"{marker} {source.filename} chunk {source.chunk_index}",
                    (
                        f"score={source.score} range={source.context_start_index}-{source.context_end_index} "
                        f"chars={source.start_char}-{source.end_char} query_hits={source.query_hits} chunk_id={source.chunk_id}"
                    ),
                    source.text.strip(),
                ]
            )
        )

    grouped_files = [
        SourcePackFile(
            file_id=group["file_id"],
            filename=group["filename"],
            source_count=group["source_count"],
            top_score=round(float(group["top_score"]), 5),
            markers=group["markers"],
            sources=group["sources"],
        )
        for group in files.values()
    ]
    grouped_files.sort(key=lambda item: item.top_score, reverse=True)
    return SourcePackResponse(
        query=query,
        diagnostics=diagnostics,
        sources=sources,
        files=grouped_files,
        context_text="\n\n".join(context_parts),
    )


def audit_answer(
    answer: str,
    sources: list[Source],
    grounding: AnswerGrounding,
    message_id: str = "",
) -> AnswerAuditResponse:
    source_terms = [meaningful_terms(source.text) for source in sources]
    sentences = []
    for index, sentence in enumerate(answer_sentences(answer), start=1):
        cited_indexes = _cited_source_indexes(sentence, len(sources))
        terms = meaningful_terms(_strip_citations(sentence))
        candidate_indexes = cited_indexes or list(range(1, len(sources) + 1))
        matched = []
        best_score = 0.0
        for source_index in candidate_indexes:
            terms_for_source = source_terms[source_index - 1] if 1 <= source_index <= len(source_terms) else set()
            overlap = len(terms & terms_for_source) / max(1, len(terms))
            if overlap > 0:
                matched.append(source_index)
            best_score = max(best_score, overlap)
        if not terms:
            status = "supported"
        elif best_score >= 0.55:
            status = "supported"
        elif best_score >= 0.25:
            status = "weak"
        else:
            status = "unsupported"
        sentences.append(
            AnswerAuditSentence(
                index=index,
                text=sentence,
                cited_markers=[f"[{source_index}]" for source_index in cited_indexes],
                matched_source_indexes=matched,
                support_score=round(best_score, 5),
                status=status,
            )
        )

    supported_count = sum(1 for item in sentences if item.status == "supported")
    weak_count = sum(1 for item in sentences if item.status == "weak")
    unsupported_count = sum(1 for item in sentences if item.status == "unsupported")
    average_score = sum(item.support_score for item in sentences) / max(1, len(sentences))
    warnings = list(grounding.warnings)
    if not sources:
        warnings.append("No persisted sources are available to audit this answer.")
    if unsupported_count:
        warnings.append("Some answer sentences were not supported by the persisted sources.")
    if weak_count:
        warnings.append("Some answer sentences have weak source overlap.")
    answer_supported = bool(sentences) and unsupported_count == 0 and grounding.missing_citation_count == 0
    return AnswerAuditResponse(
        message_id=message_id,
        answer_supported=answer_supported,
        support_score=round(average_score, 5),
        sentence_count=len(sentences),
        supported_count=supported_count,
        weak_count=weak_count,
        unsupported_count=unsupported_count,
        grounding=grounding,
        sentences=sentences,
        warnings=list(dict.fromkeys(warnings)),
    )


def answer_sentences(answer: str) -> list[str]:
    cleaned = []
    for line in answer.splitlines():
        text = line.strip()
        if not text or text.lower() in {"answer:", "sources:"} or text.lower().startswith("sources:"):
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        cleaned.append(text)
    joined = " ".join(cleaned)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", joined) if part.strip()]


def _strip_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", " ", text)


def _cited_source_indexes(text: str, source_count: int) -> list[int]:
    indexes = []
    for raw_index in re.findall(r"\[(\d+)\]", text):
        index = int(raw_index)
        if 1 <= index <= source_count and index not in indexes:
            indexes.append(index)
    return indexes


def citation_excerpt(text: str, query_terms: set[str], limit: int = 320) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if part.strip()]
    if not sentences:
        excerpt = text.strip().replace("\n", " ")
    else:
        ranked = []
        for position, sentence in enumerate(sentences):
            terms = meaningful_terms(sentence)
            ranked.append((len(query_terms & terms), -position, sentence))
        ranked.sort(reverse=True)
        excerpt = ranked[0][2]
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0].strip() + "..."
    return excerpt


def mmr_order(scored_sources: list[tuple[float, Source]], mmr_lambda: float = 0.7) -> list[tuple[float, Source]]:
    remaining = list(scored_sources)
    selected: list[tuple[float, Source]] = []
    while remaining:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best_position = 0
        best_score = float("-inf")
        for position, (score, source) in enumerate(remaining):
            redundancy = max(source_similarity(source, selected_source) for _, selected_source in selected)
            file_penalty = 0.15 if any(source.file_id == selected_source.file_id for _, selected_source in selected) else 0.0
            mmr_score = (mmr_lambda * score) - ((1.0 - mmr_lambda) * redundancy) - file_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_position = position
        selected.append(remaining.pop(best_position))
    return selected


def source_similarity(left: Source, right: Source) -> float:
    left_terms = set(TOKEN_RE.findall(left.text.lower()))
    right_terms = set(TOKEN_RE.findall(right.text.lower()))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def assess_answerability(query: str, sources: list[Source]) -> dict:
    if not sources:
        return {
            "top_source_score": 0.0,
            "average_source_score": 0.0,
            "query_term_coverage": 0.0,
            "answerability": "none",
            "warnings": ["No source context was selected."],
        }

    query_terms = meaningful_terms(query)
    source_terms = set()
    for source in sources:
        source_terms.update(meaningful_terms(source.text))
    coverage = len(query_terms & source_terms) / max(1, len(query_terms))
    scores = [max(0.0, float(source.score)) for source in sources]
    top_score = max(scores) if scores else 0.0
    average_score = sum(scores) / max(1, len(scores))

    if coverage >= 0.75 and top_score >= 0.45:
        answerability = "high"
    elif coverage >= 0.45 and top_score >= 0.20:
        answerability = "medium"
    elif coverage > 0.0 or top_score > 0.0:
        answerability = "low"
    else:
        answerability = "none"

    warnings = []
    if answerability in {"none", "low"}:
        warnings.append("Retrieved evidence may be insufficient to answer confidently.")
    if coverage < 0.45:
        warnings.append("Selected sources cover only a small portion of the query terms.")

    return {
        "top_source_score": round(top_score, 5),
        "average_source_score": round(average_score, 5),
        "query_term_coverage": round(coverage, 5),
        "answerability": answerability,
        "warnings": warnings,
    }


def meaningful_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
    return {term for term in TOKEN_RE.findall(text.lower()) if len(term) > 1 and term not in stopwords}
