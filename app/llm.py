from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import json
import re

import httpx

from app.config import Settings
from app.schemas import AnswerGrounding, ChatMessage, Source, SourceCitation, ToolResult


class AnswerGenerator(ABC):
    @abstractmethod
    async def answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> str:
        raise NotImplementedError

    async def stream_answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> AsyncIterator[str]:
        yield await self.answer(
            messages,
            question,
            sources,
            tool_results,
            system_prompt=system_prompt,
            rag_template=rag_template,
        )


class ExtractiveAnswerGenerator(AnswerGenerator):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings

    async def answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> str:
        if tool_results and not sources:
            parts = ["Tool results:", ""]
            parts.extend(f"- {result.name}: {result.output}" for result in tool_results)
            return "\n".join(parts)

        if sources:
            question_terms = set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
            parts = ["Answer:", ""]
            for idx, source in enumerate(sources[:4], start=1):
                excerpt = best_excerpt(source.text, question_terms)
                parts.append(f"[{idx}] {excerpt}")
            parts.append("")
            parts.append("Sources: " + ", ".join(f"[{idx}] {source.filename}" for idx, source in enumerate(sources[:4], start=1)))
            return "\n".join(parts)

        history_hint = ""
        memory_hint = ""
        for message in reversed(messages):
            if message.role == "system" and "Saved chat memory:" in message.content:
                memory_text = extract_saved_memory_text(message.content)
                memory_hint = best_excerpt(
                    memory_text,
                    set(re.findall(r"[a-zA-Z0-9_]+", question.lower())),
                    max_sentences=1,
                )
                break
        if memory_hint:
            return f"From saved chat memory: {memory_hint}"
        for message in reversed(messages):
            if message.role == "assistant":
                history_hint = message.content[:500]
                break
        if history_hint:
            return f"I do not have matching file context for this question. From the chat context, the last useful point was: {history_hint}"
        return "I do not have enough retrieved file context to answer that confidently."


class OpenAIAnswerGenerator(AnswerGenerator):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI chat generation")
        self.settings = settings

    async def answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> str:
        payload_messages = build_prompt_messages(
            messages,
            question,
            sources,
            tool_results,
            system_prompt=system_prompt or self.settings.system_prompt,
            rag_template=rag_template or self.settings.rag_template,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={"model": self.settings.openai_chat_model, "messages": payload_messages},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def stream_answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> AsyncIterator[str]:
        payload_messages = build_prompt_messages(
            messages,
            question,
            sources,
            tool_results,
            system_prompt=system_prompt or self.settings.system_prompt,
            rag_template=rag_template or self.settings.rag_template,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={"model": self.settings.openai_chat_model, "messages": payload_messages, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    token = event.get("choices", [{}])[0].get("delta", {}).get("content")
                    if token:
                        yield token


class OllamaAnswerGenerator(AnswerGenerator):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> str:
        payload_messages = build_prompt_messages(
            messages,
            question,
            sources,
            tool_results,
            system_prompt=system_prompt or self.settings.system_prompt,
            rag_template=rag_template or self.settings.rag_template,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={"model": self.settings.ollama_chat_model, "messages": payload_messages, "stream": False},
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    async def stream_answer(
        self,
        messages: list[ChatMessage],
        question: str,
        sources: list[Source],
        tool_results: list[ToolResult],
        system_prompt: str | None = None,
        rag_template: str | None = None,
    ) -> AsyncIterator[str]:
        payload_messages = build_prompt_messages(
            messages,
            question,
            sources,
            tool_results,
            system_prompt=system_prompt or self.settings.system_prompt,
            rag_template=rag_template or self.settings.rag_template,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={"model": self.settings.ollama_chat_model, "messages": payload_messages, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    token = event.get("message", {}).get("content")
                    if token:
                        yield token
                    if event.get("done"):
                        break


def build_prompt_messages(
    messages: list[ChatMessage],
    question: str,
    sources: list[Source],
    tool_results: list[ToolResult],
    system_prompt: str,
    rag_template: str,
) -> list[dict[str, str]]:
    source_context = "\n\n".join(
        (
            f"<source id='{idx}' file='{source.filename}' file_id='{source.file_id}' "
            f"chunk_id='{source.chunk_id}' chunk_index='{source.chunk_index}' "
            f"context_range='{source.context_start_index}-{source.context_end_index}' "
            f"char_range='{source.start_char}-{source.end_char}' score='{source.score}'>\n"
            f"{source.text}\n"
            "</source>"
        )
        for idx, source in enumerate(sources, start=1)
    )
    system = system_prompt
    if source_context:
        system += "\n\n" + render_rag_template(rag_template, source_context, question)
    if tool_results:
        tool_context = "\n".join(f"<tool name='{result.name}'>\n{result.output}\n</tool>" for result in tool_results)
        system += f"\n\nTool results:\n{tool_context}"

    prompt_messages = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "system":
            prompt_messages[0]["content"] += f"\n\n{message.content}"
        elif message.role in {"user", "assistant"}:
            prompt_messages.append({"role": message.role, "content": message.content})
    prompt_messages.append({"role": "user", "content": question})
    return prompt_messages


def best_excerpt(text: str, question_terms: set[str], limit: int = 650, max_sentences: int = 2) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if part.strip()]
    if not sentences:
        excerpt = text.strip().replace("\n", " ")
    else:
        ranked = []
        for sentence in sentences:
            terms = set(re.findall(r"[a-zA-Z0-9_]+", sentence.lower()))
            ranked.append((len(question_terms & terms), sentence))
        ranked.sort(key=lambda item: item[0], reverse=True)
        excerpt = " ".join(sentence for _, sentence in ranked[:max_sentences])
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0] + "..."
    return excerpt


def render_rag_template(template: str, context: str, question: str) -> str:
    return template.replace("{context}", context).replace("{question}", question)


def extract_saved_memory_text(system_content: str) -> str:
    if "Saved chat memory:" not in system_content:
        return system_content
    memory_part = system_content.split("Saved chat memory:", 1)[1]
    memory_part = memory_part.split("Earlier conversation summary:", 1)[0]
    lines = [line.removeprefix("- ").strip() for line in memory_part.splitlines()]
    return ". ".join(line.rstrip(".") for line in lines if line) + "."


def analyze_grounding(answer: str, sources: list[Source]) -> AnswerGrounding:
    if not sources:
        return AnswerGrounding()

    cited_indexes = []
    missing_count = 0
    for raw_index in re.findall(r"\[(\d+)\]", answer):
        index = int(raw_index)
        if 1 <= index <= len(sources):
            if index not in cited_indexes:
                cited_indexes.append(index)
        else:
            missing_count += 1

    citations = [
        SourceCitation(
            marker=f"[{index}]",
            source_index=index,
            file_id=sources[index - 1].file_id,
            filename=sources[index - 1].filename,
            chunk_id=sources[index - 1].chunk_id,
            chunk_index=sources[index - 1].chunk_index,
            context_start_index=sources[index - 1].context_start_index,
            context_end_index=sources[index - 1].context_end_index,
            start_char=sources[index - 1].start_char,
            end_char=sources[index - 1].end_char,
        )
        for index in cited_indexes
    ]
    warnings = []
    if not citations:
        warnings.append("Answer used retrieved context but did not include source citations.")
    if missing_count:
        warnings.append("Answer cited source numbers that were not present in the retrieved context.")

    return AnswerGrounding(
        has_sources=True,
        cited_source_count=len(citations),
        uncited_source_count=max(len(sources) - len(citations), 0),
        missing_citation_count=missing_count,
        citations=citations,
        warnings=warnings,
    )


def build_answer_generator(settings: Settings) -> AnswerGenerator:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAIAnswerGenerator(settings)
    if provider == "ollama":
        return OllamaAnswerGenerator(settings)
    return ExtractiveAnswerGenerator(settings)
