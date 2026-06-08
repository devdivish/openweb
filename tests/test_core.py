from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.embeddings import LocalHashEmbedder
from app.llm import ExtractiveAnswerGenerator, analyze_grounding, build_prompt_messages
from app.retrieval import (
    audit_answer,
    assess_answerability,
    build_diagnostics,
    build_source_pack,
    fuse_candidate_sources,
    plan_queries,
    rerank_and_compress_sources,
)
from app.schemas import ChatMessage, Source
from app.rag import RagService
from app.store import JsonStore, SQLiteStore
from app.text import chunk_text, chunk_text_with_spans
from app.tools import ToolRegistry
from app.vector_store import VectorStore


class CoreBackendTest(unittest.TestCase):
    def test_chunk_text_respects_overlap(self) -> None:
        text = " ".join(str(i) for i in range(300))
        chunks = chunk_text(text, chunk_size=120, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))

    def test_chunk_text_with_spans_preserves_document_offsets(self) -> None:
        text = "Opening paragraph has alpha.\n\nMiddle paragraph has beta marker.\n\nClosing paragraph has gamma."
        chunks = chunk_text_with_spans(text, chunk_size=45, overlap=10)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(text[chunk.start_char : chunk.end_char], chunk.text)
            self.assertGreater(chunk.end_char, chunk.start_char)
        beta = next(chunk for chunk in chunks if "beta marker" in chunk.text)
        self.assertEqual(text[beta.start_char : beta.end_char], "Middle paragraph has beta marker.")

    def test_mmr_diversity_prefers_less_redundant_sources(self) -> None:
        candidates = [
            Source(
                file_id="a",
                filename="a.txt",
                chunk_id="a:0",
                score=0.95,
                text="refund policy receipt receipt receipt alpha",
            ),
            Source(
                file_id="a",
                filename="a.txt",
                chunk_id="a:1",
                score=0.94,
                text="refund policy receipt receipt receipt beta",
            ),
            Source(
                file_id="b",
                filename="b.txt",
                chunk_id="b:0",
                score=0.8,
                text="refund escalation manager exception gamma",
            ),
        ]
        relevance = rerank_and_compress_sources("refund policy", candidates, top_k=2, max_context_chars=1000)
        diverse = rerank_and_compress_sources(
            "refund policy",
            candidates,
            top_k=2,
            max_context_chars=1000,
            diversity="mmr",
            mmr_lambda=0.35,
        )
        self.assertEqual([source.chunk_id for source in relevance], ["a:0", "a:1"])
        self.assertEqual([source.chunk_id for source in diverse], ["a:0", "b:0"])

    def test_query_fusion_boosts_chunks_found_by_multiple_planned_queries(self) -> None:
        candidates = [
            Source(
                file_id="a",
                filename="a.txt",
                chunk_id="a:0",
                score=0.5,
                text="refund policy exception escalation",
            ),
            Source(
                file_id="a",
                filename="a.txt",
                chunk_id="a:0",
                score=0.48,
                text="refund policy exception escalation",
            ),
            Source(
                file_id="b",
                filename="b.txt",
                chunk_id="b:0",
                score=0.52,
                text="refund policy generic note",
            ),
        ]

        fused = fuse_candidate_sources(candidates)
        fused_by_id = {source.chunk_id: source for source in fused}
        self.assertEqual(fused_by_id["a:0"].query_hits, 2)
        self.assertGreater(fused_by_id["a:0"].score, 0.5)

        selected = rerank_and_compress_sources("refund policy", candidates, top_k=1, max_context_chars=1000)
        self.assertEqual(selected[0].chunk_id, "a:0")
        self.assertEqual(selected[0].query_hits, 2)

        diagnostics = build_diagnostics("refund policy", ["refund", "policy"], "rag", "hybrid", len(candidates), selected)
        source_pack = build_source_pack("refund policy", selected, diagnostics)
        self.assertIn("query_hits=2", source_pack.context_text)

    def test_answerability_scores_selected_sources(self) -> None:
        high = assess_answerability(
            "refund window",
            [
                Source(
                    file_id="f1",
                    filename="policy.txt",
                    chunk_id="f1:0",
                    score=0.8,
                    text="The refund window is 30 days.",
                )
            ],
        )
        none = assess_answerability("refund window", [])

        self.assertEqual(high["answerability"], "high")
        self.assertEqual(high["query_term_coverage"], 1.0)
        self.assertEqual(none["answerability"], "none")
        self.assertTrue(none["warnings"])

    def test_source_pack_groups_sources_and_builds_context_text(self) -> None:
        sources = [
            Source(
                file_id="f1",
                filename="policy.txt",
                chunk_id="f1:0",
                chunk_index=0,
                start_char=10,
                end_char=90,
                score=0.8,
                text="Refund policy says customers have a 30 day refund window. Shipping takes two days.",
            ),
            Source(
                file_id="f1",
                filename="policy.txt",
                chunk_id="f1:1",
                chunk_index=1,
                start_char=91,
                end_char=140,
                score=0.5,
                text="Refund exceptions require manager approval.",
            ),
            Source(
                file_id="f2",
                filename="returns.txt",
                chunk_id="f2:0",
                chunk_index=0,
                start_char=0,
                end_char=58,
                score=0.6,
                text="Return labels are generated from the account portal.",
            ),
        ]
        diagnostics = build_diagnostics("refund window", ["refund window"], "rag", "hybrid", 3, sources)
        pack = build_source_pack("refund window", sources, diagnostics)

        self.assertEqual(pack.context_text.count("[1]"), 1)
        self.assertIn("[2] policy.txt chunk 1", pack.context_text)
        self.assertIn("chars=10-90", pack.context_text)
        self.assertIn("chunk_id=f1:0", pack.context_text)
        self.assertEqual(len(pack.files), 2)
        self.assertEqual(pack.files[0].filename, "policy.txt")
        self.assertEqual(pack.files[0].markers, ["[1]", "[2]"])
        self.assertEqual(pack.files[0].source_count, 2)
        self.assertIn("refund", pack.files[0].sources[0].matched_terms)
        self.assertIn("30 day refund window", pack.files[0].sources[0].excerpt)

    def test_retrieval_explain_returns_ranked_candidates(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=90, chunk_overlap=0)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "explain.txt"
                doc.write_text(
                    "\n\n".join(
                        [
                            "Refund policy marker A says receipts are required.",
                            "Refund policy marker B says manager approval is required.",
                            "Refund policy marker C says support must log the case.",
                        ]
                    ),
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "explain.txt", "text/plain", doc.stat().st_size)

                explained = await service.explain_retrieval(
                    "refund policy marker",
                    file_ids=[record.id],
                    retrieval_mode="keyword",
                    top_k=1,
                    candidate_limit=5,
                )
                self.assertEqual(explained.query, "refund policy marker")
                self.assertEqual(explained.diagnostics.selected_count, 1)
                self.assertEqual(len(explained.source_pack.sources), 1)
                self.assertGreaterEqual(len(explained.candidates), 2)
                self.assertEqual(explained.candidates[0].rank, 1)
                self.assertTrue(any(candidate.selected for candidate in explained.candidates))
                self.assertTrue(any(not candidate.selected for candidate in explained.candidates))
                self.assertIn("refund", explained.candidates[0].matched_terms)
                self.assertTrue(any("selected_for_context" in candidate.reasons for candidate in explained.candidates))

        asyncio.run(run())

    def test_chat_retrieval_explain_uses_context_without_mutating_chat(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=2, chunk_size=160, chunk_overlap=0)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "chat-explain.txt"
                doc.write_text(
                    "Refund policy allows returns within 30 days.\n\n"
                    "Refund exceptions require manager approval.\n\n"
                    "Shipping exceptions require warehouse approval.",
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "chat-explain.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("chat explain", file_ids=[record.id])

                await service.ask(chat.id, "What is the refund policy?", retrieval_mode="keyword")
                await service.ask(chat.id, "Remember that my preferred team is support", use_rag=False)
                before = store.get_chat(chat.id)
                self.assertIsNotNone(before)
                assert before is not None
                before_message_count = len(before.messages)

                explained = await service.explain_chat_retrieval(
                    chat.id,
                    "And exceptions?",
                    retrieval_mode="keyword",
                    top_k=1,
                    candidate_limit=5,
                )
                self.assertEqual(explained.query, "And exceptions?")
                self.assertIn("refund policy", explained.retrieval_query.lower())
                self.assertTrue(explained.source_pack.sources)
                self.assertIn("Refund exceptions", explained.source_pack.sources[0].text)
                self.assertTrue(explained.candidates)

                after = store.get_chat(chat.id)
                self.assertIsNotNone(after)
                assert after is not None
                self.assertEqual(len(after.messages), before_message_count)

        asyncio.run(run())

    def test_follow_up_planning_uses_system_summary_hint(self) -> None:
        history = [
            ChatMessage(
                role="system",
                content=(
                    "Saved chat memory:\n- User's desk color is blue\n\n"
                    "Earlier conversation summary:\nuser asked about refund policies."
                ),
            ),
        ]
        plan = plan_queries(history, "And exceptions?")
        self.assertTrue(plan.queries[0].startswith("Earlier conversation summary"))
        self.assertIn("refund policies", plan.queries[0])
        self.assertIn("And exceptions?", plan.queries[0])

    def test_ingest_retrieve_and_context(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=160, chunk_overlap=30)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                embedder = LocalHashEmbedder(128)
                vector_store = VectorStore(store)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=embedder,
                    vector_store=vector_store,
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )

                doc = tmp_path / "policy.txt"
                doc.write_text(
                    "Refund policy: customers can request a refund within 30 days. "
                    "Shipping policy: express shipping takes two days.",
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "policy.txt", "text/plain", doc.stat().st_size)
                self.assertGreater(record.chunk_count, 0)
                stored_chunks = store.file_chunks(record.id)
                self.assertTrue(stored_chunks)
                self.assertEqual(
                    doc.read_text(encoding="utf-8")[stored_chunks[0].start_char : stored_chunks[0].end_char],
                    stored_chunks[0].text,
                )

                chat = service.create_chat("support")
                answer, sources, _, _, retrieval_query, diagnostics, grounding = await service.ask(
                    chat.id, "What is the refund window?"
                )
                self.assertTrue(sources)
                self.assertGreater(sources[0].end_char, sources[0].start_char)
                self.assertEqual(doc.read_text(encoding="utf-8")[sources[0].start_char : sources[0].end_char], sources[0].text)
                self.assertIn("refund", answer.content.lower())
                self.assertIn("refund", retrieval_query.lower())
                self.assertGreaterEqual(diagnostics.selected_count, 1)
                self.assertIn(diagnostics.answerability, {"medium", "high"})
                self.assertGreater(diagnostics.query_term_coverage, 0)
                self.assertTrue(grounding.has_sources)
                self.assertEqual(grounding.citations[0].filename, "policy.txt")
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertTrue(saved.messages[-2].id)
                self.assertTrue(saved.messages[-1].id)
                self.assertNotEqual(saved.messages[-2].id, saved.messages[-1].id)
                self.assertEqual(saved.messages[-1].retrieval_query, retrieval_query)
                self.assertIsNotNone(saved.messages[-1].diagnostics)
                assert saved.messages[-1].diagnostics is not None
                self.assertEqual(saved.messages[-1].diagnostics.answerability, diagnostics.answerability)

                answer2, _, _, _, retrieval_query2, diagnostics2, _ = await service.ask(chat.id, "And shipping?")
                self.assertIn("refund window", retrieval_query2.lower())
                self.assertIn("shipping", answer2.content.lower())
                self.assertGreaterEqual(len(diagnostics2.planned_queries), 2)

                full_sources, full_diagnostics = await service.retrieve("anything", file_ids=[record.id], context_mode="full")
                self.assertEqual(len(full_sources), 1)
                self.assertIn("Refund policy", full_sources[0].text)
                self.assertEqual(full_diagnostics.context_mode, "full")

    def test_context_summary_compacts_old_messages(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=4)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                )
                chat = service.create_chat("memory")
                for idx in range(4):
                    await service.ask(chat.id, f"Remember detail {idx}", use_rag=False)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertLessEqual(len(saved.messages), 4)
                self.assertIn("Remember detail 0", saved.summary)

        asyncio.run(run())

    def test_auto_file_selection_routes_retrieval_to_ranked_files(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                refund_doc = tmp_path / "routing-refund.txt"
                shipping_doc = tmp_path / "routing-shipping.txt"
                billing_doc = tmp_path / "routing-billing.txt"
                refund_doc.write_text("Refund exceptions require manager approval and a receipt.", encoding="utf-8")
                shipping_doc.write_text("Shipping exceptions require dispatch approval.", encoding="utf-8")
                billing_doc.write_text("Billing disputes require invoice review.", encoding="utf-8")
                refund = await service.ingest_file(refund_doc, "routing-refund.txt", "text/plain", refund_doc.stat().st_size)
                shipping = await service.ingest_file(
                    shipping_doc,
                    "routing-shipping.txt",
                    "text/plain",
                    shipping_doc.stat().st_size,
                )
                billing = await service.ingest_file(billing_doc, "routing-billing.txt", "text/plain", billing_doc.stat().st_size)

                sources, diagnostics = await service.retrieve(
                    "How do refund exceptions work?",
                    file_ids=[refund.id, shipping.id, billing.id],
                    retrieval_mode="keyword",
                    file_selection_mode="auto",
                    file_selection_limit=1,
                )

                self.assertEqual(diagnostics.file_selection_mode, "auto")
                self.assertCountEqual(diagnostics.candidate_file_ids, [refund.id, shipping.id, billing.id])
                self.assertEqual(diagnostics.routed_file_ids, [refund.id])
                self.assertTrue(sources)
                self.assertTrue(all(source.file_id == refund.id for source in sources))
                self.assertTrue(any("Auto file routing selected 1 of 3" in warning for warning in diagnostics.warnings))

        asyncio.run(run())

    def test_manual_chat_compaction_keeps_requested_tail(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=50)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                chat = service.create_chat("manual compact")
                for idx in range(3):
                    await service.ask(chat.id, f"Manual compact detail {idx}", use_rag=False)

                compacted = service.compact_chat(chat.id, keep_last=2)
                self.assertEqual(len(compacted.messages), 2)
                self.assertIn("Manual compact detail 0", compacted.summary)
                self.assertIn("Manual compact detail 2", compacted.messages[0].content)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(len(saved.messages), 2)
                self.assertEqual(saved.messages[0].role, "user")
                self.assertEqual(saved.messages[1].role, "assistant")

                emptied = service.compact_chat(chat.id, keep_last=0)
                self.assertEqual(emptied.messages, [])
                self.assertIn("Manual compact detail 2", emptied.summary)

        asyncio.run(run())

    def test_prune_chat_messages_removes_retained_branch(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=20)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                chat = service.create_chat("prune")
                for idx in range(3):
                    await service.ask(chat.id, f"Prune detail {idx}", use_rag=False)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(len(saved.messages), 6)
                second_user_id = saved.messages[2].id
                second_assistant_id = saved.messages[3].id

                pruned = service.prune_chat_messages(chat.id, second_user_id)
                self.assertEqual(len(pruned.messages), 2)
                self.assertEqual(pruned.messages[-1].role, "assistant")
                self.assertNotIn(second_user_id, [message.id for message in pruned.messages])

                await service.ask(chat.id, "Prune detail replacement", use_rag=False)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                pruned_after = service.prune_chat_messages(chat.id, saved.messages[1].id, include_selected=False)
                self.assertEqual(len(pruned_after.messages), 2)
                self.assertEqual(pruned_after.messages[-1].id, saved.messages[1].id)
                self.assertNotIn(second_assistant_id, [message.id for message in pruned_after.messages])

        asyncio.run(run())

    def test_compacted_summary_guides_follow_up_retrieval(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=2, chunk_size=180, chunk_overlap=20)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "exceptions.txt"
                doc.write_text(
                    "Refund policy allows returns within 30 days.\n\n"
                    "Refund exceptions require manager approval.\n\n"
                    "Shipping exceptions require warehouse approval.",
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "exceptions.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("compacted followup")

                await service.ask(
                    chat.id,
                    "What is the refund policy?",
                    file_ids=[record.id],
                    retrieval_mode="keyword",
                )
                await service.ask(chat.id, "Remember that my desk color is blue", use_rag=False)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertIn("refund policy", saved.summary.lower())

                answer, sources, _, _, retrieval_query, diagnostics, _ = await service.ask(
                    chat.id,
                    "And exceptions?",
                    file_ids=[record.id],
                    retrieval_mode="keyword",
                    top_k=1,
                )
                self.assertIn("refund policy", retrieval_query.lower())
                self.assertTrue(sources)
                self.assertIn("Refund exceptions", sources[0].text)
                self.assertIn("manager approval", answer.content)
                self.assertGreaterEqual(diagnostics.selected_count, 1)

        asyncio.run(run())

    def test_context_preview_exposes_memory_summary_and_resolved_files(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=2)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "preview.txt"
                doc.write_text("Refund exceptions need manager approval.", encoding="utf-8")
                record = await service.ingest_file(doc, "preview.txt", "text/plain", doc.stat().st_size)
                knowledge = store.create_knowledge("Preview KB", file_ids=[record.id])
                chat = service.create_chat("preview", knowledge_ids=[knowledge.id])

                await service.ask(chat.id, "What is the refund policy?", file_ids=[record.id])
                await service.ask(chat.id, "Remember that my preferred team is support", use_rag=False)

                preview = service.preview_context(chat.id, "And exceptions?")
                self.assertEqual(preview.chat_id, chat.id)
                self.assertIn(record.id, preview.resolved_file_ids)
                self.assertEqual(preview.files[0].id, record.id)
                self.assertIn("Refund exceptions", preview.files[0].summary)
                self.assertIn("refund", preview.files[0].keywords)
                self.assertIn("preferred team", preview.memory_context)
                self.assertTrue(preview.summary)
                self.assertIn("refund policy", preview.summary.lower())
                self.assertGreaterEqual(len(preview.rolling_messages), 2)
                self.assertTrue(preview.planned_queries)
                self.assertIn("And exceptions?", preview.retrieval_query)

        asyncio.run(run())

    def test_answer_preview_builds_prompt_without_mutating_chat(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "answer-preview.txt"
                doc.write_text("The answer preview code is AP-515.", encoding="utf-8")
                record = await service.ingest_file(doc, "answer-preview.txt", "text/plain", doc.stat().st_size)
                skill = store.create_skill("Preview Skill", "Always mention preview diagnostics.", triggers=["preview"])
                chat = service.create_chat("answer preview", file_ids=[record.id])

                preview = await service.preview_answer(
                    chat.id,
                    "Remember that my preview mode is careful. What is AP-515? Calculate 2 + 3",
                    retrieval_mode="keyword",
                    tool_ids=["calculator"],
                )

                self.assertEqual(preview.chat_id, chat.id)
                self.assertTrue(preview.source_pack.sources)
                self.assertIn("AP-515", preview.source_pack.context_text)
                self.assertTrue(preview.tool_results)
                self.assertIn("5", preview.tool_results[0].output)
                self.assertEqual([item.id for item in preview.skills], [skill.id])
                self.assertTrue(preview.prompt_messages)
                self.assertEqual(preview.prompt_messages[-1].role, "user")
                self.assertIn("Remember that my preview mode", preview.prompt_messages[-1].content)
                self.assertTrue(preview.would_learn_memories)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.messages, [])
                self.assertEqual(saved.memories, [])

        asyncio.run(run())

    def test_saved_answer_persists_prompt_snapshot(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "prompt-snapshot.txt"
                doc.write_text("The prompt snapshot marker is PROMPT-717.", encoding="utf-8")
                record = await service.ingest_file(doc, "prompt-snapshot.txt", "text/plain", doc.stat().st_size)
                store.create_skill("Prompt Skill", "Mention prompt evidence carefully.", triggers=["prompt"])
                chat = service.create_chat("prompt snapshot", file_ids=[record.id])

                answer, _, _, _, _, _, _ = await service.ask(
                    chat.id,
                    "What is the prompt snapshot marker? Calculate 6 + 4",
                    retrieval_mode="keyword",
                    tool_ids=["calculator"],
                )
                snapshot = service.get_message_prompt(chat.id, answer.id)
                trace = service.get_message_trace(chat.id, answer.id)

                self.assertEqual(snapshot.message_id, answer.id)
                self.assertEqual(snapshot.prompt_chars, answer.prompt_chars)
                self.assertEqual(snapshot.prompt_messages, answer.prompt_messages)
                self.assertGreater(snapshot.prompt_chars, 0)
                self.assertIn("PROMPT-717", "\n".join(item.content for item in snapshot.prompt_messages))
                self.assertIn("Calculator", "\n".join(item.content for item in snapshot.prompt_messages))
                self.assertIn("Prompt Skill", snapshot.system_prompt)
                self.assertIn("prompt snapshot marker", snapshot.prompt_messages[-1].content)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.messages[-1].prompt_messages, snapshot.prompt_messages)
                self.assertEqual(saved.messages[-1].answer_quality.status, "supported")
                self.assertEqual(trace.message_id, answer.id)
                self.assertIn("prompt snapshot marker", trace.question)
                self.assertIsNotNone(trace.source_pack)
                assert trace.source_pack is not None
                self.assertIn("PROMPT-717", trace.source_pack.context_text)
                self.assertIsNotNone(trace.audit)
                assert trace.audit is not None
                self.assertEqual(trace.audit.message_id, answer.id)
                self.assertEqual(answer.answer_quality.status, "supported")
                self.assertTrue(answer.answer_quality.answer_supported)
                self.assertEqual(answer.answer_quality.unsupported_count, 0)
                self.assertGreater(answer.answer_quality.support_score, 0)
                self.assertIsNotNone(trace.prompt)
                assert trace.prompt is not None
                self.assertEqual(trace.prompt.prompt_messages, snapshot.prompt_messages)
                self.assertEqual([skill.name for skill in trace.skills], ["Prompt Skill"])
                self.assertEqual(trace.tool_results[0].tool_id, "calculator")

        asyncio.run(run())

    def test_chat_memory_survives_compaction_and_answers_followups(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chat_context_messages=2, chat_memory_items=10)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                )
                chat = service.create_chat("memory")
                await service.ask(chat.id, "Remember that my project codename is falcon", use_rag=False)
                for idx in range(3):
                    await service.ask(chat.id, f"Temporary turn {idx}", use_rag=False)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertLessEqual(len(saved.messages), 2)
                self.assertTrue(saved.memories)
                self.assertEqual(len(saved.memories), 1)
                self.assertIn("falcon", saved.memories[0].content.lower())

                answer, _, _, _, _, _, _ = await service.ask(chat.id, "What is my project codename?", use_rag=False)
                self.assertIn("falcon", answer.content.lower())

        asyncio.run(run())

    def test_manual_chat_memory_deduplicates_existing_fact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(data_dir=tmp_path / "data")
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            store = JsonStore(settings.state_path)
            service = RagService(
                settings=settings,
                store=store,
                embedder=LocalHashEmbedder(128),
                vector_store=VectorStore(store),
                answer_generator=ExtractiveAnswerGenerator(),
            )
            chat = service.create_chat("manual memory")
            updated = service.add_chat_memory(chat.id, "User prefers terse answers")
            self.assertEqual(len(updated.memories), 1)
            updated = service.add_chat_memory(chat.id, "User prefers terse answers")
            self.assertEqual(len(updated.memories), 1)
            updated = service.delete_chat_memory(chat.id, updated.memories[0].id)
            self.assertEqual(updated.memories, [])

    def test_chat_answer_defaults_update_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(data_dir=tmp_path / "data")
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            store = JsonStore(settings.state_path)
            service = RagService(
                settings=settings,
                store=store,
                embedder=LocalHashEmbedder(128),
                vector_store=VectorStore(store),
                answer_generator=ExtractiveAnswerGenerator(),
            )
            chat = service.create_chat("defaults")
            updated = service.update_chat_answer_defaults(
                chat.id,
                {"retrieval_mode": "keyword", "source_window": 1, "use_tools": False},
            )
            self.assertEqual(updated.answer_defaults.retrieval_mode, "keyword")
            self.assertEqual(updated.answer_defaults.source_window, 1)
            self.assertFalse(updated.answer_defaults.use_tools)

            updated = service.update_chat_answer_defaults(chat.id, {"source_window": None})
            self.assertEqual(updated.answer_defaults.retrieval_mode, "keyword")
            self.assertIsNone(updated.answer_defaults.source_window)

    def test_chat_export_import_duplicates_context_trace(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                chat = service.create_chat("portable", file_ids=["file-a"], knowledge_ids=["kb-a"])
                service.add_chat_memory(chat.id, "User prefers compact answers")
                service.update_chat_answer_defaults(chat.id, {"retrieval_mode": "keyword", "source_window": 1})
                message, *_ = await service.ask(chat.id, "Remember that my import color is blue", use_rag=False)
                exported = service.export_chat(chat.id)
                imported = service.import_chat(exported.chat, title="portable copy")

                self.assertNotEqual(imported.id, exported.chat.id)
                self.assertEqual(imported.title, "portable copy")
                self.assertEqual(imported.file_ids, ["file-a"])
                self.assertEqual(imported.knowledge_ids, ["kb-a"])
                self.assertEqual(imported.answer_defaults.retrieval_mode, "keyword")
                self.assertEqual(imported.answer_defaults.source_window, 1)
                self.assertEqual(imported.messages[-1].content, message.content)
                self.assertNotEqual(imported.messages[-1].id, exported.chat.messages[-1].id)
                self.assertEqual(imported.memories[0].content, exported.chat.memories[0].content)
                self.assertNotEqual(imported.memories[0].id, exported.chat.memories[0].id)

                preserved_payload = exported.chat.model_copy(deep=True)
                preserved_payload.id = "preserved-chat"
                preserved = service.import_chat(preserved_payload, title="preserved", preserve_ids=True)
                self.assertEqual(preserved.id, "preserved-chat")
                with self.assertRaises(ValueError):
                    service.import_chat(preserved_payload, preserve_ids=True)

        asyncio.run(run())

    def test_message_feedback_persists_and_can_be_cleared(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                settings = Settings(data_dir=Path(tmp) / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                )
                chat = service.create_chat("feedback")
                answer, *_ = await service.ask(chat.id, "Say hello", use_rag=False)
                updated = service.update_message_feedback(
                    chat.id,
                    answer.id,
                    rating="down",
                    tags=["Too Long", " too long ", "missing citation"],
                    comment="Needed a citation.",
                )
                feedback = updated.messages[-1].feedback
                self.assertIsNotNone(feedback)
                assert feedback is not None
                self.assertEqual(feedback.rating, "down")
                self.assertEqual(feedback.tags, ["too long", "missing citation"])
                self.assertEqual(feedback.comment, "Needed a citation.")

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.messages[-1].feedback.rating, "down")

                with self.assertRaises(ValueError):
                    service.update_message_feedback(chat.id, saved.messages[0].id, rating="up")

                cleared = service.delete_message_feedback(chat.id, answer.id)
                self.assertIsNone(cleared.messages[-1].feedback)

        asyncio.run(run())

    def test_feedback_list_filters_and_includes_context(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                settings = Settings(data_dir=Path(tmp) / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                )
                first_chat = service.create_chat("first feedback")
                first_answer, *_ = await service.ask(first_chat.id, "First feedback question", use_rag=False)
                service.update_message_feedback(
                    first_chat.id,
                    first_answer.id,
                    rating="down",
                    tags=["citation"],
                    comment="Needs sources.",
                )
                second_chat = service.create_chat("second feedback")
                second_answer, *_ = await service.ask(second_chat.id, "Second feedback question", use_rag=False)
                service.update_message_feedback(
                    second_chat.id,
                    second_answer.id,
                    rating="up",
                    tags=["helpful"],
                    comment="Good.",
                )

                all_feedback = service.list_feedback()
                self.assertEqual(all_feedback.total_count, 2)
                self.assertEqual(all_feedback.items[0].feedback.rating, "up")
                self.assertEqual(all_feedback.items[0].question, "Second feedback question")
                self.assertEqual(all_feedback.items[0].chat_title, "second feedback")

                down = service.list_feedback(rating="down")
                self.assertEqual(down.total_count, 1)
                self.assertEqual(down.items[0].feedback.comment, "Needs sources.")
                self.assertEqual(down.items[0].question, "First feedback question")

                tagged = service.list_feedback(tag="HELPFUL")
                self.assertEqual(tagged.total_count, 1)
                self.assertEqual(tagged.items[0].message_id, second_answer.id)

                limited = service.list_feedback(limit=1)
                self.assertEqual(limited.total_count, 2)
                self.assertEqual(len(limited.items), 1)

        asyncio.run(run())

    def test_backend_status_reports_provider_settings_and_storage_counts(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(
                    data_dir=tmp_path / "data",
                    chunk_size=120,
                    chunk_overlap=10,
                    top_k=3,
                    embedding_dimensions=128,
                )
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                )

                doc = tmp_path / "status.txt"
                doc.write_text("The status endpoint marker is STATUS-515.", encoding="utf-8")
                record = await service.ingest_file(doc, "status.txt", "text/plain", doc.stat().st_size)
                service.store.create_knowledge("Status KB", file_ids=[record.id])
                service.store.create_skill("Status Skill", "Keep status answers short.", triggers=["status"])
                chat = service.create_chat("status chat", file_ids=[record.id])
                await service.ask(chat.id, "What is STATUS 515?", retrieval_mode="keyword")

                status = service.backend_status()
                self.assertEqual(status.status, "ok")
                self.assertEqual(status.embedding_provider.provider, "local_hash")
                self.assertEqual(status.embedding_provider.model, "local_hash:128")
                self.assertEqual(status.llm_provider.provider, "extractive")
                self.assertEqual(status.retrieval.top_k, 3)
                self.assertEqual(status.retrieval.chunk_size, 120)
                self.assertEqual(status.storage.file_count, 1)
                self.assertGreaterEqual(status.storage.chunk_count, 1)
                self.assertEqual(status.storage.chat_count, 1)
                self.assertEqual(status.storage.message_count, 2)
                self.assertEqual(status.storage.knowledge_count, 1)
                self.assertEqual(status.storage.skill_count, 1)
                self.assertGreater(status.storage.total_file_text_chars, 0)

        asyncio.run(run())

    def test_tool_calling_calculator(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                chat = service.create_chat("tools")
                answer, _, tool_results, _, _, diagnostics, grounding = await service.ask(
                    chat.id, "Calculate 12 * 7", use_rag=False
                )
                self.assertTrue(tool_results)
                self.assertIn("84", tool_results[0].output)
                self.assertIn("84", answer.content)
                self.assertEqual(diagnostics.context_mode, "none")
                self.assertFalse(grounding.has_sources)

        asyncio.run(run())

    def test_matching_skill_is_injected_into_answer_prompt(self) -> None:
        class RecordingAnswerGenerator:
            def __init__(self) -> None:
                self.system_prompt = ""

            async def answer(self, messages, question, sources, tool_results, system_prompt=None, rag_template=None):
                self.system_prompt = system_prompt or ""
                return "skill-aware answer"

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                recorder = RecordingAnswerGenerator()
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=recorder,
                )
                skill = store.create_skill(
                    "Legal Style",
                    "Answer in a cautious legal-review tone.",
                    triggers=["contract"],
                )
                chat = service.create_chat("skills")
                answer, _, _, skills, _, _, _ = await service.ask(
                    chat.id,
                    "Review this contract clause.",
                    use_rag=False,
                )

                self.assertEqual(answer.content, "skill-aware answer")
                self.assertEqual([item.id for item in skills], [skill.id])
                self.assertIn("Legal Style", recorder.system_prompt)
                self.assertIn("cautious legal-review tone", recorder.system_prompt)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.messages[-1].skill_ids, [skill.id])

        asyncio.run(run())

    def test_matching_skill_can_force_configured_tools(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = JsonStore(settings.state_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                skill = store.create_skill(
                    "Inventory Operator",
                    "Use operational tools when an inventory request arrives.",
                    triggers=["inventory"],
                    tool_ids=["file_stats"],
                )
                chat = service.create_chat("skill tools")

                answer, _, tool_results, skills, _, _, _ = await service.ask(
                    chat.id,
                    "Give me an inventory overview.",
                    use_rag=False,
                )

                self.assertEqual([item.id for item in skills], [skill.id])
                self.assertEqual([result.tool_id for result in tool_results], ["file_stats"])
                self.assertIn("0 files indexed", tool_results[0].output)
                self.assertIn("File Stats", answer.content)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.messages[-1].tool_results[0].tool_id, "file_stats")

        asyncio.run(run())

    def test_minimum_answerability_guard_blocks_generation(self) -> None:
        class FailingAnswerGenerator:
            async def answer(self, *args, **kwargs):
                raise AssertionError("answer generator should not run when guard blocks generation")

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=FailingAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "policy.txt"
                doc.write_text("The refund window is 45 days.", encoding="utf-8")
                record = await service.ingest_file(doc, "policy.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("guard")

                answer, sources, _, _, _, diagnostics, _ = await service.ask(
                    chat.id,
                    "mars orbital sandwich",
                    file_ids=[record.id],
                    retrieval_mode="keyword",
                    minimum_answerability="medium",
                )
                self.assertFalse(sources)
                self.assertEqual(diagnostics.answerability, "none")
                self.assertIn("Minimum answerability guard blocked generation", diagnostics.warnings[-1])
                self.assertIn("do not have enough retrieved file context", answer.content)

        asyncio.run(run())

    def test_regenerate_last_answer_replaces_last_turn(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=80, chunk_overlap=0)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "regenerate.txt"
                doc.write_text(
                    "\n\n".join(
                        [
                            "Before context says the release name is atlas.",
                            "Center context says the regenerate marker is WINDOW-515.",
                            "After context says the owner is platform.",
                        ]
                    ),
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "regenerate.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("regenerate")

                first, first_sources, _, _, _, first_diagnostics, _ = await service.ask(
                    chat.id,
                    "What is the regenerate marker?",
                    file_ids=[record.id],
                    retrieval_mode="keyword",
                    top_k=1,
                )
                self.assertEqual(first.role, "assistant")
                self.assertTrue(first_sources)
                self.assertEqual(first_diagnostics.source_window, 0)

                regenerated, sources, _, _, _, diagnostics, _ = await service.regenerate_last(
                    chat.id,
                    file_ids=[record.id],
                    retrieval_mode="keyword",
                    top_k=1,
                    source_window=1,
                    max_context_chars=1000,
                )
                self.assertEqual(regenerated.role, "assistant")
                self.assertEqual(diagnostics.source_window, 1)
                self.assertTrue(sources)
                self.assertIn("release name is atlas", sources[0].text)
                self.assertIn("owner is platform", sources[0].text)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(len(saved.messages), 2)
                self.assertEqual(saved.messages[0].role, "user")
                self.assertEqual(saved.messages[0].content, "What is the regenerate marker?")
                self.assertEqual(saved.messages[1].role, "assistant")
                self.assertIsNotNone(saved.messages[1].diagnostics)
                assert saved.messages[1].diagnostics is not None
                self.assertEqual(saved.messages[1].diagnostics.source_window, 1)

        asyncio.run(run())

    def test_rerun_from_user_message_truncates_later_context(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=80, chunk_overlap=0)
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "rerun.txt"
                doc.write_text(
                    "\n\n".join(
                        [
                            "Opening context says rerun belongs to project atlas.",
                            "Middle context says the rerun marker is RERUN-818.",
                            "Closing context says the rerun owner is platform.",
                        ]
                    ),
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "rerun.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("rerun", file_ids=[record.id])

                await service.ask(chat.id, "What is the rerun marker?", retrieval_mode="keyword", top_k=1)
                await service.ask(chat.id, "What is the rerun owner?", retrieval_mode="keyword", top_k=1)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                original_first_user_id = saved.messages[0].id
                original_second_user_id = saved.messages[2].id
                original_second_assistant_id = saved.messages[3].id
                self.assertEqual(len(saved.messages), 4)

                rerun_answer, sources, _, _, _, diagnostics, _ = await service.rerun_from_message(
                    chat.id,
                    original_first_user_id,
                    retrieval_mode="keyword",
                    top_k=1,
                    source_window=1,
                    max_context_chars=1000,
                )
                self.assertEqual(rerun_answer.role, "assistant")
                self.assertEqual(diagnostics.source_window, 1)
                self.assertTrue(sources)
                self.assertIn("project atlas", sources[0].text)
                self.assertIn("RERUN-818", sources[0].text)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(len(saved.messages), 2)
                self.assertEqual(saved.messages[0].role, "user")
                self.assertEqual(saved.messages[0].content, "What is the rerun marker?")
                self.assertNotEqual(saved.messages[0].id, original_first_user_id)
                self.assertNotIn(original_second_user_id, [message.id for message in saved.messages])
                self.assertNotIn(original_second_assistant_id, [message.id for message in saved.messages])

                with self.assertRaises(ValueError):
                    await service.rerun_from_message(chat.id, saved.messages[1].id)

        asyncio.run(run())

    def test_edit_user_message_replaces_branch_with_new_text(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "edit.txt"
                doc.write_text(
                    "The original edit marker is EDIT-111. The corrected edit marker is EDIT-222.",
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "edit.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("edit", file_ids=[record.id])

                await service.ask(chat.id, "What is EDIT 111?", retrieval_mode="keyword")
                await service.ask(chat.id, "Temporary later question", use_rag=False)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                first_user_id = saved.messages[0].id
                later_user_id = saved.messages[2].id

                answer, _, _, _, retrieval_query, _, _ = await service.edit_user_message(
                    chat.id,
                    first_user_id,
                    "What is EDIT 222?",
                    retrieval_mode="keyword",
                )
                self.assertIn("EDIT-222", answer.content)
                self.assertIn("EDIT 222", retrieval_query)

                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(len(saved.messages), 2)
                self.assertEqual(saved.messages[0].content, "What is EDIT 222?")
                self.assertNotEqual(saved.messages[0].id, first_user_id)
                self.assertNotIn(later_user_id, [message.id for message in saved.messages])

                with self.assertRaises(ValueError):
                    await service.edit_user_message(chat.id, saved.messages[1].id, "Try editing assistant")
                with self.assertRaises(ValueError):
                    await service.edit_user_message(chat.id, saved.messages[0].id, "   ")

        asyncio.run(run())

    def test_stream_answer_yields_tokens_and_persists_final_message(self) -> None:
        class StreamingAnswerGenerator:
            async def answer(self, *args, **kwargs):
                return "unused"

            async def stream_answer(self, *args, **kwargs):
                yield "streamed"
                yield " answer"

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                settings.uploads_dir.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=StreamingAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                chat = service.create_chat("streaming")
                events = []
                async for event in service.stream_answer(chat.id, "Say hello", use_rag=False):
                    events.append(event)

                self.assertEqual([event for event, _ in events], ["retrieval", "token", "token", "done"])
                self.assertEqual([data for event, data in events if event == "token"], ["streamed", " answer"])
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.messages[-1].content, "streamed answer")

        asyncio.run(run())

    def test_sqlite_store_persists_files_chats_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(data_dir=tmp_path / "data")
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            store = SQLiteStore(settings.database_path)
            service = RagService(
                settings=settings,
                store=store,
                embedder=LocalHashEmbedder(128),
                vector_store=VectorStore(store),
                answer_generator=ExtractiveAnswerGenerator(),
                tool_registry=ToolRegistry(store),
            )

            async def run() -> str:
                doc = tmp_path / "manual.txt"
                doc.write_text("The launch code is blue. The backup code is green.", encoding="utf-8")
                record = await service.ingest_file(doc, "manual.txt", "text/plain", doc.stat().st_size)
                chat = service.create_chat("sqlite")
                await service.ask(chat.id, "What is the launch code?", file_ids=[record.id])
                return record.id

            file_id = asyncio.run(run())
            reopened = SQLiteStore(settings.database_path)
            self.assertEqual(len(reopened.list_files()), 1)
            self.assertEqual(len(reopened.list_chats()), 1)
            self.assertGreater(len(reopened.chunks()), 0)
            file_chunks = reopened.file_chunks(file_id)
            self.assertTrue(file_chunks)
            self.assertEqual(file_chunks[0].index, 0)
            self.assertIn("launch code", file_chunks[0].text)

            reindex_service = RagService(
                settings=settings,
                store=reopened,
                embedder=LocalHashEmbedder(128),
                vector_store=VectorStore(reopened),
                answer_generator=ExtractiveAnswerGenerator(),
                tool_registry=ToolRegistry(reopened),
            )
            reindexed = asyncio.run(reindex_service.reindex_file(file_id))
            self.assertGreater(reindexed.chunk_count, 0)

            self.assertTrue(reopened.delete_file(file_id))
            self.assertEqual(len(reopened.list_files()), 0)
            self.assertEqual(len(reopened.chunks()), 0)

    def test_file_summary_and_keywords_are_persisted_and_refreshed(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "summary.txt"
                doc.write_text(
                    "Refund policy covers claims. Refund exceptions require manager review. Shipping notes are separate.",
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "summary.txt", "text/plain", doc.stat().st_size)

                self.assertIn("Refund policy", record.summary)
                self.assertIn("refund", record.keywords)
                saved = store.get_file(record.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.summary, record.summary)

                Path(record.path).write_text("Warranty policy covers repairs. Warranty exceptions need approval.", encoding="utf-8")
                refreshed = await service.reindex_file(record.id)

                self.assertIn("Warranty policy", refreshed.summary)
                self.assertIn("warranty", refreshed.keywords)
                self.assertNotIn("refund", refreshed.keywords)

        asyncio.run(run())

    def test_file_search_ranks_by_filename_keywords_and_summary(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                refund_doc = tmp_path / "refund-guide.txt"
                shipping_doc = tmp_path / "shipping-guide.txt"
                refund_doc.write_text("Refund windows and refund exceptions require manager approval.", encoding="utf-8")
                shipping_doc.write_text("Shipping windows and courier exceptions require dispatch approval.", encoding="utf-8")
                refund = await service.ingest_file(refund_doc, "refund-guide.txt", "text/plain", refund_doc.stat().st_size)
                shipping = await service.ingest_file(
                    shipping_doc,
                    "shipping-guide.txt",
                    "text/plain",
                    shipping_doc.stat().st_size,
                )
                knowledge = store.create_knowledge("Refund KB", file_ids=[refund.id])

                result = service.search_files("refund exceptions")
                scoped = service.search_files("exceptions", knowledge_id=knowledge.id)
                limited = service.search_files("exceptions", limit=1)

                self.assertEqual(result.items[0].file.id, refund.id)
                self.assertIn("refund", result.items[0].matched_terms)
                self.assertTrue(result.items[0].reasons)
                self.assertEqual([item.file.id for item in scoped.items], [refund.id])
                self.assertEqual(len(limited.items), 1)
                self.assertEqual(limited.total_count, 2)
                self.assertTrue(any(item.file.id == shipping.id for item in result.items))

        asyncio.run(run())

    def test_chat_context_suggests_files_without_mutating_context(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                refund_doc = tmp_path / "refund-suggest.txt"
                shipping_doc = tmp_path / "shipping-suggest.txt"
                refund_doc.write_text("Refund exceptions require manager approval.", encoding="utf-8")
                shipping_doc.write_text("Shipping exceptions require dispatch approval.", encoding="utf-8")
                refund = await service.ingest_file(
                    refund_doc,
                    "refund-suggest.txt",
                    "text/plain",
                    refund_doc.stat().st_size,
                )
                shipping = await service.ingest_file(
                    shipping_doc,
                    "shipping-suggest.txt",
                    "text/plain",
                    shipping_doc.stat().st_size,
                )
                knowledge = store.create_knowledge("Support", file_ids=[refund.id, shipping.id])
                chat = service.create_chat("suggest", knowledge_ids=[knowledge.id])

                suggestion = service.suggest_chat_context(chat.id, "How do refund exceptions work?", limit=1)

                self.assertEqual(suggestion.chat_id, chat.id)
                self.assertCountEqual(suggestion.candidate_file_ids, [refund.id, shipping.id])
                self.assertEqual(suggestion.suggested_file_ids, [refund.id])
                self.assertEqual(suggestion.files[0].id, refund.id)
                self.assertIn("Refund exceptions", suggestion.files[0].summary)
                saved = store.get_chat(chat.id)
                self.assertIsNotNone(saved)
                assert saved is not None
                self.assertEqual(saved.file_ids, [])
                self.assertEqual(saved.knowledge_ids, [knowledge.id])

        asyncio.run(run())

    def test_chat_context_apply_suggestions_updates_file_context(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                refund_doc = tmp_path / "refund-apply.txt"
                shipping_doc = tmp_path / "shipping-apply.txt"
                refund_doc.write_text("Refund exceptions require manager approval.", encoding="utf-8")
                shipping_doc.write_text("Shipping exceptions require dispatch approval.", encoding="utf-8")
                refund = await service.ingest_file(
                    refund_doc,
                    "refund-apply.txt",
                    "text/plain",
                    refund_doc.stat().st_size,
                )
                shipping = await service.ingest_file(
                    shipping_doc,
                    "shipping-apply.txt",
                    "text/plain",
                    shipping_doc.stat().st_size,
                )
                chat = service.create_chat("apply")

                applied = service.apply_context_suggestions(
                    chat.id,
                    "How do refund exceptions work?",
                    file_ids=[refund.id, shipping.id],
                    limit=1,
                )

                self.assertEqual(applied.chat.file_ids, [refund.id])
                self.assertEqual(applied.applied_file_ids, [refund.id])
                self.assertFalse(applied.replaced)
                self.assertEqual(applied.suggestion.suggested_file_ids, [refund.id])

                repeated = service.apply_context_suggestions(
                    chat.id,
                    "How do refund exceptions work?",
                    file_ids=[refund.id, shipping.id],
                    limit=1,
                )

                self.assertEqual(repeated.chat.file_ids, [refund.id])
                self.assertEqual(repeated.applied_file_ids, [])

                replaced = service.apply_context_suggestions(
                    chat.id,
                    "How do shipping exceptions work?",
                    file_ids=[refund.id, shipping.id],
                    limit=1,
                    replace=True,
                )

                self.assertEqual(replaced.chat.file_ids, [shipping.id])
                self.assertEqual(replaced.applied_file_ids, [shipping.id])
                self.assertTrue(replaced.replaced)

        asyncio.run(run())

    def test_bulk_reindex_refreshes_selected_knowledge_files_and_reports_failures(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data")
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                first_doc = tmp_path / "first.txt"
                second_doc = tmp_path / "second.txt"
                first_doc.write_text("The first marker is OLD-FIRST.", encoding="utf-8")
                second_doc.write_text("The second marker is KEEP-SECOND.", encoding="utf-8")
                first = await service.ingest_file(first_doc, "first.txt", "text/plain", first_doc.stat().st_size)
                second = await service.ingest_file(second_doc, "second.txt", "text/plain", second_doc.stat().st_size)
                knowledge = store.create_knowledge("Refresh", file_ids=[first.id])

                Path(first.path).write_text("The first marker is NEW-FIRST.", encoding="utf-8")
                Path(second.path).write_text("The changed marker is NEWSECONDONLY.", encoding="utf-8")
                refreshed, failures = await service.reindex_files(knowledge_ids=[knowledge.id])

                self.assertEqual([record.id for record in refreshed], [first.id])
                self.assertEqual(failures, [])
                first_hits, _ = await service.retrieve("NEW FIRST", file_ids=[first.id], retrieval_mode="keyword")
                second_hits, _ = await service.retrieve("NEWSECONDONLY", file_ids=[second.id], retrieval_mode="keyword")
                self.assertTrue(first_hits)
                self.assertIn("NEW-FIRST", first_hits[0].text)
                self.assertFalse(second_hits)

                Path(first.path).unlink()
                refreshed, failures = await service.reindex_files(file_ids=[first.id])
                self.assertEqual(refreshed, [])
                self.assertEqual(failures[0]["file_id"], first.id)
                self.assertIn("Stored file is missing", failures[0]["error"])

        asyncio.run(run())

    def test_retrieval_narrows_attached_files_when_query_mentions_filename(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                alpha_doc = tmp_path / "alpha-policy.txt"
                beta_doc = tmp_path / "beta-policy.txt"
                alpha_doc.write_text("The target code is ALPHA-111.", encoding="utf-8")
                beta_doc.write_text("The target code is BETA-222.", encoding="utf-8")
                alpha = await service.ingest_file(alpha_doc, "alpha-policy.txt", "text/plain", alpha_doc.stat().st_size)
                beta = await service.ingest_file(beta_doc, "beta-policy.txt", "text/plain", beta_doc.stat().st_size)
                chat = service.create_chat("filename route", file_ids=[alpha.id, beta.id])

                answer, sources, _, _, _, diagnostics, _ = await service.ask(
                    chat.id,
                    "What is the target code in beta-policy.txt?",
                    retrieval_mode="keyword",
                    top_k=2,
                )

                self.assertTrue(sources)
                self.assertEqual({source.file_id for source in sources}, {beta.id})
                self.assertIn("BETA-222", answer.content)
                self.assertTrue(any("Filename-aware retrieval narrowed context" in item for item in diagnostics.warnings))

        asyncio.run(run())

    def test_sqlite_fts_exact_term_search(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=20)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "codes.txt"
                doc.write_text(
                    "The billing reference is ZXQ-991. The lunch menu includes rice and tea.",
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "codes.txt", "text/plain", doc.stat().st_size)

                text_hits = store.search_text_chunks("ZXQ 991", top_k=3, file_ids=[record.id])
                self.assertTrue(text_hits)
                self.assertIn("ZXQ-991", text_hits[0]["text"])

                rag_hits, rag_diagnostics = await service.retrieve("What is ZXQ 991?", file_ids=[record.id], top_k=1)
                self.assertTrue(rag_hits)
                self.assertIn("ZXQ-991", rag_hits[0].text)
                self.assertEqual(rag_hits[0].chunk_index, 0)
                self.assertGreaterEqual(rag_diagnostics.candidate_count, 1)

                keyword_hits, keyword_diagnostics = await service.retrieve(
                    "ZXQ 991",
                    file_ids=[record.id],
                    top_k=1,
                    retrieval_mode="keyword",
                )
                self.assertTrue(keyword_hits)
                self.assertIn("ZXQ-991", keyword_hits[0].text)
                self.assertEqual(keyword_diagnostics.retrieval_mode, "keyword")

        asyncio.run(run())

    def test_sqlite_fts_applies_file_scope_before_limit(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                target_doc = tmp_path / "target.txt"
                target_doc.write_text("The scoped lookup token is SHARED-777 target-only.", encoding="utf-8")
                target = await service.ingest_file(target_doc, "target.txt", "text/plain", target_doc.stat().st_size)
                for index in range(12):
                    distractor_doc = tmp_path / f"distractor-{index}.txt"
                    distractor_doc.write_text(
                        f"The scoped lookup token is SHARED-777 distractor-{index}.",
                        encoding="utf-8",
                    )
                    await service.ingest_file(
                        distractor_doc,
                        f"distractor-{index}.txt",
                        "text/plain",
                        distractor_doc.stat().st_size,
                    )

                hits = store.search_text_chunks("scoped lookup token SHARED 777", top_k=1, file_ids=[target.id])

                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["file_id"], target.id)
                self.assertIn("target-only", hits[0]["text"])

        asyncio.run(run())

    def test_retrieval_compresses_context_to_requested_budget(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=260, chunk_overlap=30)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "long.txt"
                doc.write_text(("alpha " * 80) + "\n\n" + ("beta " * 80), encoding="utf-8")
                record = await service.ingest_file(doc, "long.txt", "text/plain", doc.stat().st_size)
                sources, diagnostics = await service.retrieve(
                    "alpha beta",
                    file_ids=[record.id],
                    top_k=5,
                    max_context_chars=600,
                )
                self.assertTrue(sources)
                self.assertLessEqual(diagnostics.total_context_chars, 600)

        asyncio.run(run())

    def test_retrieval_can_expand_neighboring_chunk_window(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=80, chunk_overlap=0)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "window.txt"
                doc.write_text(
                    "\n\n".join(
                        [
                            "Opening context says the claim belongs to project atlas.",
                            "Middle context says the rare marker is ORBIT-424.",
                            "Closing context says the deadline is Friday.",
                        ]
                    ),
                    encoding="utf-8",
                )
                record = await service.ingest_file(doc, "window.txt", "text/plain", doc.stat().st_size)
                sources, diagnostics = await service.retrieve(
                    "ORBIT 424",
                    file_ids=[record.id],
                    top_k=1,
                    retrieval_mode="keyword",
                    source_window=1,
                    max_context_chars=1000,
                )
                self.assertTrue(sources)
                self.assertEqual(diagnostics.source_window, 1)
                self.assertEqual(sources[0].chunk_index, 1)
                self.assertEqual(sources[0].context_start_index, 0)
                self.assertEqual(sources[0].context_end_index, 2)
                self.assertIn("project atlas", sources[0].text)
                self.assertIn("ORBIT-424", sources[0].text)
                self.assertIn("deadline is Friday", sources[0].text)

        asyncio.run(run())

    def test_knowledge_base_retrieval_resolves_file_membership(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=20)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "benefits.txt"
                doc.write_text("Dental coverage starts after 90 days. Vision coverage starts immediately.", encoding="utf-8")
                record = await service.ingest_file(doc, "benefits.txt", "text/plain", doc.stat().st_size)
                knowledge = store.create_knowledge("HR benefits", "Benefits policy files", [record.id])

                sources, diagnostics = await service.retrieve(
                    "When does dental coverage start?",
                    knowledge_ids=[knowledge.id],
                    top_k=1,
                )
                self.assertTrue(sources)
                self.assertIn("Dental coverage", sources[0].text)
                self.assertGreaterEqual(diagnostics.selected_count, 1)

                self.assertTrue(store.delete_file(record.id))
                updated = store.get_knowledge(knowledge.id)
                self.assertIsNotNone(updated)
                assert updated is not None
                self.assertEqual(updated.file_ids, [])

        asyncio.run(run())

    def test_auto_context_uses_full_for_small_files_and_rag_for_large_budget_overflow(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=180, chunk_overlap=20)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                small_doc = tmp_path / "small.txt"
                small_doc.write_text("Small policy says onboarding takes five days.", encoding="utf-8")
                small = await service.ingest_file(small_doc, "small.txt", "text/plain", small_doc.stat().st_size)
                small_sources, small_diag = await service.retrieve(
                    "onboarding",
                    file_ids=[small.id],
                    context_mode="auto",
                    max_context_chars=1000,
                )
                self.assertTrue(small_sources)
                self.assertEqual(small_diag.context_mode, "auto")
                self.assertEqual(small_diag.effective_context_mode, "full")

                large_doc = tmp_path / "large.txt"
                large_doc.write_text(("Large policy says compliance review takes seven days. " * 30), encoding="utf-8")
                large = await service.ingest_file(large_doc, "large.txt", "text/plain", large_doc.stat().st_size)
                large_sources, large_diag = await service.retrieve(
                    "compliance review",
                    file_ids=[large.id],
                    context_mode="auto",
                    max_context_chars=500,
                )
                self.assertTrue(large_sources)
                self.assertEqual(large_diag.context_mode, "auto")
                self.assertEqual(large_diag.effective_context_mode, "rag")

        asyncio.run(run())

    def test_prompt_template_renders_source_context(self) -> None:
        messages = build_prompt_messages(
            messages=[],
            question="What is the code?",
            sources=[
                Source(
                    file_id="f1",
                    filename="codes.txt",
                    chunk_id="f1:0",
                    chunk_index=2,
                    context_start_index=1,
                    context_end_index=3,
                    start_char=20,
                    end_char=42,
                    score=0.9,
                    text="The code is blue.",
                )
            ],
            tool_results=[],
            system_prompt="Answer with citations.",
            rag_template="CONTEXT:\n{context}\nQUESTION:\n{question}",
        )
        self.assertIn("Answer with citations.", messages[0]["content"])
        self.assertIn("CONTEXT:", messages[0]["content"])
        self.assertIn("The code is blue.", messages[0]["content"])
        self.assertIn("chunk_id='f1:0'", messages[0]["content"])
        self.assertIn("chunk_index='2'", messages[0]["content"])
        self.assertIn("context_range='1-3'", messages[0]["content"])
        self.assertIn("char_range='20-42'", messages[0]["content"])
        self.assertIn("QUESTION:\nWhat is the code?", messages[0]["content"])

    def test_extractive_answer_uses_citation_style_sources(self) -> None:
        async def run() -> None:
            answer = await ExtractiveAnswerGenerator().answer(
                messages=[],
                question="What is the refund window?",
                sources=[
                    Source(
                        file_id="f1",
                        filename="policy.txt",
                        chunk_id="f1:0",
                        score=0.9,
                        text="Customers can request a refund within 30 days. Shipping takes two days.",
                    )
                ],
                tool_results=[],
            )
            self.assertIn("[1]", answer)
            self.assertIn("policy.txt", answer)
            self.assertIn("30 days", answer)

        asyncio.run(run())

    def test_grounding_citations_include_source_offsets(self) -> None:
        source = Source(
            file_id="f1",
            filename="policy.txt",
            chunk_id="f1:2",
            chunk_index=2,
            context_start_index=1,
            context_end_index=3,
            start_char=120,
            end_char=240,
            score=0.8,
            text="Refund exceptions require manager approval.",
        )
        grounding = analyze_grounding("Use the exception process [1].", [source])
        self.assertEqual(len(grounding.citations), 1)
        citation = grounding.citations[0]
        self.assertEqual(citation.chunk_index, 2)
        self.assertEqual(citation.context_start_index, 1)
        self.assertEqual(citation.context_end_index, 3)
        self.assertEqual(citation.start_char, 120)
        self.assertEqual(citation.end_char, 240)

    def test_answer_audit_scores_supported_and_unsupported_sentences(self) -> None:
        source = Source(
            file_id="f1",
            filename="policy.txt",
            chunk_id="f1:0",
            score=0.9,
            text="Refunds are available within 30 days. Manager approval is required for exceptions.",
        )
        answer = "Refunds are available within 30 days [1]. The office is on Mars."
        grounding = analyze_grounding(answer, [source])
        audit = audit_answer(answer, [source], grounding, message_id="m1")
        self.assertEqual(audit.message_id, "m1")
        self.assertFalse(audit.answer_supported)
        self.assertEqual(audit.sentence_count, 2)
        self.assertEqual(audit.supported_count, 1)
        self.assertEqual(audit.unsupported_count, 1)
        self.assertEqual(audit.sentences[0].cited_markers, ["[1]"])
        self.assertEqual(audit.sentences[0].status, "supported")
        self.assertEqual(audit.sentences[1].status, "unsupported")
        self.assertTrue(audit.warnings)

    def test_retrieval_can_limit_sources_per_file_for_diversity(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=10)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )

                first = tmp_path / "refund-a.txt"
                first.write_text(
                    "\n\n".join(
                        [
                            "Refund policy alpha says refunds take 10 days.",
                            "Refund policy beta says refunds require receipts.",
                            "Refund policy gamma says refunds use original payment.",
                        ]
                    ),
                    encoding="utf-8",
                )
                second = tmp_path / "refund-b.txt"
                second.write_text("Refund escalation policy says managers review exceptions.", encoding="utf-8")
                first_record = await service.ingest_file(first, "refund-a.txt", "text/plain", first.stat().st_size)
                second_record = await service.ingest_file(second, "refund-b.txt", "text/plain", second.stat().st_size)

                sources, _ = await service.retrieve(
                    "refund policy",
                    file_ids=[first_record.id, second_record.id],
                    top_k=4,
                    max_sources_per_file=1,
                )
                self.assertGreaterEqual(len(sources), 2)
                counts: dict[str, int] = {}
                for source in sources:
                    counts[source.file_id] = counts.get(source.file_id, 0) + 1
                self.assertTrue(all(count == 1 for count in counts.values()))
                self.assertIn(first_record.id, counts)
                self.assertIn(second_record.id, counts)

        asyncio.run(run())

    def test_retrieval_diagnostics_report_mmr_diversity(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=10)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                first = tmp_path / "alpha.txt"
                first.write_text("Refund policy requires receipts. Refund policy allows returns.", encoding="utf-8")
                second = tmp_path / "beta.txt"
                second.write_text("Refund escalation goes to a manager.", encoding="utf-8")
                first_record = await service.ingest_file(first, "alpha.txt", "text/plain", first.stat().st_size)
                second_record = await service.ingest_file(second, "beta.txt", "text/plain", second.stat().st_size)

                sources, diagnostics = await service.retrieve(
                    "refund policy",
                    file_ids=[first_record.id, second_record.id],
                    top_k=2,
                    diversity="mmr",
                    mmr_lambda=0.4,
                )
                self.assertTrue(sources)
                self.assertEqual(diagnostics.diversity, "mmr")
                self.assertEqual(diagnostics.mmr_lambda, 0.4)

        asyncio.run(run())

    def test_chat_uses_saved_knowledge_context_without_repeating_ids(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                settings = Settings(data_dir=tmp_path / "data", chunk_size=160, chunk_overlap=20)
                store = SQLiteStore(settings.database_path)
                service = RagService(
                    settings=settings,
                    store=store,
                    embedder=LocalHashEmbedder(128),
                    vector_store=VectorStore(store),
                    answer_generator=ExtractiveAnswerGenerator(),
                    tool_registry=ToolRegistry(store),
                )
                doc = tmp_path / "context.txt"
                doc.write_text("The retained context answer is marigold.", encoding="utf-8")
                record = await service.ingest_file(doc, "context.txt", "text/plain", doc.stat().st_size)
                knowledge = store.create_knowledge("Saved context", file_ids=[record.id])
                chat = service.create_chat("context chat", knowledge_ids=[knowledge.id])

                answer, sources, _, _, _, diagnostics, grounding = await service.ask(chat.id, "What is the retained answer?")
                self.assertTrue(sources)
                self.assertIn("marigold", answer.content.lower())
                self.assertGreaterEqual(diagnostics.selected_count, 1)
                self.assertTrue(grounding.citations)
                self.assertEqual(grounding.citations[0].chunk_index, sources[0].chunk_index)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
