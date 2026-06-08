from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("app.main", None)

    def test_app_imports_and_retrieval_returns_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("api.txt", b"The secret marker is API-777.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            upload_body = upload.json()
            file_id = upload_body["id"]
            self.assertIn("secret marker", upload_body["summary"].lower())
            self.assertIn("marker", upload_body["keywords"])

            status = client.get("/api/status")
            self.assertEqual(status.status_code, 200)
            status_body = status.json()
            self.assertEqual(status_body["status"], "ok")
            self.assertEqual(status_body["embedding_provider"]["provider"], "local_hash")
            self.assertTrue(status_body["embedding_provider"]["configured"])
            self.assertIn("retrieval", status_body)
            self.assertGreaterEqual(status_body["storage"]["file_count"], 1)
            self.assertGreaterEqual(status_body["storage"]["chunk_count"], 1)

            response = client.post(
                "/api/retrieval/search",
                json={
                    "query": "API 777",
                    "file_ids": [file_id],
                    "retrieval_mode": "keyword",
                    "max_context_chars": 1000,
                    "source_window": 0,
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["sources"])
            self.assertIn("chunk_index", body["sources"][0])
            self.assertEqual(body["diagnostics"]["retrieval_mode"], "keyword")
            self.assertGreaterEqual(body["diagnostics"]["selected_count"], 1)
            self.assertIn(body["diagnostics"]["answerability"], ["medium", "high"])
            self.assertGreater(body["diagnostics"]["query_term_coverage"], 0)

            chunks = client.get(f"/api/files/{file_id}/chunks")
            self.assertEqual(chunks.status_code, 200)
            self.assertEqual(chunks.json()[0]["index"], 0)
            self.assertEqual(chunks.json()[0]["start_char"], 0)
            self.assertGreater(chunks.json()[0]["end_char"], chunks.json()[0]["start_char"])
            self.assertIn("API-777", chunks.json()[0]["text"])
            self.assertEqual(body["sources"][0]["start_char"], chunks.json()[0]["start_char"])
            self.assertEqual(body["sources"][0]["end_char"], chunks.json()[0]["end_char"])

            full_text = client.get(f"/api/files/{file_id}/text")
            self.assertEqual(full_text.status_code, 200)
            self.assertEqual(full_text.json()["text"], "The secret marker is API-777.")
            self.assertEqual(full_text.json()["total_chars"], len("The secret marker is API-777."))

            summary = client.get(f"/api/files/{file_id}/summary")
            self.assertEqual(summary.status_code, 200)
            summary_body = summary.json()
            self.assertEqual(summary_body["file_id"], file_id)
            self.assertIn("API-777", summary_body["summary"])
            self.assertIn("marker", summary_body["keywords"])

            chunk = chunks.json()[0]
            text_slice = client.get(
                f"/api/files/{file_id}/text?start={chunk['start_char']}&end={chunk['end_char']}"
            )
            self.assertEqual(text_slice.status_code, 200)
            self.assertEqual(text_slice.json()["text"], chunk["text"])
            self.assertEqual(text_slice.json()["start_char"], chunk["start_char"])
            self.assertEqual(text_slice.json()["end_char"], chunk["end_char"])

            invalid = client.get(f"/api/files/{file_id}/text?start=10&end=5")
            self.assertEqual(invalid.status_code, 400)

    def test_retrieval_search_routes_to_filename_mentioned_in_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            alpha = client.post(
                "/api/files",
                files={"file": ("alpha-guide.txt", b"The shared lookup code is ALPHA-101.", "text/plain")},
            )
            beta = client.post(
                "/api/files",
                files={"file": ("beta-guide.txt", b"The shared lookup code is BETA-202.", "text/plain")},
            )
            self.assertEqual(alpha.status_code, 200)
            self.assertEqual(beta.status_code, 200)

            routed = client.post(
                "/api/retrieval/search",
                json={
                    "query": "What is the shared lookup code in beta-guide.txt?",
                    "file_ids": [alpha.json()["id"], beta.json()["id"]],
                    "retrieval_mode": "keyword",
                    "top_k": 2,
                },
            )

            self.assertEqual(routed.status_code, 200)
            body = routed.json()
            self.assertTrue(body["sources"])
            self.assertEqual({source["filename"] for source in body["sources"]}, {"beta-guide.txt"})
            self.assertIn("BETA-202", body["sources"][0]["text"])
            self.assertTrue(
                any("Filename-aware retrieval narrowed context" in warning for warning in body["diagnostics"]["warnings"])
            )

    def test_file_search_endpoint_uses_summaries_keywords_and_knowledge_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            refund = client.post(
                "/api/files",
                files={"file": ("refund-search.txt", b"Refund exceptions require manager approval.", "text/plain")},
            )
            shipping = client.post(
                "/api/files",
                files={"file": ("shipping-search.txt", b"Shipping exceptions require dispatch approval.", "text/plain")},
            )
            self.assertEqual(refund.status_code, 200)
            self.assertEqual(shipping.status_code, 200)
            knowledge = client.post(
                "/api/knowledge",
                json={"name": "Refund Search", "file_ids": [refund.json()["id"]]},
            )
            self.assertEqual(knowledge.status_code, 200)

            result = client.get("/api/files/search?q=refund+exceptions")
            self.assertEqual(result.status_code, 200)
            body = result.json()
            self.assertEqual(body["items"][0]["file"]["id"], refund.json()["id"])
            self.assertIn("refund", body["items"][0]["matched_terms"])
            self.assertTrue(body["items"][0]["reasons"])

            scoped = client.get(f"/api/files/search?q=exceptions&knowledge_id={knowledge.json()['id']}")
            self.assertEqual(scoped.status_code, 200)
            scoped_body = scoped.json()
            self.assertEqual(scoped_body["total_count"], 1)
            self.assertEqual(scoped_body["items"][0]["file"]["id"], refund.json()["id"])

            invalid_limit = client.get("/api/files/search?q=refund&limit=0")
            self.assertEqual(invalid_limit.status_code, 400)
            missing_knowledge = client.get("/api/files/search?q=refund&knowledge_id=missing")
            self.assertEqual(missing_knowledge.status_code, 404)

    def test_z_bulk_reindex_endpoints_refresh_file_and_knowledge_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("bulk-reindex.txt", b"The visible marker is BEFOREONLY.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_body = upload.json()
            file_id = file_body["id"]
            Path(file_body["path"]).write_text("The visible marker is AFTERONLY.", encoding="utf-8")

            stale = client.post(
                "/api/retrieval/search",
                json={"query": "AFTERONLY", "file_ids": [file_id], "retrieval_mode": "keyword"},
            )
            self.assertEqual(stale.status_code, 200)
            self.assertEqual(stale.json()["sources"], [])

            refreshed = client.post("/api/files/reindex", json={"file_ids": [file_id]})
            self.assertEqual(refreshed.status_code, 200)
            self.assertEqual(refreshed.json()["requested_count"], 1)
            self.assertEqual(refreshed.json()["reindexed_count"], 1)
            self.assertEqual(refreshed.json()["failures"], [])

            fresh = client.post(
                "/api/retrieval/search",
                json={"query": "AFTERONLY", "file_ids": [file_id], "retrieval_mode": "keyword"},
            )
            self.assertEqual(fresh.status_code, 200)
            self.assertTrue(fresh.json()["sources"])
            self.assertIn("AFTERONLY", fresh.json()["sources"][0]["text"])

            knowledge = client.post("/api/knowledge", json={"name": "Bulk", "file_ids": [file_id]})
            self.assertEqual(knowledge.status_code, 200)
            knowledge_id = knowledge.json()["id"]
            Path(file_body["path"]).write_text("The visible marker is KNOWLEDGEONLY.", encoding="utf-8")

            knowledge_refreshed = client.post(f"/api/knowledge/{knowledge_id}/reindex")
            self.assertEqual(knowledge_refreshed.status_code, 200)
            self.assertEqual(knowledge_refreshed.json()["reindexed_count"], 1)
            knowledge_search = client.post(
                "/api/retrieval/search",
                json={"query": "KNOWLEDGEONLY", "knowledge_ids": [knowledge_id], "retrieval_mode": "keyword"},
            )
            self.assertEqual(knowledge_search.status_code, 200)
            self.assertTrue(knowledge_search.json()["sources"])
            self.assertIn("KNOWLEDGEONLY", knowledge_search.json()["sources"][0]["text"])

            missing = client.post("/api/knowledge/not-real/reindex")
            self.assertEqual(missing.status_code, 404)

    def test_retrieval_api_expands_neighboring_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={
                    "file": (
                        "window-api.txt",
                        (
                            b"Before context names project lotus.\n\n"
                            b"Center context contains WINDOW 515.\n\n"
                            b"After context names project river."
                        ),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]

            response = client.post(
                "/api/retrieval/search",
                json={
                    "query": "WINDOW 515",
                    "file_ids": [file_id],
                    "context_mode": "full",
                    "retrieval_mode": "hybrid",
                    "top_k": 1,
                    "source_window": 1,
                    "diversity": "mmr",
                    "mmr_lambda": 0.4,
                    "max_context_chars": 1000,
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["diagnostics"]["source_window"], 1)
            self.assertEqual(body["diagnostics"]["diversity"], "mmr")
            self.assertEqual(body["diagnostics"]["mmr_lambda"], 0.4)
            self.assertTrue(body["sources"])
            self.assertEqual(body["sources"][0]["context_start_index"], 0)
            self.assertIn("project lotus", body["sources"][0]["text"])
            self.assertIn("WINDOW 515", body["sources"][0]["text"])
            self.assertIn("project river", body["sources"][0]["text"])

    def test_file_chunk_window_endpoint_returns_neighboring_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={
                    "file": (
                        "chunk-window.txt",
                        (
                            (b"Opening chunk text belongs to alpha. " * 35)
                            + b"\n\n"
                            + (b"Middle chunk text contains WINDOW-909. " * 35)
                            + b"\n\n"
                            + (b"Closing chunk text belongs to omega. " * 35)
                        ),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chunks = client.get(f"/api/files/{file_id}/chunks")
            self.assertEqual(chunks.status_code, 200)
            self.assertGreaterEqual(len(chunks.json()), 3)

            response = client.get(f"/api/files/{file_id}/chunks/1/window?window=1")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["target_index"], 1)
            self.assertEqual(body["start_index"], 0)
            self.assertEqual(body["end_index"], 2)
            self.assertFalse(body["has_previous"])
            self.assertEqual([chunk["index"] for chunk in body["chunks"]], [0, 1, 2])
            self.assertTrue(all(chunk["end_char"] > chunk["start_char"] for chunk in body["chunks"]))
            self.assertIn("[chunk 1]", body["context_text"])

            edge = client.get(f"/api/files/{file_id}/chunks/0/window?window=0")
            self.assertEqual(edge.status_code, 200)
            self.assertEqual(edge.json()["start_index"], 0)
            self.assertTrue(edge.json()["has_next"])

            marker_index = next(
                chunk["index"] for chunk in chunks.json() if "WINDOW-909" in chunk["text"]
            )
            marker_window = client.get(f"/api/files/{file_id}/chunks/{marker_index}/window?window=0")
            self.assertEqual(marker_window.status_code, 200)
            self.assertIn("WINDOW-909", marker_window.json()["context_text"])

            invalid = client.get(f"/api/files/{file_id}/chunks/0/window?window=11")
            self.assertEqual(invalid.status_code, 400)
            missing = client.get(f"/api/files/{file_id}/chunks/99/window")
            self.assertEqual(missing.status_code, 404)

    def test_retrieval_source_pack_groups_evidence_for_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            first = client.post(
                "/api/files",
                files={"file": ("policy-a.txt", b"Refund policy says refunds take 30 days.", "text/plain")},
            )
            second = client.post(
                "/api/files",
                files={"file": ("policy-b.txt", b"Refund policy exceptions need manager approval.", "text/plain")},
            )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            file_ids = [first.json()["id"], second.json()["id"]]

            response = client.post(
                "/api/retrieval/source-pack",
                json={
                    "query": "refund policy exceptions",
                    "file_ids": file_ids,
                    "context_mode": "full",
                    "top_k": 2,
                    "max_sources_per_file": 1,
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["query"], "refund policy exceptions")
            self.assertEqual(len(body["sources"]), 2)
            self.assertEqual(len(body["files"]), 2)
            self.assertIn("[1]", body["context_text"])
            self.assertIn("chunk", body["context_text"])
            self.assertTrue(body["files"][0]["markers"])
            self.assertIn("marker", body["files"][0]["sources"][0])
            self.assertIn("excerpt", body["files"][0]["sources"][0])
            self.assertTrue(body["files"][0]["sources"][0]["matched_terms"])

    def test_retrieval_explain_endpoint_returns_candidate_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            first = client.post(
                "/api/files",
                files={"file": ("explain-a.txt", b"Explain candidate refund policy marker A requires receipts.", "text/plain")},
            )
            second = client.post(
                "/api/files",
                files={"file": ("explain-b.txt", b"Explain candidate refund policy marker B requires manager approval.", "text/plain")},
            )
            third = client.post(
                "/api/files",
                files={"file": ("explain-c.txt", b"Explain candidate refund policy marker C requires finance review.", "text/plain")},
            )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(third.status_code, 200)

            response = client.post(
                "/api/retrieval/explain",
                json={
                    "query": "explain candidate refund policy marker",
                    "file_ids": [first.json()["id"], second.json()["id"], third.json()["id"]],
                    "retrieval_mode": "keyword",
                    "top_k": 2,
                    "candidate_limit": 10,
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["query"], "explain candidate refund policy marker")
            self.assertEqual(body["diagnostics"]["selected_count"], 2)
            self.assertEqual(len(body["source_pack"]["sources"]), 2)
            self.assertGreaterEqual(len(body["candidates"]), 3)
            self.assertTrue(any(candidate["selected"] for candidate in body["candidates"]))
            self.assertTrue(any(not candidate["selected"] for candidate in body["candidates"]))
            self.assertIn("refund", body["candidates"][0]["matched_terms"])
            self.assertTrue(body["candidates"][0]["reasons"])

    def test_chat_answerability_guard_blocks_weak_file_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("guard.txt", b"The refund window is 45 days.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chat = client.post("/api/chats", json={"title": "guard", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={
                    "message": "mars orbital sandwich",
                    "retrieval_mode": "keyword",
                    "minimum_answerability": "medium",
                },
            )
            self.assertEqual(answer.status_code, 200)
            body = answer.json()
            self.assertEqual(body["diagnostics"]["answerability"], "none")
            self.assertTrue(body["diagnostics"]["warnings"])
            self.assertIn("Required answerability: medium", body["message"]["content"])
            self.assertEqual(body["message"]["diagnostics"]["answerability"], "none")

    def test_chat_regenerate_endpoint_replaces_last_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={
                    "file": (
                        "regen-api.txt",
                        (
                            b"Before context names project atlas.\n\n"
                            b"Center context contains REGEN-424.\n\n"
                            b"After context names team platform."
                        ),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chat = client.post("/api/chats", json={"title": "regen api", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={
                    "message": "What contains REGEN 424?",
                    "retrieval_mode": "keyword",
                    "top_k": 1,
                },
            )
            self.assertEqual(answer.status_code, 200)
            self.assertEqual(answer.json()["diagnostics"]["source_window"], 0)

            regenerated = client.post(
                f"/api/chats/{chat_id}/messages/regenerate",
                json={
                    "retrieval_mode": "keyword",
                    "top_k": 1,
                    "source_window": 1,
                    "max_context_chars": 1000,
                },
            )
            self.assertEqual(regenerated.status_code, 200)
            body = regenerated.json()
            self.assertEqual(body["diagnostics"]["source_window"], 1)
            self.assertIn("project atlas", body["sources"][0]["text"])
            self.assertIn("team platform", body["sources"][0]["text"])

            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            messages = saved.json()["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[1]["role"], "assistant")
            self.assertEqual(messages[1]["diagnostics"]["source_window"], 1)

    def test_chat_rerun_endpoint_truncates_after_selected_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={
                    "file": (
                        "rerun-api.txt",
                        (
                            b"Before context names rerun project atlas.\n\n"
                            b"Center context contains RERUN-API-515.\n\n"
                            b"After context names rerun team platform."
                        ),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chat = client.post("/api/chats", json={"title": "rerun api", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            first = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "What contains RERUN API 515?", "retrieval_mode": "keyword", "top_k": 1},
            )
            self.assertEqual(first.status_code, 200)
            second = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Who is the rerun team?", "retrieval_mode": "keyword", "top_k": 1},
            )
            self.assertEqual(second.status_code, 200)
            saved = client.get(f"/api/chats/{chat_id}").json()
            first_user_id = saved["messages"][0]["id"]
            second_user_id = saved["messages"][2]["id"]

            rerun = client.post(
                f"/api/chats/{chat_id}/messages/{first_user_id}/rerun",
                json={
                    "retrieval_mode": "keyword",
                    "top_k": 1,
                    "source_window": 1,
                    "max_context_chars": 1000,
                },
            )
            self.assertEqual(rerun.status_code, 200)
            body = rerun.json()
            self.assertEqual(body["diagnostics"]["source_window"], 1)
            self.assertIn("project atlas", body["sources"][0]["text"])
            self.assertIn("team platform", body["sources"][0]["text"])

            saved = client.get(f"/api/chats/{chat_id}").json()
            self.assertEqual(len(saved["messages"]), 2)
            self.assertEqual(saved["messages"][0]["content"], "What contains RERUN API 515?")
            self.assertNotIn(second_user_id, [message["id"] for message in saved["messages"]])

            assistant_rerun = client.post(
                f"/api/chats/{chat_id}/messages/{saved['messages'][1]['id']}/rerun",
                json={},
            )
            self.assertEqual(assistant_rerun.status_code, 400)

    def test_chat_edit_endpoint_replaces_user_message_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("edit-api.txt", b"The original marker is API-111. The corrected marker is API-222.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chat = client.post("/api/chats", json={"title": "edit api", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            first = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "What is API 111?", "retrieval_mode": "keyword"},
            )
            self.assertEqual(first.status_code, 200)
            second = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Temporary later question", "use_rag": False},
            )
            self.assertEqual(second.status_code, 200)
            saved = client.get(f"/api/chats/{chat_id}").json()
            first_user_id = saved["messages"][0]["id"]
            later_user_id = saved["messages"][2]["id"]

            edited = client.post(
                f"/api/chats/{chat_id}/messages/{first_user_id}/edit",
                json={"message": "What is API 222?", "retrieval_mode": "keyword"},
            )
            self.assertEqual(edited.status_code, 200)
            body = edited.json()
            self.assertIn("API-222", body["message"]["content"])
            self.assertIn("API 222", body["retrieval_query"])

            saved = client.get(f"/api/chats/{chat_id}").json()
            self.assertEqual(len(saved["messages"]), 2)
            self.assertEqual(saved["messages"][0]["content"], "What is API 222?")
            self.assertNotEqual(saved["messages"][0]["id"], first_user_id)
            self.assertNotIn(later_user_id, [message["id"] for message in saved["messages"]])

            assistant_edit = client.post(
                f"/api/chats/{chat_id}/messages/{saved['messages'][1]['id']}/edit",
                json={"message": "Try editing assistant"},
            )
            self.assertEqual(assistant_edit.status_code, 400)
            empty_edit = client.post(
                f"/api/chats/{chat_id}/messages/{saved['messages'][0]['id']}/edit",
                json={"message": "   "},
            )
            self.assertEqual(empty_edit.status_code, 400)

    def test_chat_stream_endpoint_emits_events_and_persists_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            chat = client.post("/api/chats", json={"title": "stream api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            with client.stream(
                "POST",
                f"/api/chats/{chat_id}/messages/stream",
                json={"message": "Calculate 6 * 7", "use_rag": False},
            ) as response:
                self.assertEqual(response.status_code, 200)
                body = response.read().decode("utf-8")

            self.assertIn("event: retrieval", body)
            self.assertIn("event: tools", body)
            self.assertIn("event: token", body)
            self.assertIn("event: done", body)
            self.assertIn("42", body)

            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            messages = saved.json()["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[-1]["role"], "assistant")
            self.assertIn("42", messages[-1]["content"])

    def test_knowledge_api_drives_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("kb.txt", b"Support tier gold includes priority callback.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]

            knowledge = client.post(
                "/api/knowledge",
                json={"name": "Support KB", "description": "support docs", "file_ids": [file_id]},
            )
            self.assertEqual(knowledge.status_code, 200)
            knowledge_id = knowledge.json()["id"]

            response = client.post(
                "/api/retrieval/search",
                json={"query": "priority callback", "knowledge_ids": [knowledge_id], "top_k": 1},
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["sources"])
            self.assertIn("priority callback", body["sources"][0]["text"])

            removed = client.delete(f"/api/knowledge/{knowledge_id}/files/{file_id}")
            self.assertEqual(removed.status_code, 200)
            self.assertEqual(removed.json()["file_ids"], [])

    def test_upload_file_directly_into_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            knowledge = client.post("/api/knowledge", json={"name": "Direct Upload KB"})
            self.assertEqual(knowledge.status_code, 200)
            knowledge_id = knowledge.json()["id"]

            upload = client.post(
                f"/api/knowledge/{knowledge_id}/files/upload",
                files={"file": ("direct.txt", b"Direct upload files are searchable immediately.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            body = upload.json()
            file_id = body["file"]["id"]
            self.assertIn(file_id, body["knowledge"]["file_ids"])

            response = client.post(
                "/api/retrieval/search",
                json={"query": "searchable immediately", "knowledge_ids": [knowledge_id], "top_k": 1},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["sources"])

    def test_batch_upload_into_knowledge_base_searches_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            knowledge = client.post("/api/knowledge", json={"name": "Batch KB"})
            self.assertEqual(knowledge.status_code, 200)
            knowledge_id = knowledge.json()["id"]

            upload = client.post(
                f"/api/knowledge/{knowledge_id}/files/upload/batch",
                files=[
                    ("files", ("refund.txt", b"Refunds are available for 30 days.", "text/plain")),
                    ("files", ("shipping.txt", b"Express shipping arrives in two days.", "text/plain")),
                ],
            )
            self.assertEqual(upload.status_code, 200)
            body = upload.json()
            self.assertEqual(len(body["files"]), 2)
            self.assertEqual(len(body["knowledge"]["file_ids"]), 2)

            refund = client.post(
                "/api/retrieval/search",
                json={"query": "refund days", "knowledge_ids": [knowledge_id], "top_k": 1},
            )
            self.assertEqual(refund.status_code, 200)
            self.assertIn("Refunds", refund.json()["sources"][0]["text"])

            shipping = client.post(
                "/api/retrieval/search",
                json={"query": "express shipping", "knowledge_ids": [knowledge_id], "top_k": 1},
            )
            self.assertEqual(shipping.status_code, 200)
            self.assertIn("Express shipping", shipping.json()["sources"][0]["text"])

    def test_chat_context_endpoint_attaches_knowledge_for_later_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("remembered.txt", b"The remembered backend color is indigo.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            knowledge = client.post("/api/knowledge", json={"name": "Remembered KB", "file_ids": [file_id]})
            self.assertEqual(knowledge.status_code, 200)
            knowledge_id = knowledge.json()["id"]

            chat = client.post("/api/chats", json={"title": "context api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            updated = client.put(f"/api/chats/{chat_id}/context", json={"knowledge_ids": [knowledge_id]})
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(updated.json()["knowledge_ids"], [knowledge_id])

            answer = client.post(f"/api/chats/{chat_id}/messages", json={"message": "What is the backend color?"})
            self.assertEqual(answer.status_code, 200)
            body = answer.json()
            self.assertTrue(body["sources"])
            self.assertIn("indigo", body["message"]["content"].lower())
            self.assertTrue(body["grounding"]["has_sources"])
            self.assertEqual(body["grounding"]["citations"][0]["filename"], "remembered.txt")
            self.assertIn("chunk_index", body["grounding"]["citations"][0])
            self.assertEqual(body["grounding"]["citations"][0]["start_char"], body["sources"][0]["start_char"])
            self.assertEqual(body["grounding"]["citations"][0]["end_char"], body["sources"][0]["end_char"])
            self.assertEqual(body["message"]["answer_quality"]["status"], "supported")
            self.assertTrue(body["message"]["answer_quality"]["answer_supported"])
            self.assertGreater(body["message"]["answer_quality"]["support_score"], 0)

            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            assistant = saved.json()["messages"][-1]
            self.assertIn("What is the backend color?", assistant["retrieval_query"])
            self.assertEqual(assistant["diagnostics"]["answerability"], body["diagnostics"]["answerability"])
            self.assertEqual(assistant["grounding"]["citations"][0]["filename"], "remembered.txt")
            self.assertEqual(assistant["grounding"]["citations"][0]["start_char"], body["sources"][0]["start_char"])
            self.assertEqual(assistant["answer_quality"]["status"], "supported")

    def test_chat_context_preview_endpoint_shows_planned_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("preview-api.txt", b"The preview backend color is violet.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            knowledge = client.post("/api/knowledge", json={"name": "Preview API KB", "file_ids": [file_id]})
            self.assertEqual(knowledge.status_code, 200)
            knowledge_id = knowledge.json()["id"]
            chat = client.post("/api/chats", json={"title": "preview api", "knowledge_ids": [knowledge_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            memory = client.post(
                f"/api/chats/{chat_id}/memories",
                json={"content": "User's preview mode is careful"},
            )
            self.assertEqual(memory.status_code, 200)
            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "What is the preview backend color?"},
            )
            self.assertEqual(answer.status_code, 200)

            preview = client.post(
                f"/api/chats/{chat_id}/context/preview",
                json={"message": "And preview mode?"},
            )
            self.assertEqual(preview.status_code, 200)
            body = preview.json()
            self.assertEqual(body["chat_id"], chat_id)
            self.assertIn(file_id, body["resolved_file_ids"])
            self.assertEqual(body["files"][0]["id"], file_id)
            self.assertEqual(body["files"][0]["filename"], "preview-api.txt")
            self.assertIn("preview backend color", body["files"][0]["summary"].lower())
            self.assertIn("preview", body["files"][0]["keywords"])
            self.assertIn("careful", body["memory_context"])
            self.assertTrue(body["rolling_messages"])
            self.assertTrue(body["planned_queries"])
            self.assertIn("And preview mode?", body["retrieval_query"])

    def test_chat_context_suggest_endpoint_returns_ranked_file_overviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            refund = client.post(
                "/api/files",
                files={"file": ("refund-suggest-api.txt", b"Refund exceptions require manager approval.", "text/plain")},
            )
            shipping = client.post(
                "/api/files",
                files={"file": ("shipping-suggest-api.txt", b"Shipping exceptions require dispatch approval.", "text/plain")},
            )
            self.assertEqual(refund.status_code, 200)
            self.assertEqual(shipping.status_code, 200)
            knowledge = client.post(
                "/api/knowledge",
                json={"name": "Suggest API KB", "file_ids": [refund.json()["id"], shipping.json()["id"]]},
            )
            self.assertEqual(knowledge.status_code, 200)
            chat = client.post("/api/chats", json={"title": "suggest api", "knowledge_ids": [knowledge.json()["id"]]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            suggestion = client.post(
                f"/api/chats/{chat_id}/context/suggest",
                json={"message": "How do refund exceptions work?", "limit": 1},
            )

            self.assertEqual(suggestion.status_code, 200)
            body = suggestion.json()
            self.assertEqual(body["chat_id"], chat_id)
            self.assertEqual(body["suggested_file_ids"], [refund.json()["id"]])
            self.assertEqual(body["files"][0]["filename"], "refund-suggest-api.txt")
            self.assertIn("Refund exceptions", body["files"][0]["summary"])
            self.assertIn("refund", body["suggestions"][0]["matched_terms"])

            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()["file_ids"], [])

    def test_chat_context_apply_suggestions_endpoint_saves_context_for_later_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            refund = client.post(
                "/api/files",
                files={
                    "file": (
                        "refund-apply-api.txt",
                        b"Refund exceptions require manager approval from support leadership.",
                        "text/plain",
                    )
                },
            )
            shipping = client.post(
                "/api/files",
                files={
                    "file": (
                        "shipping-apply-api.txt",
                        b"Shipping exceptions require dispatch approval before release.",
                        "text/plain",
                    )
                },
            )
            self.assertEqual(refund.status_code, 200)
            self.assertEqual(shipping.status_code, 200)
            refund_id = refund.json()["id"]
            shipping_id = shipping.json()["id"]
            chat = client.post("/api/chats", json={"title": "apply suggestion api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            applied = client.post(
                f"/api/chats/{chat_id}/context/apply-suggestions",
                json={
                    "message": "How do refund exceptions work?",
                    "file_ids": [refund_id, shipping_id],
                    "limit": 1,
                },
            )

            self.assertEqual(applied.status_code, 200)
            body = applied.json()
            self.assertEqual(body["chat"]["file_ids"], [refund_id])
            self.assertEqual(body["applied_file_ids"], [refund_id])
            self.assertEqual(body["suggestion"]["suggested_file_ids"], [refund_id])

            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()["file_ids"], [refund_id])

            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={
                    "message": "What do refund exceptions require?",
                    "retrieval_mode": "keyword",
                    "context_mode": "rag",
                },
            )

            self.assertEqual(answer.status_code, 200)
            answer_body = answer.json()
            self.assertIn("manager approval", answer_body["message"]["content"].lower())
            self.assertEqual(answer_body["sources"][0]["file_id"], refund_id)

    def test_chat_retrieval_explain_endpoint_uses_saved_context_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            first = client.post(
                "/api/files",
                files={"file": ("chat-explain-a.txt", b"Chat explain refund policy marker A requires receipts.", "text/plain")},
            )
            second = client.post(
                "/api/files",
                files={"file": ("chat-explain-b.txt", b"Chat explain refund policy marker B requires manager approval.", "text/plain")},
            )
            third = client.post(
                "/api/files",
                files={"file": ("chat-explain-c.txt", b"Chat explain refund policy marker C requires finance review.", "text/plain")},
            )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(third.status_code, 200)
            file_ids = [first.json()["id"], second.json()["id"], third.json()["id"]]
            chat = client.post("/api/chats", json={"title": "chat explain api", "file_ids": file_ids})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            defaults = client.put(
                f"/api/chats/{chat_id}/answer-defaults",
                json={"retrieval_mode": "keyword", "top_k": 2, "source_window": 0},
            )
            self.assertEqual(defaults.status_code, 200)
            before = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(before.status_code, 200)
            self.assertEqual(before.json()["messages"], [])

            explained = client.post(
                f"/api/chats/{chat_id}/retrieval/explain",
                json={"message": "chat explain refund policy marker", "candidate_limit": 10},
            )
            self.assertEqual(explained.status_code, 200)
            body = explained.json()
            self.assertEqual(body["query"], "chat explain refund policy marker")
            self.assertEqual(body["diagnostics"]["retrieval_mode"], "keyword")
            self.assertEqual(body["diagnostics"]["selected_count"], 2)
            self.assertEqual(len(body["source_pack"]["sources"]), 2)
            self.assertGreaterEqual(len(body["candidates"]), 3)
            self.assertTrue(any(candidate["selected"] for candidate in body["candidates"]))
            self.assertTrue(any(not candidate["selected"] for candidate in body["candidates"]))

            after = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(after.status_code, 200)
            self.assertEqual(after.json()["messages"], [])

    def test_chat_answer_defaults_are_inherited_and_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={
                    "file": (
                        "defaults-api.txt",
                        (
                            b"Before default context says project iris.\n\n"
                            b"Middle default context contains DEFAULT-313.\n\n"
                            b"After default context says owner support."
                        ),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chat = client.post("/api/chats", json={"title": "defaults api", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            defaults = client.put(
                f"/api/chats/{chat_id}/answer-defaults",
                json={"retrieval_mode": "keyword", "source_window": 1, "top_k": 1, "use_tools": False},
            )
            self.assertEqual(defaults.status_code, 200)
            self.assertEqual(defaults.json()["answer_defaults"]["retrieval_mode"], "keyword")
            self.assertEqual(defaults.json()["answer_defaults"]["source_window"], 1)
            self.assertFalse(defaults.json()["answer_defaults"]["use_tools"])

            inherited = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "What contains DEFAULT 313?"},
            )
            self.assertEqual(inherited.status_code, 200)
            inherited_body = inherited.json()
            self.assertEqual(inherited_body["diagnostics"]["retrieval_mode"], "keyword")
            self.assertEqual(inherited_body["diagnostics"]["source_window"], 1)
            self.assertIn("project iris", inherited_body["sources"][0]["text"])
            self.assertFalse(inherited_body["tool_results"])

            overridden = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Calculate 3 + 4", "use_rag": False, "use_tools": True},
            )
            self.assertEqual(overridden.status_code, 200)
            overridden_body = overridden.json()
            self.assertTrue(overridden_body["tool_results"])
            self.assertIn("7", overridden_body["tool_results"][0]["output"])

            cleared = client.put(f"/api/chats/{chat_id}/answer-defaults", json={"source_window": None})
            self.assertEqual(cleared.status_code, 200)
            self.assertIsNone(cleared.json()["answer_defaults"]["source_window"])

    def test_chat_answer_defaults_can_auto_route_files_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            refund = client.post(
                "/api/files",
                files={"file": ("route-refund-api.txt", b"Refund exceptions require manager approval.", "text/plain")},
            )
            shipping = client.post(
                "/api/files",
                files={"file": ("route-shipping-api.txt", b"Shipping exceptions require dispatch approval.", "text/plain")},
            )
            billing = client.post(
                "/api/files",
                files={"file": ("route-billing-api.txt", b"Billing disputes require invoice review.", "text/plain")},
            )
            self.assertEqual(refund.status_code, 200)
            self.assertEqual(shipping.status_code, 200)
            self.assertEqual(billing.status_code, 200)
            file_ids = [refund.json()["id"], shipping.json()["id"], billing.json()["id"]]
            chat = client.post("/api/chats", json={"title": "route defaults api", "file_ids": file_ids})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            defaults = client.put(
                f"/api/chats/{chat_id}/answer-defaults",
                json={
                    "retrieval_mode": "keyword",
                    "file_selection_mode": "auto",
                    "file_selection_limit": 1,
                    "use_tools": False,
                },
            )
            self.assertEqual(defaults.status_code, 200)
            self.assertEqual(defaults.json()["answer_defaults"]["file_selection_mode"], "auto")
            self.assertEqual(defaults.json()["answer_defaults"]["file_selection_limit"], 1)

            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "How do refund exceptions work?"},
            )

            self.assertEqual(answer.status_code, 200)
            body = answer.json()
            self.assertEqual(body["diagnostics"]["file_selection_mode"], "auto")
            self.assertCountEqual(body["diagnostics"]["candidate_file_ids"], file_ids)
            self.assertEqual(body["diagnostics"]["routed_file_ids"], [refund.json()["id"]])
            self.assertEqual(body["sources"][0]["file_id"], refund.json()["id"])
            self.assertIn("manager approval", body["message"]["content"].lower())

    def test_chat_compact_endpoint_summarizes_older_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            chat = client.post("/api/chats", json={"title": "compact api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]
            for idx in range(3):
                answer = client.post(
                    f"/api/chats/{chat_id}/messages",
                    json={"message": f"Compact API detail {idx}", "use_rag": False},
                )
                self.assertEqual(answer.status_code, 200)

            compacted = client.post(f"/api/chats/{chat_id}/compact", json={"keep_last": 2})
            self.assertEqual(compacted.status_code, 200)
            body = compacted.json()
            self.assertEqual(len(body["messages"]), 2)
            self.assertIn("Compact API detail 0", body["summary"])
            self.assertIn("Compact API detail 2", body["messages"][0]["content"])

            emptied = client.post(f"/api/chats/{chat_id}/compact", json={"keep_last": 0})
            self.assertEqual(emptied.status_code, 200)
            empty_body = emptied.json()
            self.assertEqual(empty_body["messages"], [])
            self.assertIn("Compact API detail 2", empty_body["summary"])

    def test_chat_message_prune_endpoint_removes_later_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            chat = client.post("/api/chats", json={"title": "prune api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]
            for idx in range(3):
                answer = client.post(
                    f"/api/chats/{chat_id}/messages",
                    json={"message": f"Prune API detail {idx}", "use_rag": False},
                )
                self.assertEqual(answer.status_code, 200)
            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            body = saved.json()
            self.assertEqual(len(body["messages"]), 6)
            second_user_id = body["messages"][2]["id"]

            pruned = client.post(f"/api/chats/{chat_id}/messages/{second_user_id}/prune", json={})
            self.assertEqual(pruned.status_code, 200)
            pruned_body = pruned.json()
            self.assertEqual(len(pruned_body["messages"]), 2)
            self.assertNotIn(second_user_id, [message["id"] for message in pruned_body["messages"]])

            kept = client.post(
                f"/api/chats/{chat_id}/messages/{pruned_body['messages'][1]['id']}/prune",
                json={"include_selected": False},
            )
            self.assertEqual(kept.status_code, 200)
            self.assertEqual(len(kept.json()["messages"]), 2)

            missing = client.post(f"/api/chats/{chat_id}/messages/not-real/prune", json={})
            self.assertEqual(missing.status_code, 404)

    def test_chat_answer_preview_endpoint_does_not_persist_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("answer-preview-api.txt", b"The answer preview marker is API-PREVIEW-616.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            skill = client.post(
                "/api/skills",
                json={
                    "name": "Preview API Skill",
                    "instruction": "Mention preview evidence.",
                    "triggers": ["preview"],
                },
            )
            self.assertEqual(skill.status_code, 200)
            chat = client.post("/api/chats", json={"title": "answer preview api", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            preview = client.post(
                f"/api/chats/{chat_id}/messages/preview",
                json={
                    "message": "Remember that my draft color is cyan. What is API PREVIEW 616? Calculate 4 + 5",
                    "retrieval_mode": "keyword",
                    "tool_ids": ["calculator"],
                },
            )
            self.assertEqual(preview.status_code, 200)
            body = preview.json()
            self.assertEqual(body["chat_id"], chat_id)
            self.assertTrue(body["source_pack"]["sources"])
            self.assertIn("API-PREVIEW-616", body["source_pack"]["context_text"])
            self.assertTrue(body["tool_results"])
            self.assertIn("9", body["tool_results"][0]["output"])
            self.assertEqual(body["skills"][0]["id"], skill.json()["id"])
            self.assertTrue(body["prompt_messages"])
            self.assertGreater(body["prompt_chars"], 0)
            self.assertTrue(body["would_learn_memories"])

            saved = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()["messages"], [])
            self.assertEqual(saved.json()["memories"], [])

    def test_chat_message_source_pack_replays_persisted_answer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            upload = client.post(
                "/api/files",
                files={"file": ("message-pack.txt", b"The persisted answer marker is CITED-909.", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            file_id = upload.json()["id"]
            chat = client.post("/api/chats", json={"title": "message pack", "file_ids": [file_id]})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "What is the persisted answer marker?", "retrieval_mode": "keyword"},
            )
            self.assertEqual(answer.status_code, 200)
            assistant_message = answer.json()["message"]
            self.assertTrue(assistant_message["id"])

            pack = client.get(f"/api/chats/{chat_id}/messages/{assistant_message['id']}/source-pack")
            self.assertEqual(pack.status_code, 200)
            body = pack.json()
            self.assertEqual(body["query"], "What is the persisted answer marker?")
            self.assertTrue(body["sources"])
            self.assertIn("CITED-909", body["context_text"])
            self.assertEqual(body["files"][0]["markers"], ["[1]"])

            audit = client.get(f"/api/chats/{chat_id}/messages/{assistant_message['id']}/audit")
            self.assertEqual(audit.status_code, 200)
            audit_body = audit.json()
            self.assertEqual(audit_body["message_id"], assistant_message["id"])
            self.assertTrue(audit_body["answer_supported"])
            self.assertGreaterEqual(audit_body["supported_count"], 1)
            self.assertEqual(audit_body["unsupported_count"], 0)
            self.assertEqual(audit_body["grounding"]["citations"][0]["filename"], "message-pack.txt")

            prompt = client.get(f"/api/chats/{chat_id}/messages/{assistant_message['id']}/prompt")
            self.assertEqual(prompt.status_code, 200)
            prompt_body = prompt.json()
            self.assertEqual(prompt_body["message_id"], assistant_message["id"])
            self.assertGreater(prompt_body["prompt_chars"], 0)
            self.assertEqual(prompt_body["prompt_messages"][-1]["role"], "user")
            self.assertIn("persisted answer marker", prompt_body["prompt_messages"][-1]["content"])
            self.assertIn("CITED-909", "\n".join(item["content"] for item in prompt_body["prompt_messages"]))

            feedback = client.put(
                f"/api/chats/{chat_id}/messages/{assistant_message['id']}/feedback",
                json={"rating": "up", "tags": ["Helpful", " helpful "], "comment": "Good source use."},
            )
            self.assertEqual(feedback.status_code, 200)
            feedback_body = feedback.json()
            saved_feedback = feedback_body["messages"][-1]["feedback"]
            self.assertEqual(saved_feedback["rating"], "up")
            self.assertEqual(saved_feedback["tags"], ["helpful"])
            self.assertEqual(saved_feedback["comment"], "Good source use.")

            listed = client.get("/api/feedback?rating=up&tag=helpful")
            self.assertEqual(listed.status_code, 200)
            listed_body = listed.json()
            self.assertGreaterEqual(listed_body["total_count"], 1)
            self.assertEqual(listed_body["items"][0]["message_id"], assistant_message["id"])
            self.assertEqual(listed_body["items"][0]["question"], "What is the persisted answer marker?")
            self.assertEqual(listed_body["items"][0]["feedback"]["rating"], "up")
            self.assertGreaterEqual(listed_body["items"][0]["source_count"], 1)

            trace = client.get(f"/api/chats/{chat_id}/messages/{assistant_message['id']}/trace")
            self.assertEqual(trace.status_code, 200)
            trace_body = trace.json()
            self.assertEqual(trace_body["message_id"], assistant_message["id"])
            self.assertEqual(trace_body["question"], "What is the persisted answer marker?")
            self.assertEqual(trace_body["answer"]["id"], assistant_message["id"])
            self.assertIn("CITED-909", trace_body["source_pack"]["context_text"])
            self.assertEqual(trace_body["audit"]["message_id"], assistant_message["id"])
            self.assertTrue(trace_body["prompt"]["prompt_messages"])
            self.assertEqual(trace_body["feedback"]["rating"], "up")

            invalid_rating = client.get("/api/feedback?rating=sideways")
            self.assertEqual(invalid_rating.status_code, 400)
            invalid_limit = client.get("/api/feedback?limit=0")
            self.assertEqual(invalid_limit.status_code, 400)

            cleared = client.delete(f"/api/chats/{chat_id}/messages/{assistant_message['id']}/feedback")
            self.assertEqual(cleared.status_code, 200)
            self.assertIsNone(cleared.json()["messages"][-1]["feedback"])

            user_message_id = client.get(f"/api/chats/{chat_id}").json()["messages"][0]["id"]
            user_pack = client.get(f"/api/chats/{chat_id}/messages/{user_message_id}/source-pack")
            self.assertEqual(user_pack.status_code, 400)
            user_audit = client.get(f"/api/chats/{chat_id}/messages/{user_message_id}/audit")
            self.assertEqual(user_audit.status_code, 400)
            user_prompt = client.get(f"/api/chats/{chat_id}/messages/{user_message_id}/prompt")
            self.assertEqual(user_prompt.status_code, 400)
            user_trace = client.get(f"/api/chats/{chat_id}/messages/{user_message_id}/trace")
            self.assertEqual(user_trace.status_code, 400)
            user_feedback = client.put(
                f"/api/chats/{chat_id}/messages/{user_message_id}/feedback",
                json={"rating": "down"},
            )
            self.assertEqual(user_feedback.status_code, 400)

    def test_chat_memory_api_and_auto_learning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            chat = client.post("/api/chats", json={"title": "memory api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]

            manual = client.post(
                f"/api/chats/{chat_id}/memories",
                json={"content": "User's deployment region is Mumbai"},
            )
            self.assertEqual(manual.status_code, 200)
            self.assertIn("Mumbai", manual.json()["memories"][0]["content"])

            learned = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Remember that my backend color is emerald", "use_rag": False},
            )
            self.assertEqual(learned.status_code, 200)

            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "What is my backend color?", "use_rag": False},
            )
            self.assertEqual(answer.status_code, 200)
            self.assertIn("emerald", answer.json()["message"]["content"].lower())

            chat_body = client.get(f"/api/chats/{chat_id}").json()
            self.assertEqual(len(chat_body["memories"]), 2)
            memory_id = chat_body["memories"][0]["id"]
            deleted = client.delete(f"/api/chats/{chat_id}/memories/{memory_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertNotIn(memory_id, [item["id"] for item in deleted.json()["memories"]])

    def test_chat_export_import_endpoint_duplicates_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            chat = client.post("/api/chats", json={"title": "portable api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]
            memory = client.post(
                f"/api/chats/{chat_id}/memories",
                json={"content": "User's export mode is careful"},
            )
            self.assertEqual(memory.status_code, 200)
            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Remember that my export color is indigo", "use_rag": False},
            )
            self.assertEqual(answer.status_code, 200)
            original_assistant_id = answer.json()["message"]["id"]

            exported = client.get(f"/api/chats/{chat_id}/export")
            self.assertEqual(exported.status_code, 200)
            export_body = exported.json()
            self.assertEqual(export_body["version"], "1")
            self.assertEqual(export_body["chat"]["id"], chat_id)
            self.assertTrue(export_body["chat"]["messages"])
            self.assertTrue(export_body["chat"]["memories"])

            imported = client.post(
                "/api/chats/import",
                json={"chat": export_body["chat"], "title": "portable api copy"},
            )
            self.assertEqual(imported.status_code, 200)
            imported_body = imported.json()
            self.assertNotEqual(imported_body["id"], chat_id)
            self.assertEqual(imported_body["title"], "portable api copy")
            self.assertEqual(imported_body["messages"][-1]["content"], export_body["chat"]["messages"][-1]["content"])
            self.assertNotEqual(imported_body["messages"][-1]["id"], original_assistant_id)
            self.assertEqual(imported_body["memories"][0]["content"], export_body["chat"]["memories"][0]["content"])
            self.assertNotEqual(imported_body["memories"][0]["id"], export_body["chat"]["memories"][0]["id"])

            conflict = client.post(
                "/api/chats/import",
                json={"chat": export_body["chat"], "preserve_ids": True},
            )
            self.assertEqual(conflict.status_code, 409)

    def test_skill_api_matches_chat_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APP_DATA_DIR"] = str(Path(tmp) / "data")

            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            skill = client.post(
                "/api/skills",
                json={
                    "name": "Policy Summarizer",
                    "description": "Summarizes policy questions.",
                    "instruction": "Keep policy answers concise and operational.",
                    "triggers": ["policy"],
                    "tool_ids": ["time"],
                },
            )
            self.assertEqual(skill.status_code, 200)
            skill_body = skill.json()
            skill_id = skill_body["id"]
            self.assertEqual(skill_body["tool_ids"], ["time"])

            listed = client.get("/api/skills")
            self.assertEqual(listed.status_code, 200)
            self.assertIn(skill_id, [item["id"] for item in listed.json()])

            chat = client.post("/api/chats", json={"title": "skill api"})
            self.assertEqual(chat.status_code, 200)
            chat_id = chat.json()["id"]
            answer = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Explain this policy", "use_rag": False},
            )
            self.assertEqual(answer.status_code, 200)
            body = answer.json()
            self.assertEqual(body["skills"][0]["id"], skill_id)
            self.assertEqual(body["tool_results"][0]["tool_id"], "time")
            self.assertEqual(body["message"]["skill_ids"], [skill_id])
            self.assertEqual(body["message"]["tool_results"][0]["tool_id"], "time")

            invalid = client.post(
                "/api/skills",
                json={
                    "name": "Bad Skill",
                    "instruction": "Use a missing tool.",
                    "tool_ids": ["missing_tool"],
                },
            )
            self.assertEqual(invalid.status_code, 400)

            invalid_update = client.put(f"/api/skills/{skill_id}", json={"tool_ids": ["missing_tool"]})
            self.assertEqual(invalid_update.status_code, 400)

            updated = client.put(f"/api/skills/{skill_id}", json={"enabled": False, "tool_ids": ["calculator"]})
            self.assertEqual(updated.status_code, 200)
            self.assertFalse(updated.json()["enabled"])
            self.assertEqual(updated.json()["tool_ids"], ["calculator"])
            deleted = client.delete(f"/api/skills/{skill_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertTrue(deleted.json()["deleted"])


if __name__ == "__main__":
    unittest.main()
