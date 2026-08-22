import sys
import types
import unittest

import httpx


async def async_placeholder(*args, **kwargs):
    return None


database_stub = types.ModuleType("database")
database_names = """
_parse_backup_datetime import_memories_v2 repair_broken_merge_references
search_memories_with_mode init_tables close_pool save_message search_memories
save_memory get_all_memories_count get_recent_memories get_all_memories get_pool
get_all_memories_detail delete_archived_memory delete_archived_memories_batch
soft_delete_memories_batch restore_archived_memories_batch get_gateway_config
set_gateway_config get_all_gateway_config get_conversation_messages
get_session_cache_state save_session_cache_state delete_session_cache_state
save_token_usage ensure_token_usage_table get_conversations_paginated
delete_conversation batch_delete_conversations merge_sessions_to_target
list_all_session_cache_states export_all_conversations import_conversations
get_last_user_content update_last_assistant_message db_row_to_message
backfill_memory_embeddings get_pending_memory_embedding_count search_conversations
update_message_content delete_single_message rename_session_id get_fragments_by_date
create_consolidated_events promote_to_core merge_memories check_duplicate_memory
update_memory_with_layer get_layer_statistics cleanup_old_fragments revert_merge
search_chat_fragments rebuild_content_tsv kick_embedding_backfill
get_embedding_backfill_status mark_fragments_seen
""".split()
for name in database_names:
    setattr(database_stub, name, async_placeholder)


class BrokenMergeReferencesError(Exception):
    pass


database_stub.BrokenMergeReferencesError = BrokenMergeReferencesError
sys.modules["database"] = database_stub

memory_stub = types.ModuleType("memory_extractor")
memory_stub.extract_memories = async_placeholder
memory_stub.score_memories = async_placeholder
sys.modules["memory_extractor"] = memory_stub

import main


class DylanDiaryEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_initializes_diary_table_between_core_tables(self):
        calls = []

        async def record(name, result=None):
            calls.append(name)
            return result

        originals = {
            name: getattr(main, name)
            for name in (
                "init_tables",
                "ensure_dylan_diary_table",
                "ensure_token_usage_table",
                "get_all_gateway_config",
                "get_gateway_config",
                "close_pool",
            )
        }
        main.init_tables = lambda: record("core")
        main.ensure_dylan_diary_table = lambda: record("diary")
        main.ensure_token_usage_table = lambda: record("tokens")
        main.get_all_gateway_config = lambda: record("config", {})
        main.get_gateway_config = lambda *args: record("session", "")
        main.close_pool = lambda: record("close")

        try:
            async with main.lifespan(main.app):
                pass
        finally:
            for name, value in originals.items():
                setattr(main, name, value)

        self.assertEqual(calls[:3], ["core", "diary", "tokens"])

    async def asyncSetUp(self):
        self.assertTrue(hasattr(main, "write_dylan_diary"), "diary endpoint must exist")
        self.original_secret = main.GATEWAY_SECRET
        self.original_save = main.save_dylan_diary
        main.GATEWAY_SECRET = "gateway-secret"
        self.transport = httpx.ASGITransport(app=main.app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        if hasattr(self, "client"):
            await self.client.aclose()
        if hasattr(self, "original_secret"):
            main.GATEWAY_SECRET = self.original_secret
        if hasattr(self, "original_save"):
            main.save_dylan_diary = self.original_save

    async def test_requires_existing_gateway_key(self):
        response = await self.client.post("/internal/dylan-diary", json={"content": "entry"})
        self.assertEqual(response.status_code, 401)

    async def test_validates_body_and_metadata(self):
        headers = {"X-Gateway-Key": "gateway-secret"}
        invalid_json = await self.client.post(
            "/internal/dylan-diary", content="{", headers=headers
        )
        blank = await self.client.post(
            "/internal/dylan-diary", json={"content": " "}, headers=headers
        )
        invalid_metadata = await self.client.post(
            "/internal/dylan-diary",
            json={"content": "entry", "metadata": []},
            headers=headers,
        )
        self.assertEqual(invalid_json.status_code, 400)
        self.assertEqual(blank.status_code, 400)
        self.assertEqual(invalid_metadata.status_code, 400)

    async def test_returns_created_id(self):
        async def fake_save(content, metadata):
            self.assertEqual(content, "entry")
            self.assertEqual(metadata, {"source": "test"})
            return 42

        main.save_dylan_diary = fake_save
        response = await self.client.post(
            "/internal/dylan-diary",
            json={"content": " entry ", "metadata": {"source": "test"}},
            headers={"X-Gateway-Key": "gateway-secret"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"ok": True, "id": 42})

    async def test_returns_500_without_leaking_exception(self):
        async def failing_save(content, metadata):
            raise RuntimeError("database secret")

        main.save_dylan_diary = failing_save
        response = await self.client.post(
            "/internal/dylan-diary",
            json={"content": "entry"},
            headers={"X-Gateway-Key": "gateway-secret"},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "Failed to persist diary."})
