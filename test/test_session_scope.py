"""对话线分区：#剧情 / #工具 标记、召回范围、剧情记忆不进主池。"""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

import database
import main
import session_scope

TEST_URL = os.getenv("PAWWAKE_TEST_DATABASE_URL", "")


class MarkerTests(unittest.TestCase):
    def test_story_marker_with_name(self):
        self.assertEqual(
            session_scope.parse_marker("#剧情 猫妖篇\n今天我们从山里开始"),
            ("story", "猫妖篇", "今天我们从山里开始"),
        )

    def test_full_width_hash_and_no_name(self):
        self.assertEqual(session_scope.parse_marker("＃剧情\n开始"), ("story", None, "开始"))

    def test_tool_marker(self):
        self.assertEqual(session_scope.parse_marker("#工具 帮我改代码"), ("tool", "帮我改代码", ""))

    def test_marker_only_counts_at_the_start(self):
        self.assertIsNone(session_scope.parse_marker("我说 #剧情 这个词"))
        self.assertIsNone(session_scope.parse_marker("#剧情感觉怪怪的"[:0] + "今天好累"))

    def test_strip_from_multipart_content(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "#剧情 吵架篇\n你走开"},
                                           {"type": "image_url", "image_url": {"url": "x"}}]}
        parsed = session_scope.strip_marker_from_message(msg)
        self.assertEqual(parsed[:2], ("story", "吵架篇"))
        self.assertEqual(msg["content"][0]["text"], "你走开")


class ScopeSqlTests(unittest.TestCase):
    def test_default_is_main_pool_only(self):
        self.assertEqual(session_scope.scope_sql(), " AND scope_session IS NULL")

    def test_story_line_sees_main_plus_itself(self):
        token = session_scope.CURRENT_SCOPE.set("abc-123")
        try:
            self.assertEqual(
                session_scope.scope_sql(),
                " AND (scope_session IS NULL OR scope_session = 'abc-123')",
            )
        finally:
            session_scope.CURRENT_SCOPE.reset(token)

    def test_all_has_no_filter(self):
        token = session_scope.CURRENT_SCOPE.set(session_scope.ALL)
        try:
            self.assertEqual(session_scope.scope_sql(), "")
        finally:
            session_scope.CURRENT_SCOPE.reset(token)

    def test_unsafe_session_id_falls_back_to_main(self):
        token = session_scope.CURRENT_SCOPE.set("x' OR 1=1 --")
        try:
            self.assertEqual(session_scope.scope_sql(), " AND scope_session IS NULL")
        finally:
            session_scope.CURRENT_SCOPE.reset(token)


class ToolLineExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_line_skips_extraction(self):
        with patch.object(main, "get_session_tag", AsyncMock(return_value={"session_id": "s", "kind": "tool"})), \
             patch.object(main, "extract_memories", AsyncMock()) as extract, \
             redirect_stdout(io.StringIO()):
            await main._extract_and_store_memories("s", "u", "a", [], None, None, None)
        extract.assert_not_called()


class McpLabelTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_sees_all_lines_and_labels_story_memories(self):
        seen_scopes = []

        async def fake_search(query, limit=10):
            seen_scopes.append(session_scope.CURRENT_SCOPE.get())
            return [{"id": 1, "content": "她是白猫", "created_at": None, "event_date": None},
                    {"id": 2, "content": "她在考研", "created_at": None, "event_date": None}]

        with patch.object(main, "MEMORY_ENABLED", True), \
             patch.object(main, "search_memories", fake_search), \
             patch.object(main, "get_memory_scope_labels", AsyncMock(return_value={1: "猫妖篇"})), \
             patch.object(main, "_format_memory_line", lambda mem, today: mem["content"]):
            text = await main._mcp_search_memory("猫", 5)
        self.assertIs(seen_scopes[0], session_scope.ALL)
        self.assertIsNone(session_scope.CURRENT_SCOPE.get())
        first, second = text.splitlines()
        self.assertIn("剧情《猫妖篇》", first)
        self.assertNotIn("剧情", second)


@unittest.skipUnless(TEST_URL, "需要 PAWWAKE_TEST_DATABASE_URL")
class RealDatabaseScopeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            patch.object(database, "DATABASE_URL", TEST_URL),
            patch.object(database, "_pool", None),
            patch.object(database, "MEMORY_VECTOR_ENABLED", False),
            patch.object(database, "CONVERSATION_RECALL_ENABLED", False),
            patch.object(database, "MIN_SCORE_THRESHOLD", 0),
        ]
        for p in self.patches:
            p.start()
        database._session_tag_cache.clear()
        database._session_tag_cache_loaded = False
        with redirect_stdout(io.StringIO()):
            pool = await database.get_pool()
            async with pool.acquire() as conn:
                await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            await database.init_tables()
            await database.init_tables()  # 迁移要能重复跑
        self.pool = pool

    async def asyncTearDown(self):
        await database.close_pool()
        for p in reversed(self.patches):
            p.stop()

    async def _insert(self, content, session, layer=1):
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "INSERT INTO memories (content, importance, source_session, layer) VALUES ($1, 8, $2, $3) RETURNING id",
                content, session, layer,
            )

    async def _search(self, query, scope=None):
        token = session_scope.CURRENT_SCOPE.set(scope)
        try:
            with redirect_stdout(io.StringIO()):
                rows = await database.search_memories(query, limit=20)
        finally:
            session_scope.CURRENT_SCOPE.reset(token)
        return {r["content"] for r in rows}

    async def test_story_memories_stay_in_their_line(self):
        with redirect_stdout(io.StringIO()):
            await database.set_session_tag("cat", "story", "猫妖篇", source="marker")
        await self._insert("白猫锦锦怕冷", "cat")
        await self._insert("锦锦在考研怕冷", "main-1")

        self.assertEqual(await self._search("怕冷"), {"锦锦在考研怕冷"})
        self.assertEqual(await self._search("怕冷", "cat"), {"白猫锦锦怕冷", "锦锦在考研怕冷"})
        self.assertEqual(await self._search("怕冷", "other"), {"锦锦在考研怕冷"})
        self.assertEqual(await self._search("怕冷", session_scope.ALL), {"白猫锦锦怕冷", "锦锦在考研怕冷"})

        labels = await database.get_memory_scope_labels(
            [r for r in [await self._find("白猫锦锦怕冷"), await self._find("锦锦在考研怕冷")]]
        )
        self.assertEqual(list(labels.values()), ["猫妖篇"])

    async def _find(self, content):
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT id FROM memories WHERE content = $1", content)

    async def test_retagging_moves_existing_memories_both_ways(self):
        await self._insert("吵架时说的狠话", "fight")
        self.assertEqual(await self._search("狠话"), {"吵架时说的狠话"})

        with redirect_stdout(io.StringIO()):
            result = await database.set_session_tag("fight", "story", "吵架篇")
        self.assertEqual(result["moved"], 1)
        self.assertEqual(await self._search("狠话"), set())

        with redirect_stdout(io.StringIO()):
            result = await database.set_session_tag("fight", "main")
        self.assertEqual(result["moved"], 1)
        self.assertEqual(await self._search("狠话"), {"吵架时说的狠话"})
        self.assertIsNone(await database.get_session_tag("fight"))

    async def test_core_memories_main_first_and_story_only_in_its_line(self):
        with redirect_stdout(io.StringIO()):
            await database.set_session_tag("cat", "story", "猫妖篇")
        await self._insert("剧情核心", "cat", layer=3)
        await self._insert("主线核心", "m", layer=3)

        main_core = [m["content"] for m in await database.get_core_memories(8)]
        self.assertEqual(main_core, ["主线核心"])
        token = session_scope.CURRENT_SCOPE.set("cat")
        try:
            story_core = [m["content"] for m in await database.get_core_memories(8)]
        finally:
            session_scope.CURRENT_SCOPE.reset(token)
        self.assertEqual(story_core, ["主线核心", "剧情核心"])


if __name__ == "__main__":
    unittest.main()


class MarkerChatFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_marker_is_stripped_and_tags_the_line(self):
        import json as _json
        import httpx
        from contextlib import ExitStack
        from test.test_kelivo_direct import FakeRequest, REAL_ASYNC_CLIENT, completion, settle

        saved, upstream_bodies = [], []

        async def save_message(session_id, role, content, model, metadata=None):
            saved.append((role, content))
            return len(saved)

        def upstream(request):
            upstream_bodies.append(_json.loads(request.content))
            return httpx.Response(200, json=completion("喵"))

        set_tag = AsyncMock(return_value={"kind": "story", "name": "猫妖篇", "moved": 0})
        stack = ExitStack()
        for name, value in {"CACHE_PARTITION_ENABLED": False, "MEMORY_ENABLED": False,
                            "MEMORY_EXTRACT_ENABLED": False, "FORCE_STREAM": False,
                            "REASONING_EFFORT": ""}.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="")))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", save_message))
        stack.enter_context(patch.object(main, "process_memories_background", AsyncMock()))
        stack.enter_context(patch.object(main, "recent_quarantine_fingerprints", AsyncMock(return_value=set())))
        stack.enter_context(patch.object(main, "set_session_tag", set_tag))
        stack.enter_context(patch.object(main, "get_session_tag",
                                         AsyncMock(return_value={"session_id": "conv-1", "kind": "story", "name": "猫妖篇"})))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kw: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(upstream), **kw)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        with stack:
            response = await main._chat_completions_inner(FakeRequest([
                {"role": "user", "content": "#剧情 猫妖篇\n你是黑豹，我是白猫"},
            ]))
            await settle()
        self.assertEqual(response.status_code, 200)
        set_tag.assert_awaited_once_with("conv-1", "story", "猫妖篇", source="marker")
        sent = upstream_bodies[0]["messages"][-1]["content"]
        self.assertEqual(sent, "你是黑豹，我是白猫")
        self.assertNotIn("#剧情", "".join(c for _, c in saved))
