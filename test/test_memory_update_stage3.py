"""阶段 3：提取时用相关记忆对比、action=update、observe/apply 两种落地方式。"""

import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx

import database
import main

REAL_ASYNC_CLIENT = httpx.AsyncClient


def row(memory_id, content, layer=1):
    return {"id": memory_id, "content": content, "importance": 5, "layer": layer}


class DedupDecisionTests(unittest.TestCase):
    T = database.DEFAULT_MEMORY_DEDUP_THRESHOLD

    def test_new_content_is_inserted(self):
        decision = database.plan_dedup_write("她养了一只叫团子的橘猫", [row(1, "她在准备考研")], self.T)
        self.assertEqual((decision["op"], decision["supersede"]), ("insert", None))

    def test_richer_fragment_is_still_updated_in_place(self):
        old = row(1, "她喜欢喝咖啡每天早上一杯")
        decision = database.plan_dedup_write("她喜欢喝咖啡每天早上一杯，最爱楼下那家的拿铁", [old], self.T)
        self.assertEqual(decision["op"], "update_in_place")

    def test_richer_than_core_memory_becomes_an_update_instead_of_being_dropped(self):
        old = row(7, "她喜欢喝咖啡每天早上一杯", layer=3)
        decision = database.plan_dedup_write("她喜欢喝咖啡每天早上一杯，最爱楼下那家的拿铁", [old], self.T)
        self.assertEqual(decision["op"], "insert")
        self.assertEqual(decision["supersede"]["id"], 7)

    def test_model_update_wins_over_literal_and_negation_checks(self):
        old = row(5, "她喜欢喝咖啡")
        decision = database.plan_dedup_write("她现在不喜欢喝咖啡了，改喝茶", [old], self.T, target_row=old)
        self.assertEqual((decision["op"], decision["supersede"]["id"]), ("insert", 5))

    def test_model_update_that_duplicates_another_memory_exactly_is_skipped(self):
        target = row(5, "她喜欢喝咖啡")
        other = row(6, "以前喜欢喝咖啡，2026-11 起改喝茶")
        decision = database.plan_dedup_write("以前喜欢喝咖啡，2026-11 起改喝茶", [target, other], self.T,
                                             target_row=target)
        self.assertEqual((decision["op"], decision["matched"]["id"]), ("skip", 6))

    def test_plain_duplicate_is_skipped(self):
        decision = database.plan_dedup_write("她在准备考研", [row(1, "她在准备考研", layer=3)], self.T)
        self.assertEqual(decision["op"], "skip")


class FakeConn:
    def __init__(self):
        self.executed = []
        self.proposals = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "UPDATE 1"

    async def fetchval(self, sql, *args):
        self.proposals.append(args)
        return 31


class SupersedeModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_observe_only_records_a_proposal(self):
        conn = FakeConn()
        with redirect_stdout(io.StringIO()):
            result = await database._record_supersede(conn, 20, {"id": 5, "content": "旧"}, "新", "observe", "s1")
        self.assertEqual(result, {"supersedes": 5, "update_mode": "observe", "proposal_id": 31})
        self.assertEqual(conn.executed, [])
        self.assertEqual(conn.proposals, [(20, 5, "新", "旧", "s1")])

    async def test_apply_archives_old_and_links_it(self):
        conn = FakeConn()
        with redirect_stdout(io.StringIO()):
            result = await database._record_supersede(conn, 20, {"id": 5, "content": "旧"}, "新", "apply", "s1")
        self.assertEqual(result["update_mode"], "apply")
        sql, args = conn.executed[0]
        self.assertIn("is_active = FALSE, superseded_by = $1", sql)
        self.assertEqual(args, (20, 5))
        self.assertEqual(conn.proposals, [])

    def test_mode_defaults_to_observe(self):
        with patch.object(database, "MEMORY_UPDATE_MODE", "weird"):
            self.assertEqual(database.current_update_mode(), "observe")
        with patch.object(database, "MEMORY_UPDATE_MODE", "apply"):
            self.assertEqual(database.current_update_mode(), "apply")


class ExtractionContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_related_plus_core_deduped(self):
        search = AsyncMock(return_value=[{"id": 1, "content": "a"}, {"id": 2, "content": "b"}])
        core = AsyncMock(return_value=[{"id": 2, "content": "b"}, {"id": 9, "content": "core"}])
        with patch.object(main, "search_memories", search), \
                patch.object(main, "get_core_memories", core), \
                patch.object(main, "MEMORY_EXTRACT_CONTEXT_LIMIT", 30):
            result = await main._extraction_context_memories([
                {"role": "user", "content": "我改喝茶了"}, {"role": "assistant", "content": "好"}])
        self.assertEqual([m["id"] for m in result], [1, 2, 9])
        search.assert_awaited_once_with("我改喝茶了\n好", limit=30)
        core.assert_awaited_once_with(15)

    async def test_update_target_must_have_been_shown(self):
        saved = []

        async def save(**kwargs):
            saved.append(kwargs)
            return {"action": "inserted", "memory_id": 50}

        extracted = [
            {"content": "以前喜欢咖啡，2026-11 起改喝茶", "importance": 5, "source_ids": [10],
             "kind": "fact", "event_date": None, "action": "update", "target_id": 1},
            {"content": "瞎指的更新", "importance": 5, "source_ids": [10],
             "kind": "fact", "event_date": None, "action": "update", "target_id": 999},
        ]
        with (
            patch.object(main, "search_memories", AsyncMock(return_value=[
                {"id": 1, "content": "喜欢喝咖啡", "created_at": datetime(2026, 10, 2)}])),
            patch.object(main, "get_core_memories", AsyncMock(return_value=[])),
            patch.object(main, "extract_memories", AsyncMock(return_value=extracted)),
            patch.object(main, "save_memory_deduplicated", save),
            patch.object(main, "get_active_memories_count", AsyncMock(return_value=2)),
            patch.object(main, "MEMORY_EXTRACT_INTERVAL", 1),
            redirect_stdout(io.StringIO()),
        ):
            await main._extract_and_store_memories(
                "s1", "我改喝茶了", "好", [{"role": "user", "content": "我改喝茶了", "source_id": 10}],
                10, 11, None,
            )
        self.assertEqual([item["supersede_target_id"] for item in saved], [1, None])


class ProposalApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = REAL_ASYNC_CLIENT(transport=httpx.ASGITransport(app=main.app), base_url="http://test")
        self.headers = {"X-Gateway-Key": "k" * 40}
        self.patch = patch.object(main, "GATEWAY_SECRET", "k" * 40)
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.client.aclose()

    async def test_list_and_resolve(self):
        items = [{"id": 3, "created_at": datetime(2026, 11, 18, 15, 0), "resolved_at": None,
                  "new_memory_id": 20, "old_memory_id": 5, "new_content": "新", "old_content": "旧",
                  "source": "s1", "status": "pending", "new_active": True, "old_active": True,
                  "old_superseded_by": None}]
        with patch.object(main, "list_update_proposals", AsyncMock(return_value=(items, 1))):
            data = (await self.client.get("/api/memories/update-proposals", headers=self.headers)).json()
        self.assertEqual(data["pending"], 1)
        self.assertEqual(data["items"][0]["created_at"], "2026-11-18 23:00")

        resolve = AsyncMock(return_value={"status": "accepted", "archived": 1})
        with patch.object(main, "resolve_update_proposal", resolve):
            response = await self.client.post("/api/memories/update-proposals/3/accept", headers=self.headers)
        self.assertEqual(response.json()["archived"], 1)
        resolve.assert_awaited_once_with(3, accept=True)

        with patch.object(main, "resolve_update_proposal",
                          AsyncMock(side_effect=database.ProposalConflict("这条提议已经处理过了"))):
            response = await self.client.post("/api/memories/update-proposals/3/ignore", headers=self.headers)
        self.assertEqual(response.status_code, 409)


class McpUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_memory_tool_goes_through_update_mode(self):
        save = AsyncMock(return_value={"action": "inserted", "memory_id": 21, "update_mode": "observe",
                                       "supersedes": 5, "proposal_id": 2})
        target = {"id": 5, "content": "喜欢喝咖啡", "importance": 6, "is_active": True,
                  "superseded_by": None, "kind": "fact", "event_date": None}
        with patch.object(main, "MEMORY_ENABLED", True), \
                patch.object(main, "get_memory_by_id", AsyncMock(return_value=target)), \
                patch.object(main, "save_memory_deduplicated", save), \
                redirect_stdout(io.StringIO()):
            text = await main._mcp_update_memory(5, "以前喜欢咖啡，2026-11 起改喝茶")
        self.assertEqual(save.await_args.kwargs["supersede_target_id"], 5)
        self.assertEqual(save.await_args.kwargs["source_session"], "mcp")
        self.assertIn("提议", text)


if __name__ == "__main__":
    unittest.main()
