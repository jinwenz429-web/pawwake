"""阶段 7：修补旧记忆（相对时间 / 补 kind），预览 → 应用 → 撤回。"""

import json
import unittest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx

import database
import main
import memory_fix

REAL_ASYNC_CLIENT = httpx.AsyncClient
CREATED = datetime(2026, 11, 18, 4, 0, tzinfo=timezone.utc)  # 东八区 2026-11-18 周三 12:00


def mem(memory_id, content, **extra):
    return {"id": memory_id, "content": content, "created_at": CREATED, "kind": None,
            "event_date": None, "valid_until": None, "is_active": True, **extra}


class ScanTests(unittest.TestCase):
    def test_relative_time_detection(self):
        for text in ("昨天和妈妈吵架", "下周三考试", "最近在赶论文", "今晚想吃火锅"):
            self.assertTrue(memory_fix.has_relative_time(text), text)
        for text in ("2026-11-17 和妈妈吵架", "喜欢喝茶"):
            self.assertFalse(memory_fix.has_relative_time(text), text)

    def test_plan_items_keep_only_real_changes_and_never_overwrite(self):
        sources = [mem(1, "昨天和妈妈吵架"), mem(2, "最近在赶论文", kind="state"),
                   mem(3, "下周考试", kind="plan", event_date=date(2026, 11, 25))]
        items = memory_fix.build_plan_items("time", sources, [
            {"id": 1, "content": "2026-11-17 和妈妈吵架", "kind": "event", "event_date": "2026-11-17"},
            {"id": 2, "content": "截至 2026-11-18 在赶论文", "kind": "fact", "event_date": None},
            {"id": 3, "content": "下周考试", "kind": "plan", "event_date": "2026-12-01"},
            {"id": 99, "content": "不存在"},
        ])
        self.assertEqual([item["id"] for item in items], [1, 2])
        first, second = items
        self.assertEqual((first["new_content"], first["new_kind"], first["new_event_date"]),
                         ("2026-11-17 和妈妈吵架", "event", "2026-11-17"))
        self.assertEqual(first["recorded_at"], "2026-11-18 周三 12:00")
        self.assertEqual(second["new_kind"], "state")  # 已有的 kind 不覆盖

    def test_kind_mode_does_not_touch_content(self):
        items = memory_fix.build_plan_items("kind", [mem(1, "喜欢喝茶")], [
            {"id": 1, "content": "被模型改了", "kind": "fact"}])
        self.assertEqual((items[0]["new_content"], items[0]["new_kind"]), ("喜欢喝茶", "fact"))


class FakeConn:
    def __init__(self, items, memories):
        self.plan = {"id": 1, "status": "preview", "items": json.dumps(items)}
        self.memories = {m["id"]: dict(m) for m in memories}
        self.log = []

    def transaction(self):
        conn = self

        class Tx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *args):
                return False
        return Tx()

    async def fetchrow(self, sql, *args):
        if "memory_fix_plans" in sql:
            return self.plan
        row = self.memories.get(args[0])
        return dict(row) if row else None  # asyncpg 返回的是快照

    async def fetch(self, sql, *args):
        return [dict(entry, id=index + 1) for index, entry in enumerate(self.log)
                if entry["reverted_at"] is None]

    async def fetchval(self, sql, *args):
        return self.memories[args[0]]["content"]

    async def execute(self, sql, *args):
        if sql.lstrip().startswith("UPDATE memories SET content"):
            self.memories[args[0]].update(content=args[1], kind=args[2], event_date=args[3],
                                          valid_until=args[4])
        elif "INSERT INTO memory_edit_log" in sql:
            keys = ("plan_id", "memory_id", "old_content", "new_content", "old_kind", "new_kind",
                    "old_event_date", "new_event_date", "old_valid_until", "new_valid_until")
            self.log.append(dict(zip(keys, args), reverted_at=None))
        elif "UPDATE memory_edit_log" in sql:
            self.log[args[0] - 1]["reverted_at"] = "now"
        elif "status = 'applied'" in sql:
            self.plan["status"] = "applied"
        elif "status = 'reverted'" in sql:
            self.plan["status"] = "reverted"
        return "UPDATE 1"


class ApplyRevertTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        sources = [mem(1, "昨天和妈妈吵架"), mem(2, "最近在赶论文"), mem(3, "下周三考试")]
        self.items = memory_fix.build_plan_items("time", sources, [
            {"id": 1, "content": "2026-11-17 和妈妈吵架", "kind": "event", "event_date": "2026-11-17"},
            {"id": 2, "content": "截至 2026-11-18 在赶论文", "kind": "state"},
            {"id": 3, "content": "2026-11-25 考试", "kind": "plan", "event_date": "2026-11-25"},
        ])
        self.conn = FakeConn(self.items, sources)
        self.embedded = []

    def patches(self):
        conn = self.conn

        class Pool:
            def acquire(self):
                class Ctx:
                    async def __aenter__(self_inner):
                        return conn

                    async def __aexit__(self_inner, *args):
                        return False
                return Ctx()

        async def save_embedding(c, memory_id, embedding):
            self.embedded.append(memory_id)

        return [
            patch.object(memory_fix, "ensure_fix_tables", AsyncMock()),
            patch.object(memory_fix, "get_pool", AsyncMock(return_value=Pool())),
            patch.object(database, "MEMORY_VECTOR_ENABLED", True),
            patch.object(database, "MEMORY_STATE_TTL_DAYS", 14),
            patch.object(database, "compute_embedding", AsyncMock(return_value=[0.1])),
            patch.object(database, "save_memory_embedding", save_embedding),
        ]

    async def run_with(self, coro_factory):
        managers = self.patches()
        for manager in managers:
            manager.start()
        try:
            return await coro_factory()
        finally:
            for manager in reversed(managers):
                manager.stop()

    async def test_apply_respects_exclusions_and_skips_changed_rows(self):
        self.conn.memories[2]["content"] = "预览后被手动改过"
        result = await self.run_with(lambda: memory_fix.apply_fix_plan(1, 1, excluded_ids=[3]))
        self.assertEqual((result["applied"], result["excluded"], result["skipped_changed"]), (1, 1, 1))
        first = self.conn.memories[1]
        self.assertEqual((first["content"], first["kind"], first["event_date"]),
                         ("2026-11-17 和妈妈吵架", "event", date(2026, 11, 17)))
        self.assertEqual(self.conn.memories[3]["content"], "下周三考试")
        self.assertEqual(self.conn.log[0]["old_content"], "昨天和妈妈吵架")
        self.assertEqual(self.embedded, [1])

    async def test_state_and_plan_get_valid_until(self):
        await self.run_with(lambda: memory_fix.apply_fix_plan(1, 1))
        self.assertEqual(self.conn.memories[2]["valid_until"], date(2026, 12, 2))
        self.assertEqual(self.conn.memories[3]["valid_until"], date(2026, 11, 28))

    async def test_apply_twice_or_wrong_confirmation_is_refused(self):
        with self.assertRaises(ValueError):
            await self.run_with(lambda: memory_fix.apply_fix_plan(1, 2))
        await self.run_with(lambda: memory_fix.apply_fix_plan(1, 1))
        with self.assertRaises(memory_fix.FixConflict):
            await self.run_with(lambda: memory_fix.apply_fix_plan(1, 1))

    async def test_revert_restores_untouched_rows_only(self):
        await self.run_with(lambda: memory_fix.apply_fix_plan(1, 1))
        self.conn.memories[3]["content"] = "应用后又手动改了"
        self.embedded.clear()
        result = await self.run_with(lambda: memory_fix.revert_fix_plan(1))
        self.assertEqual((result["reverted"], result["skipped_changed"]), (2, 1))
        self.assertEqual(self.conn.memories[1]["content"], "昨天和妈妈吵架")
        self.assertIsNone(self.conn.memories[1]["kind"])
        self.assertEqual(self.conn.memories[3]["content"], "应用后又手动改了")
        self.assertEqual(sorted(self.embedded), [1, 2])
        self.assertEqual(self.conn.plan["status"], "reverted")


class FixEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_apply_needs_the_apply_token(self):
        client = REAL_ASYNC_CLIENT(transport=httpx.ASGITransport(app=main.app), base_url="http://test")
        headers = {"X-Gateway-Key": "g" * 40}
        try:
            with patch.object(main, "GATEWAY_SECRET", "g" * 40), \
                    patch.dict("os.environ", {"MEMORY_REBUILD_APPLY_TOKEN": "t" * 32}):
                response = await client.post("/api/memories/fix/plan/1/apply", headers=headers,
                                             json={"confirmed_plan_id": 1})
                self.assertEqual(response.status_code, 403)
                apply = AsyncMock(return_value={"status": "applied", "applied": 2})
                with patch.object(memory_fix, "apply_fix_plan", apply):
                    response = await client.post(
                        "/api/memories/fix/plan/1/apply",
                        headers={**headers, "X-Memory-Rebuild-Apply-Key": "t" * 32},
                        json={"confirmed_plan_id": 1, "excluded_ids": [4, "5"]},
                    )
                self.assertEqual(response.json()["applied"], 2)
                apply.assert_awaited_once_with(1, 1, [4, 5])
                response = await client.post("/api/memories/fix/preview", headers=headers,
                                             json={"mode": "bogus"})
                self.assertEqual(response.status_code, 400)
        finally:
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
