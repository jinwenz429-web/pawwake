"""阶段 5：召回搜索词、核心记忆常驻位置、被取代记忆不召回、备份 v3。"""

import io
import unittest
from contextlib import asynccontextmanager, redirect_stdout
from datetime import datetime
from unittest.mock import AsyncMock, patch

import database
import main


def _text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


class RecallQueryTests(unittest.TestCase):
    HISTORY = [
        {"role": "user", "content": "上周说的那家火锅店"},
        {"role": "assistant", "content": "老码头那家？"},
        {"role": "user", "content": "我想周末约妈妈去吃"},
        {"role": "assistant", "content": "好呀，" + "妈" * 300},
    ]

    def test_uses_last_rounds_and_previous_reply(self):
        query = main.build_recall_query(self.HISTORY + [{"role": "user", "content": "你觉得她会喜欢吗"}], 2)
        lines = query.split("\n")
        self.assertEqual(lines[0], "我想周末约妈妈去吃")
        self.assertTrue(lines[1].startswith("好呀，"))
        self.assertEqual(len(lines[1]), 200)
        self.assertEqual(lines[2], "你觉得她会喜欢吗")
        self.assertNotIn("火锅", query)

    def test_short_latest_message_pulls_one_more_round(self):
        query = main.build_recall_query(self.HISTORY + [{"role": "user", "content": "嗯嗯！"}], 2)
        self.assertIn("火锅", query)
        self.assertIn("周末约妈妈", query)

    def test_multimodal_and_tool_messages(self):
        query = main.build_recall_query([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}},
                                         {"type": "text", "text": "看看这只猫"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
            {"role": "tool", "content": "{\"results\": []}"},
            {"role": "assistant", "content": "好可爱的橘猫"},
        ], 2)
        self.assertEqual(query, "看看这只猫\n好可爱的橘猫")


class SupersededFilterTests(unittest.TestCase):
    def test_all_memory_search_paths_skip_superseded(self):
        import inspect
        hybrid = inspect.getsource(database.search_memories_hybrid)
        keyword = inspect.getsource(database.search_memories)
        core = inspect.getsource(database.get_core_memories)
        self.assertEqual(hybrid.count("superseded_by IS NULL"), 3)
        self.assertIn("superseded_by IS NULL", keyword)
        self.assertIn("superseded_by IS NULL", core)


class PartitionCorePlacementTests(unittest.IsolatedAsyncioTestCase):
    async def test_core_block_is_in_cached_system_and_retrieval_in_user(self):
        core = [{"id": 1, "content": "她生日是 3 月 4 日", "importance": 10, "layer": 3,
                 "event_date": None, "created_at": datetime(2026, 1, 1), "kind": "fact"}]
        search = AsyncMock(return_value=[
            {"id": 1, "content": "她生日是 3 月 4 日", "importance": 10, "event_date": None,
             "created_at": datetime(2026, 1, 1)},
            {"id": 2, "content": "最近在学吉他", "importance": 6, "event_date": None,
             "created_at": datetime(2026, 11, 1)},
        ])
        state = {"summary_parts": [], "a_start_round": 0}
        with (
            patch.object(main, "search_memories", search),
            patch.object(main, "get_core_memories", AsyncMock(return_value=core)),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "CACHE_PARTITION_X", 15),
            patch.object(main, "get_session_cache_state", AsyncMock(return_value=state)),
            patch.object(main, "compact_summary_parts", AsyncMock(return_value=None)),
            redirect_stdout(io.StringIO()),
        ):
            core_memories = await main._load_core_memories()
            prompt = "persona" + main.MEMORY_USAGE_GUIDE + "\n\n" + main.format_core_memory_block(core_memories)
            messages = await main.build_partitioned_messages(
                "s1",
                [{"role": "user", "content": "教我弹吉他"}],
                prompt,
                "教我弹吉他",
                memory_exclude_ids=[mem["id"] for mem in core_memories],
            )
        system = messages[0]
        self.assertIn("<core_memories>", _text(system))
        self.assertIn("cache_control", system["content"][0])
        user = _text(messages[-1])
        self.assertIn("最近在学吉他", user)
        self.assertNotIn("生日", user)


class FakeImportConn:
    def __init__(self):
        self.rows = {}
        self.next_id = 500

    @asynccontextmanager
    async def transaction(self):
        yield

    async def fetch(self, sql, *args):
        return [{"id": rid} for rid, row in self.rows.items() if row["content"] == args[0]]

    async def fetchrow(self, sql, *args):
        self.next_id += 1
        self.rows[self.next_id] = {
            "content": args[0], "kind": args[8], "valid_until": args[9],
            "merged_from": None, "superseded_by": None,
        }
        return {"id": self.next_id}

    async def execute(self, sql, *args):
        if "merged_from" in sql:
            self.rows[args[1]]["merged_from"] = args[0]
        elif "superseded_by" in sql:
            self.rows[args[1]]["superseded_by"] = args[0]
        return "UPDATE 1"


class BackupV3ImportTests(unittest.IsolatedAsyncioTestCase):
    async def _import(self, memories, version):
        conn = FakeImportConn()

        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield conn

        with (
            patch.object(database, "get_pool", AsyncMock(return_value=Pool())),
            patch.object(database, "get_all_memories_count", AsyncMock(return_value=len(memories))),
            patch.object(database, "MEMORY_VECTOR_ENABLED", False),
        ):
            result = await database.import_memories_v2(memories, schema_version=version)
        return conn, result

    async def test_v3_restores_kind_valid_until_and_superseded_by(self):
        conn, result = await self._import([
            {"backup_id": 1, "content": "以前喝咖啡", "kind": "fact", "superseded_by": 2,
             "is_active": False},
            {"backup_id": 2, "content": "2026-11 起改喝茶", "kind": "fact"},
            {"backup_id": 3, "content": "在赶论文", "kind": "state", "valid_until": "2026-12-02"},
            {"backup_id": 4, "content": "奇怪的 kind", "kind": "nonsense"},
        ], 3)
        by_content = {row["content"]: (rid, row) for rid, row in conn.rows.items()}
        old_id, old = by_content["以前喝咖啡"]
        new_id, _ = by_content["2026-11 起改喝茶"]
        self.assertEqual(old["superseded_by"], new_id)
        self.assertEqual(str(by_content["在赶论文"][1]["valid_until"]), "2026-12-02")
        self.assertIsNone(by_content["奇怪的 kind"][1]["kind"])
        self.assertEqual(result["schema_version"], 3)

    async def test_v2_backup_still_imports(self):
        conn, result = await self._import([
            {"backup_id": 1, "content": "老备份", "layer": 2, "event_date": "2026-01-01"},
        ], 2)
        row = next(iter(conn.rows.values()))
        self.assertIsNone(row["kind"])
        self.assertEqual(result["imported"], 1)

    async def test_dangling_superseded_by_is_degraded_not_fatal(self):
        conn, result = await self._import([
            {"backup_id": 1, "content": "旧条", "superseded_by": 99},
        ], 3)
        self.assertIsNone(next(iter(conn.rows.values()))["superseded_by"])
        self.assertEqual(result["degraded"], [{"backup_id": 1, "unresolved_superseded_by": 99}])


if __name__ == "__main__":
    unittest.main()
