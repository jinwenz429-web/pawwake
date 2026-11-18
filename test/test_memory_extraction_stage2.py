"""阶段 2：提取窗口带时间、去掉注入块和 tool；kind / event_date / valid_until 入库。"""

import io
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import database
import main


class ExtractionWindowTests(unittest.TestCase):
    def test_window_carries_created_at_and_drops_tool_and_injected_blocks(self):
        sent = datetime(2026, 11, 18, 15, 40, tzinfo=timezone.utc)
        messages, rounds = main._build_memory_extraction_messages([
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "<retrieved_memories>\n- 旧记忆\n</retrieved_memories>\n\n我改喝茶了",
             "source_id": 5, "created_at": sent},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}], "source_id": 6},
            {"role": "tool", "content": "- [记于 2026-10-01] 喜欢咖啡", "source_id": 7},
        ], "好，记住了", 1, current_assistant_id=8)
        self.assertEqual(rounds, 1)
        self.assertEqual(messages, [
            {"role": "user", "content": "我改喝茶了", "source_id": 5, "created_at": sent},
            {"role": "assistant", "content": "好，记住了", "source_id": 8, "created_at": None},
        ])


class ValidUntilTests(unittest.TestCase):
    def test_rules(self):
        with patch.object(database, "MEMORY_STATE_TTL_DAYS", 14):
            self.assertEqual(database.compute_valid_until("state", None, date(2026, 11, 18)),
                             date(2026, 12, 2))
        self.assertEqual(database.compute_valid_until("plan", date(2026, 12, 20), date(2026, 11, 18)),
                         date(2026, 12, 23))
        self.assertIsNone(database.compute_valid_until("plan", None, date(2026, 11, 18)))
        for kind in ("fact", "event", "promise", None):
            self.assertIsNone(database.compute_valid_until(kind, date(2026, 1, 1), date(2026, 11, 18)))


class ExtractionStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_kind_event_date_and_code_computed_valid_until_are_stored(self):
        saved = []

        async def save(**kwargs):
            saved.append(kwargs)
            return {"action": "inserted", "memory_id": len(saved)}

        extracted = [
            {"content": "截至 2026-11-18 在赶毕业论文", "importance": 6, "source_ids": [10],
             "kind": "state", "event_date": None, "action": "new", "target_id": None},
            {"content": "计划 2026-12-20 考研初试", "importance": 9, "source_ids": [10],
             "kind": "plan", "event_date": date(2026, 12, 20), "action": "new", "target_id": None},
        ]
        with (
            patch.object(main, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(main, "extract_memories", AsyncMock(return_value=extracted)),
            patch.object(main, "save_memory_deduplicated", save),
            patch.object(main, "get_active_memories_count", AsyncMock(return_value=2)),
            patch.object(main, "search_memories", AsyncMock(return_value=[])),
            patch.object(main, "get_core_memories", AsyncMock(return_value=[])),
            patch.object(database, "MEMORY_STATE_TTL_DAYS", 14),
            patch.object(main, "MEMORY_EXTRACT_INTERVAL", 1),
            patch.object(main, "datetime") as fake_datetime,
            redirect_stdout(io.StringIO()),
        ):
            fake_datetime.now.return_value = datetime(2026, 11, 18, 4, 0, tzinfo=timezone.utc)
            await main._extract_and_store_memories(
                "s1", "我在赶论文，12 月 20 号考研", "加油",
                [{"role": "user", "content": "我在赶论文，12 月 20 号考研", "source_id": 10}],
                10, 11, None,
            )
        self.assertEqual(saved[0]["kind"], "state")
        self.assertEqual(saved[0]["valid_until"], date(2026, 12, 2))
        self.assertEqual(saved[1]["event_date"], date(2026, 12, 20))
        self.assertEqual(saved[1]["valid_until"], date(2026, 12, 23))


if __name__ == "__main__":
    unittest.main()
