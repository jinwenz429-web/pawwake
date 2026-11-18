import unittest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import main


def _message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


class MemoryInjectionFormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_partition_memory_block_uses_non_citation_date_format(self):
        memories = [
            {
                "id": 1,
                "content": "event memory",
                "importance": 8,
                "event_date": date(2026, 8, 22),
                "created_at": datetime(2025, 1, 1, 0, 0),
            },
            {
                "id": 2,
                "content": "timestamp memory",
                "importance": 7,
                "event_date": None,
                "created_at": datetime(2021, 8, 21, 18, 30),
            },
            {
                "id": 3,
                "content": "undated memory",
                "importance": 6,
                "event_date": None,
                "created_at": None,
            },
        ]

        with (
            patch.object(main, "search_memories", AsyncMock(return_value=memories)),
            patch.object(main, "TIMEZONE_HOURS", 8),
            patch.object(main, "_local_today", return_value=date(2026, 8, 23)),
        ):
            memory_block = await main.build_memory_text("query")

        self.assertEqual(
            memory_block,
            "<retrieved_memories>\n"
            "以下是网关从过往对话中自动检索的相关记忆，供参考，非用户本次输入：\n"
            "- [2026-08-22｜1 天前] event memory\n"
            "- [记于 2021-08-22｜1827 天前] timestamp memory\n"
            "- undated memory\n"
            "</retrieved_memories>",
        )

    async def test_partition_system_instruction_forbids_citation_labels(self):
        memories = [
            {
                "id": 1,
                "content": "private context",
                "importance": 8,
                "event_date": date(2026, 8, 22),
                "created_at": datetime(2026, 8, 22, 0, 0),
            }
        ]

        with (
            patch.object(main, "search_memories", AsyncMock(return_value=memories)),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "_local_today", return_value=date(2026, 8, 22)),
        ):
            messages = await main._build_basic_cached(
                history=[],
                base_prompt="persona" + main.MEMORY_USAGE_GUIDE,
                user_message="hello",
                current_user_msg={"role": "user", "content": "hello"},
            )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        system_text = _message_text(messages[0])
        self.assertIn("自然吸收、运用这些记忆", system_text)
        self.assertIn("citation", system_text)
        self.assertIn("source", system_text)
        self.assertIn("reference", system_text)
        self.assertIn("[cite:...]", system_text)
        self.assertIn("日期仅用于内部判断时间先后，不是来源标记", system_text)
        self.assertIn("不应在回答中复现", system_text)
        self.assertNotIn("自然引用", system_text)
        self.assertNotIn("相关话题出现时引用", system_text)

        user_text = _message_text(messages[-1])
        self.assertIn("- [2026-08-22｜今天] private context", user_text)
        self.assertIn("以它为准", system_text)
        self.assertLess(user_text.index("<retrieved_memories>"), user_text.index("hello"))

    async def test_non_partition_prompt_uses_same_memory_line_format(self):
        memories = [
            {
                "id": 1,
                "content": "shared memory",
                "importance": 8,
                "event_date": date(2026, 8, 22),
                "created_at": datetime(2026, 8, 22, 0, 0),
            }
        ]

        with (
            patch.object(main, "search_memories", AsyncMock(return_value=memories)),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "get_core_memories", AsyncMock(return_value=[])),
            patch.object(main, "_local_today", return_value=date(2026, 8, 25)),
        ):
            prompt = await main.build_system_prompt_with_memories("query", "persona")

        self.assertIn("- [2026-08-22｜3 天前] shared memory", prompt)
        self.assertIn("以它为准", prompt)


class MemoryLineFormatTests(unittest.TestCase):
    TODAY = date(2026, 11, 19)

    def line(self, **mem):
        mem.setdefault("content", "内容")
        return main._format_memory_line(mem, self.TODAY)

    def test_event_with_kind(self):
        self.assertEqual(self.line(event_date=date(2026, 11, 18), kind="event"),
                         "- [2026-11-18｜1 天前｜事件] 内容")

    def test_future_plan(self):
        self.assertEqual(self.line(event_date=date(2026, 12, 20), kind="plan",
                                   valid_until=date(2026, 12, 23)),
                         "- [计划 2026-12-20｜还有 31 天] 内容")

    def test_expired_plan(self):
        self.assertEqual(self.line(event_date=date(2026, 11, 10), kind="plan",
                                   valid_until=date(2026, 11, 13)),
                         "- [计划 2026-11-10｜9 天前] 内容（原定 2026-11-10，已过去）")

    def test_plan_within_grace_period_is_not_marked_expired(self):
        self.assertEqual(self.line(event_date=date(2026, 11, 17), kind="plan",
                                   valid_until=date(2026, 11, 20)),
                         "- [计划 2026-11-17｜2 天前] 内容")

    def test_no_event_date_uses_record_date(self):
        self.assertEqual(self.line(created_at=datetime(2026, 11, 1, 2, 0), kind="fact"),
                         "- [记于 2026-11-01｜18 天前｜事实] 内容")

    def test_created_at_converts_to_local_date(self):
        # UTC 2026-11-01 20:00 = 东八区 11-02 04:00
        with patch.object(main, "TIMEZONE_HOURS", 8):
            line = self.line(created_at=datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc))
        self.assertEqual(line, "- [记于 2026-11-02｜17 天前] 内容")

    def test_stale_state_is_flagged(self):
        self.assertEqual(self.line(created_at=datetime(2026, 11, 1, 2, 0), kind="state",
                                   valid_until=date(2026, 11, 15)),
                         "- [记于 2026-11-01｜18 天前｜状态] 内容（这是 2026-11-01 时的状态，可能已经变了）")

    def test_fresh_state_is_not_flagged(self):
        self.assertEqual(self.line(created_at=datetime(2026, 11, 18, 2, 0), kind="state",
                                   valid_until=date(2026, 12, 2)),
                         "- [记于 2026-11-18｜1 天前｜状态] 内容")

    def test_string_dates_from_json(self):
        self.assertEqual(self.line(event_date="2026-11-19", kind="promise"),
                         "- [2026-11-19｜今天｜约定] 内容")


class CoreMemoryInjectionTests(unittest.IsolatedAsyncioTestCase):
    CORE = [{"id": 7, "content": "她叫阿锦", "importance": 10, "layer": 3,
             "event_date": None, "created_at": datetime(2026, 1, 1), "kind": "fact"}]

    async def test_core_is_injected_first_and_deduped_from_search(self):
        searched = []

        async def search(query, limit=10):
            searched.append(limit)
            return [{"id": 7, "content": "她叫阿锦", "importance": 10, "event_date": None,
                     "created_at": datetime(2026, 1, 1)},
                    {"id": 8, "content": "喜欢喝茶", "importance": 6, "event_date": None,
                     "created_at": datetime(2026, 1, 1)}]

        with (
            patch.object(main, "search_memories", search),
            patch.object(main, "get_core_memories", AsyncMock(return_value=self.CORE)),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "MAX_MEMORIES_INJECT", 5),
            patch.object(main, "MEMORY_CORE_ALWAYS_INJECT", 8),
        ):
            prompt = await main.build_system_prompt_with_memories("query", "persona")
        self.assertEqual(searched, [6])
        self.assertEqual(prompt.count("她叫阿锦"), 1)
        self.assertLess(prompt.index("核心记忆"), prompt.index("喜欢喝茶"))

    async def test_core_is_injected_even_without_a_query(self):
        with (
            patch.object(main, "search_memories", AsyncMock(return_value=[])),
            patch.object(main, "get_core_memories", AsyncMock(return_value=self.CORE)),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
        ):
            prompt = await main.build_system_prompt_with_memories("", "persona")
        self.assertIn("她叫阿锦", prompt)

    async def test_core_block_goes_into_system_and_search_excludes_it(self):
        searched = []

        async def search(query, limit=10):
            searched.append((query, limit))
            return [{"id": 7, "content": "她叫阿锦", "importance": 10, "event_date": None,
                     "created_at": datetime(2026, 1, 1)}]

        with (
            patch.object(main, "search_memories", search),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
        ):
            messages = await main._build_basic_cached(
                history=[],
                base_prompt="persona\n\n" + main.format_core_memory_block(self.CORE),
                user_message="hello",
                current_user_msg={"role": "user", "content": "hello"},
                memory_exclude_ids=[7],
            )
        self.assertIn("<core_memories>", _message_text(messages[0]))
        self.assertNotIn("<retrieved_memories>", _message_text(messages[-1]))


if __name__ == "__main__":
    unittest.main()
