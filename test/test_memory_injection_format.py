import unittest
from datetime import date, datetime
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
        ):
            memory_block = await main.build_memory_text("query")

        self.assertEqual(
            memory_block,
            "<retrieved_memories>\n"
            "以下是网关从过往对话中自动检索的相关记忆，供参考，非用户本次输入：\n"
            "- 发生日期：2026-08-22；event memory\n"
            "- 发生日期：2021-08-22；timestamp memory\n"
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
        self.assertIn("- 发生日期：2026-08-22；private context", user_text)
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
        ):
            prompt = await main.build_system_prompt_with_memories("query", "persona")

        self.assertIn("- 发生日期：2026-08-22；shared memory", prompt)
        self.assertNotIn("- [2026-08-22] shared memory", prompt)


if __name__ == "__main__":
    unittest.main()
