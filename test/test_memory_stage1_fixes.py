"""阶段 1：来源核对、meta 过滤、主动回合搜索词、按会话串行提取、跨批次 KEEP。"""

import asyncio
import copy
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx

import companion_turns
import main
import memory_rebuilder
import persona_guard

REAL_ASYNC_CLIENT = httpx.AsyncClient
CUE = companion_turns.PROACTIVE_CUE_OPENING + "\n[醒来] 现在是 2026年11月19日 星期四 09:10。"


def completion(content):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}]}


class FakeRequest:
    def __init__(self, messages):
        self.headers = {"X-Conversation-Id": "conv-1"}
        self._body = {"model": "gemini-test", "stream": False, "messages": messages}

    async def json(self):
        return copy.deepcopy(self._body)


WINDOW = [
    {"role": "user", "content": "我下周三要考试", "source_id": 10},
    {"role": "assistant", "content": "你下周三考试，加油", "source_id": 11},
]


class SourceVerificationTests(unittest.TestCase):
    def test_user_source_is_required(self):
        self.assertEqual(main._verified_memory_sources({"source_ids": [10, 11]}, WINDOW), [10, 11])
        self.assertIsNone(main._verified_memory_sources({"source_ids": [11]}, WINDOW))

    def test_promise_may_rest_on_assistant_alone(self):
        memory = {"source_ids": [11], "kind": "promise"}
        self.assertEqual(main._verified_memory_sources(memory, WINDOW), [11])

    def test_ids_outside_the_window_are_ignored(self):
        self.assertIsNone(main._verified_memory_sources({"source_ids": [99, "x", True]}, WINDOW))
        self.assertEqual(main._verified_memory_sources({"source_ids": ["10", 99]}, WINDOW), [10])

    def test_meta_blacklist_keeps_project_progress(self):
        for text in ("把 pawwake 部署到 Render 了", "修好了数据库的 bug", "网关导入完成"):
            self.assertFalse(any(word in text for word in main.META_BLACKLIST), text)
        self.assertTrue(any(word in "记忆系统没有被记录这件事" for word in main.META_BLACKLIST))


class RecallQueryTests(unittest.TestCase):
    def test_proactive_cue_is_skipped_and_real_dialogue_is_used(self):
        query = main.build_recall_query([
            {"role": "user", "content": "今天面试好紧张"},
            {"role": "assistant", "content": "面试完告诉我结果"},
            {"role": "user", "content": CUE},
        ])
        self.assertNotIn(companion_turns.PROACTIVE_CUE_OPENING, query)
        self.assertIn("今天面试好紧张", query)
        self.assertIn("面试完告诉我结果", query)

    def test_injected_blocks_are_stripped(self):
        query = main.build_recall_query([
            {"role": "user", "content": "<retrieved_memories>\n- 旧记忆\n</retrieved_memories>\n\n真正的问题"},
        ])
        self.assertEqual(query, "真正的问题")

    def test_only_cue_gives_empty_query(self):
        self.assertEqual(main.build_recall_query([{"role": "user", "content": CUE}]), "")


class ProactiveRecallIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_proactive_turn_searches_with_recent_dialogue_not_the_cue(self):
        searched = []

        async def search(query, limit=10):
            searched.append(query)
            return []

        stack = ExitStack()
        for name, value in {
            "CACHE_PARTITION_ENABLED": False, "MEMORY_ENABLED": True,
            "MEMORY_EXTRACT_ENABLED": True, "FORCE_STREAM": False, "REASONING_EFFORT": "",
        }.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="persona")))
        stack.enter_context(patch.object(main, "search_memories", search))
        stack.enter_context(patch.object(main, "get_core_memories", AsyncMock(return_value=[]), create=True))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", AsyncMock(return_value=1)))
        stack.enter_context(patch.object(main, "save_dylan_diary", AsyncMock()))
        stack.enter_context(patch.object(main, "recent_diaries", AsyncMock(return_value=[])))
        stack.enter_context(patch.object(main, "process_memories_background", AsyncMock()))
        stack.enter_context(patch.object(persona_guard, "guard_applies_to_model", return_value=False))
        stack.enter_context(patch.object(main, "recent_quarantine_fingerprints", AsyncMock(return_value=set())))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=completion("早"))),
            **kwargs)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        with stack:
            await main._chat_completions_inner(FakeRequest([
                {"role": "user", "content": "明天要去医院复查"},
                {"role": "assistant", "content": "我陪你，记得带病历"},
                {"role": "user", "content": CUE},
            ]))
        self.assertEqual(len(searched), 1)
        self.assertNotIn(companion_turns.PROACTIVE_CUE_OPENING, searched[0])
        self.assertIn("明天要去医院复查", searched[0])


class ExtractionLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_session_extractions_run_one_at_a_time(self):
        running = 0
        peak = 0

        async def fake_extract(*args, **kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return []

        with (
            patch.object(main, "_extract_and_store_memories", fake_extract),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_INTERVAL", 1),
            patch.object(main, "save_message", AsyncMock(return_value=1)),
            redirect_stdout(io.StringIO()),
        ):
            await asyncio.gather(*[
                main.process_memories_background("s1", f"消息{i}", "回复", "m", context_round_count=1)
                for i in range(3)
            ])
        self.assertEqual(peak, 1)

    async def test_different_sessions_do_not_block_each_other(self):
        running = 0
        peak = 0

        async def fake_extract(*args, **kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

        with (
            patch.object(main, "_extract_and_store_memories", fake_extract),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_INTERVAL", 1),
            patch.object(main, "save_message", AsyncMock(return_value=1)),
            redirect_stdout(io.StringIO()),
        ):
            await asyncio.gather(*[
                main.process_memories_background(f"s{i}", "消息", "回复", "m", context_round_count=1)
                for i in range(3)
            ])
        self.assertEqual(peak, 3)


class ExtractionStatsTests(unittest.IsolatedAsyncioTestCase):
    async def test_assistant_only_memory_is_dropped_and_total_counts_active(self):
        saved = []

        async def save(**kwargs):
            saved.append(kwargs)
            return {"action": "inserted", "memory_id": 1}

        out = io.StringIO()
        with (
            patch.object(main, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(main, "extract_memories", AsyncMock(return_value=[
                {"content": "用户下周三考试", "importance": 7, "source_ids": [10]},
                {"content": "AI 猜用户很焦虑", "importance": 5, "source_ids": [11]},
                {"content": "答应考完陪她复盘", "importance": 7, "source_ids": [11], "kind": "promise"},
            ])),
            patch.object(main, "save_memory_deduplicated", save),
            patch.object(main, "get_active_memories_count", AsyncMock(return_value=42)),
            patch.object(main, "get_all_memories_count", AsyncMock(return_value=999)),
            patch.object(main, "MEMORY_EXTRACT_INTERVAL", 1),
            redirect_stdout(out),
        ):
            await main._extract_and_store_memories(
                "s1", "我下周三要考试", "你下周三考试，加油",
                [{"role": "user", "content": "我下周三要考试", "source_id": 10}],
                10, 11, None,
            )
        self.assertEqual([item["content"] for item in saved], ["用户下周三考试", "答应考完陪她复盘"])
        self.assertIn("总计 42 条", out.getvalue())
        self.assertNotIn("999", out.getvalue())


class CrossBatchKeepTests(unittest.IsolatedAsyncioTestCase):
    async def test_keep_of_a_provisional_merge_keeps_that_merge(self):
        candidates = [
            {"id": i, "decision": "core", "topic": "t", "normalized_content": f"c{i}",
             "importance": 5, "source": {"layer": 1, "content": f"c{i}"}}
            for i in (1, 2, 3)
        ]
        calls = []

        async def fake_once(items, label):
            calls.append([item["id"] for item in items])
            if label.startswith("跨批次"):
                return [{"action": "KEEP", "source_ids": [items[0]["id"]], "target_layer": 3,
                         "title": "新标题", "content": "", "importance": 9, "reason": ""},
                        {"action": "KEEP", "source_ids": [items[1]["id"]], "target_layer": 3,
                         "title": "", "content": "", "importance": 5, "reason": ""}]
            if len(items) == 2:
                return [{"action": "MERGE", "source_ids": [1, 2], "target_layer": 3,
                         "title": "合并", "content": "c1+c2", "importance": 7, "reason": ""}]
            return [{"action": "KEEP", "source_ids": [3], "target_layer": 1,
                     "title": "", "content": "c3", "importance": 5, "reason": ""}]

        with patch.object(memory_rebuilder, "REBUILD_GROUP_SIZE", 2), \
                patch.object(memory_rebuilder, "_synthesize_once", fake_once):
            actions = await memory_rebuilder._synthesize_domain(candidates, "goals")
        merge = next(a for a in actions if a["action"] == "MERGE")
        self.assertEqual(merge["source_ids"], [1, 2])
        self.assertEqual(merge["content"], "c1+c2")
        self.assertEqual((merge["title"], merge["importance"]), ("新标题", 9))
        keep = next(a for a in actions if a["action"] == "KEEP")
        self.assertEqual(keep["source_ids"], [3])


if __name__ == "__main__":
    unittest.main()
