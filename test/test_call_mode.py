"""Voice calls: lean per-call context, nothing stored mid-call, one round after hang-up."""

import asyncio
import copy
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx

import call_mode
import companion_turns
import main

REAL_ASYNC_CLIENT = httpx.AsyncClient


def completion(content):
    return {
        "id": "c", "object": "chat.completion", "model": "gemini-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


class FakeRequest:
    def __init__(self, body, headers=None):
        self.headers = {"X-Conversation-Id": "conv-1", **(headers or {})}
        self._body = body

    async def json(self):
        return copy.deepcopy(self._body)


def mem(mem_id, content):
    return {"id": mem_id, "content": content, "kind": "fact",
            "created_at": datetime(2026, 11, 1, tzinfo=timezone.utc)}


class HelperTests(unittest.TestCase):
    def test_recent_history_keeps_last_rounds_as_plain_text(self):
        rows = [
            {"role": "user", "content": "一"}, {"role": "assistant", "content": "回一"},
            {"role": "user", "content": "二"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
            {"role": "tool", "content": "x"},
            {"role": "assistant", "content": "回二"},
            {"role": "user", "content": "三"}, {"role": "assistant", "content": "回三"},
        ]
        self.assertEqual(
            call_mode.recent_history(rows, 2),
            [{"role": "user", "content": "二"}, {"role": "assistant", "content": "回二"},
             {"role": "user", "content": "三"}, {"role": "assistant", "content": "回三"}],
        )

    def test_call_messages_merge_same_role_neighbours(self):
        merged = call_mode.call_messages([
            {"role": "user", "content": "喂"},
            {"role": "assistant", "content": "嗯？"},
            {"role": "assistant", "content": "怎么不说话"},
            {"role": "user", "content": [{"type": "text", "text": "在呢"}]},
        ])
        self.assertEqual([m["role"] for m in merged], ["user", "assistant", "user"])
        self.assertEqual(merged[1]["content"], "嗯？\n怎么不说话")
        self.assertEqual(merged[2]["content"], "在呢")

    def test_memory_lookup_turns_are_kept(self):
        call = [
            {"role": "user", "content": "我们上次说去哪"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
             "function": {"name": "search_memory", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "海边"},
        ]
        turns = call_mode.call_messages(call)
        self.assertEqual([m["role"] for m in turns], ["user", "assistant", "tool"])
        self.assertTrue(call_mode.ends_awaiting_reply(turns))
        out = call_mode.build_messages(call_mode.CallContext("s", "sys"), turns, "[时间]", "")
        self.assertEqual(out[-1]["role"], "tool")
        self.assertEqual(out[1]["content"], "我们上次说去哪")

    def test_build_messages_puts_time_and_memories_on_the_last_line(self):
        ctx = call_mode.CallContext("s", "人设", ["之前聊了考研"],
                                    [{"role": "user", "content": "晚安"},
                                     {"role": "assistant", "content": "睡吧"}])
        out = call_mode.build_messages(
            ctx, [{"role": "user", "content": "在吗"}], "[时间]", "<retrieved_memories>x</retrieved_memories>",
        )
        self.assertEqual(out[0], {"role": "system", "content": "人设"})
        self.assertIn("之前聊了考研", out[1]["content"])
        self.assertEqual(out[3]["content"], "晚安")
        self.assertTrue(out[-1]["content"].startswith("[时间]"))
        self.assertTrue(out[-1]["content"].endswith("在吗"))
        roles = [m["role"] for m in out[1:]]
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])))

    def test_context_store_expires_and_caps(self):
        store = call_mode.CallContextStore(ttl_seconds=10, max_entries=2)
        for i in range(3):
            store.put(f"c{i}", call_mode.CallContext("s", "", created_at=float(i)), now=float(i))
        self.assertIsNone(store.get("c0", now=3))
        self.assertIsNotNone(store.get("c2", now=3))
        self.assertIsNone(store.get("c2", now=20))

    def test_transcript_round_and_extraction_ids(self):
        t = call_mode.parse_transcript({
            "lines": [
                {"role": "assistant", "text": "[laughs] 喂？"},
                {"role": "user", "text": "我明天考试", "at": "2026-11-20T13:31:00Z"},
                {"role": "assistant", "text": "别紧张"},
            ],
            "started_at": "2026-11-20T13:30:00Z", "ended_at": "2026-11-20T13:58:20Z",
            "user_label": "锦", "assistant_label": "沈聿安",
        })
        user_content, assistant_content = call_mode.stored_round(t, 8)
        self.assertTrue(user_content.startswith("[语音通话 2026-11-20 21:30–21:58，28 分钟]\n"))
        self.assertIn("锦：我明天考试", user_content)
        self.assertIn("沈聿安：别紧张", user_content)
        self.assertEqual(assistant_content, call_mode.CALL_ENDED_REPLY)

        msgs, roles = call_mode.extraction_messages(t)
        self.assertEqual(len({m["source_id"] for m in msgs}), 3)
        user_sid = msgs[1]["source_id"]
        ai_sid = msgs[0]["source_id"]
        # A fact backed only by his lines is rejected by the existing provenance check.
        self.assertIsNone(main._verified_memory_sources({"source_ids": [ai_sid], "kind": "fact"}, msgs))
        verified = main._verified_memory_sources({"source_ids": [user_sid, ai_sid], "kind": "event"}, msgs)
        self.assertEqual(call_mode.map_source_ids(verified, roles, 41, 42), [41, 42])

    def test_transcript_validation(self):
        for bad in ({}, {"lines": []}, {"lines": [{"role": "system", "text": "x"}]}):
            with self.assertRaises(call_mode.TranscriptError):
                call_mode.parse_transcript(bad)


class CallTurnTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        call_mode.CONTEXTS = call_mode.CallContextStore()
        self.upstream = []
        self.saved = []

    def boundaries(self, responses, search_results=None):
        searches = list(search_results or [])
        self.searches = []

        async def search(query, limit, exclude_ids=None):
            self.searches.append((query, limit, set(exclude_ids or [])))
            return searches.pop(0) if searches else []

        async def save_message(session_id, role, content, model, metadata=None):
            self.saved.append((session_id, role, content))
            return len(self.saved) + 100

        def upstream(request):
            self.upstream.append(json.loads(request.content))
            return responses.pop(0)

        stack = ExitStack()
        for name, value in {
            "CACHE_PARTITION_ENABLED": True,
            "MEMORY_ENABLED": True,
            "MEMORY_EXTRACT_ENABLED": True,
            "MEMORY_EXTRACT_INTERVAL": 8,
            "FORCE_STREAM": False,
            "REASONING_EFFORT": "high",
        }.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="你是沈聿安")))
        stack.enter_context(patch.object(main, "_load_core_memories", AsyncMock(return_value=[mem(1, "核心")])))
        stack.enter_context(patch.object(main, "get_conversation_tail", AsyncMock(return_value=[
            {"id": 7, "role": "user", "content": "晚安", "metadata": None},
            {"id": 8, "role": "assistant", "content": "睡吧", "metadata": None},
        ])))
        stack.enter_context(patch.object(main, "get_session_cache_state", AsyncMock(
            return_value={"summary_parts": ["摘要"], "a_start_round": 0})))
        stack.enter_context(patch.object(main, "get_session_tag", AsyncMock(return_value=None)))
        stack.enter_context(patch.object(main, "search_memories_excluding", search))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", save_message))
        stack.enter_context(patch.object(main, "process_memories_background", AsyncMock()))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(upstream), **kwargs)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    async def test_call_turns_reuse_one_base_and_store_nothing(self):
        responses = [httpx.Response(200, json=completion("喂？")), httpx.Response(200, json=completion("嗯，我在"))]
        cue = companion_turns.PROACTIVE_CUE_OPENING + "电话刚接通，你先开口。"
        with self.boundaries(responses, [[mem(2, "她在准备考研")], [mem(3, "她怕黑")]]):
            first = await main._chat_completions_inner(FakeRequest({
                "model": "gemini-test", "reasoning_effort": "low",
                "messages": [{"role": "system", "content": "现在在打电话，说短句"},
                             {"role": "user", "content": cue}],
            }, {"X-Call-Id": "call-1"}))
            second = await main._chat_completions_inner(FakeRequest({
                "model": "gemini-test",
                "messages": [{"role": "system", "content": "现在在打电话，说短句"},
                             {"role": "user", "content": cue},
                             {"role": "assistant", "content": "喂？"},
                             {"role": "user", "content": "我有点怕黑"}],
            }, {"X-Call-Id": "call-1"}))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(self.saved, [])

        # Base built once (one base search), then one per-turn search for the real line only.
        self.assertEqual(len(self.searches), 2)
        self.assertEqual(self.searches[0][1], call_mode.BASE_MEMORIES)
        self.assertEqual(self.searches[1], ("我有点怕黑", call_mode.TURN_MEMORIES, {1, 2}))

        sent = self.upstream[1]
        system = sent["messages"][0]["content"]
        self.assertIn("你是沈聿安", system)
        self.assertIn("核心", system)
        self.assertIn("她在准备考研", system)
        self.assertIn("现在在打电话，说短句", system)
        self.assertIn("她怕黑", sent["messages"][-1]["content"])
        self.assertTrue(sent["messages"][-1]["content"].endswith("我有点怕黑"))
        self.assertNotIn("她怕黑", system)
        # The client's own reasoning level is kept; the gateway's "high" is not forced on calls.
        self.assertEqual(self.upstream[0].get("reasoning_effort"), "low")

    async def test_transcript_is_stored_once_and_mined_per_line(self):
        extracted = AsyncMock()
        body = {
            "lines": [{"role": "assistant", "text": "喂？"}, {"role": "user", "text": "我明天考试"}],
            "started_at": "2026-11-20T13:30:00Z", "ended_at": "2026-11-20T13:40:00Z",
            "user_label": "锦", "assistant_label": "沈聿安",
        }
        with self.boundaries([]):
            with patch.object(main, "_extract_and_store_memories", extracted):
                result = await main.save_call_transcript("call-1", FakeRequest(body))
                for _ in range(10):
                    await asyncio.sleep(0)
        self.assertTrue(result["stored"])
        self.assertEqual([role for _, role, _ in self.saved], ["user", "assistant"])
        self.assertEqual(self.saved[0][2], result["user_content"])
        self.assertEqual(self.saved[1][2], result["assistant_content"])
        kwargs = extracted.await_args.kwargs
        self.assertEqual(len(kwargs["prepared_messages"]), 2)
        self.assertEqual(set(kwargs["synthetic_roles"].values()), {"user", "assistant"})

    async def test_retried_transcript_is_not_stored_twice(self):
        body = {"lines": [{"role": "user", "text": "在吗"}], "started_at": "2026-11-20T13:30:00Z"}
        with self.boundaries([]):
            user_content, _ = call_mode.stored_round(call_mode.parse_transcript(body), main.TIMEZONE_HOURS)
            with patch.object(main, "get_conversation_tail", AsyncMock(return_value=[
                {"id": 9, "role": "user", "content": user_content},
                {"id": 10, "role": "assistant", "content": call_mode.CALL_ENDED_REPLY},
            ])):
                result = await main.save_call_transcript("call-1", FakeRequest(body))
        self.assertTrue(result["duplicate"])
        self.assertEqual(self.saved, [])


if __name__ == "__main__":
    unittest.main()
