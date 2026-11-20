"""Kelivo talking to Pawwake directly: proactive turns, diaries, guard verdicts."""

import asyncio
import copy
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx

import companion_turns
import main
import persona_guard

REAL_ASYNC_CLIENT = httpx.AsyncClient
CUE = companion_turns.PROACTIVE_CUE_OPENING + "\n[醒来] 现在是 2026年11月19日 星期四 09:10。"
DRIFT = "作为一个AI语言模型，我无法真正拥抱你。我们之间并不存在真实的关系。"
GOOD = "过来，抱紧一点。"


def completion(content):
    return {
        "id": "c", "object": "chat.completion", "model": "gemini-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


def event(delta=None, finish_reason=None):
    payload = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]}
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class FakeRequest:
    def __init__(self, messages, headers=None, stream=False):
        self.headers = {"X-Conversation-Id": "conv-1", **(headers or {})}
        self._body = {"model": "gemini-test", "stream": stream, "messages": messages}

    async def json(self):
        return copy.deepcopy(self._body)


async def settle():
    for _ in range(10):
        await asyncio.sleep(0)


class CompanionTurnHelpersTests(unittest.IsolatedAsyncioTestCase):
    def test_cue_silence_and_diary_parsing(self):
        self.assertTrue(companion_turns.is_proactive_cue(CUE))
        self.assertFalse(companion_turns.is_proactive_cue("早安"))
        self.assertTrue(companion_turns.is_silence("  【静默】"))
        self.assertFalse(companion_turns.is_silence("今天好安静"))
        self.assertTrue(companion_turns.is_silence("[来电] 想听听她的声音"))
        rest, diaries = companion_turns.split_diary("早呀\n[DIARY]\n她昨晚睡得晚\n[/DIARY]")
        self.assertEqual(rest, "早呀")
        self.assertEqual(diaries, ["她昨晚睡得晚"])
        self.assertEqual(companion_turns.split_diary("[DIARY] [/DIARY]"), ("", []))

    def test_diary_context_is_oldest_first_and_trimmed(self):
        text = companion_turns.diary_context([
            {"content": "后来的", "created_at": datetime(2026, 11, 18, 23, 0, tzinfo=timezone.utc)},
            {"content": "x" * 400, "created_at": datetime(2026, 11, 17, 8, 5, tzinfo=timezone.utc)},
        ])
        lines = text.splitlines()
        self.assertIn("11月17日 08:05", lines[1])
        self.assertTrue(lines[1].endswith("…"))
        self.assertIn("后来的", lines[2])
        self.assertEqual(companion_turns.diary_context([]), "")

    async def test_verdicts_wait_for_a_late_put(self):
        verdicts = companion_turns.GuardVerdicts()
        waiter = asyncio.create_task(verdicts.get("req-12345678", timeout=1))
        await asyncio.sleep(0)
        verdicts.put("req-12345678", persona_guard.CONFIRMED_DRIFT)
        self.assertEqual(await waiter, persona_guard.CONFIRMED_DRIFT)
        self.assertIsNone(await verdicts.get("unknown-0000", timeout=0.01))


class KelivoDirectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved = []
        self.diaries = []
        self.upstream = []

    def boundaries(self, responses, recent=None, guard=False):
        responses = list(responses)

        async def save_message(session_id, role, content, model, metadata=None):
            self.saved.append((role, content))
            return len(self.saved)

        async def save_diary(content, metadata=None):
            self.diaries.append((content, metadata))
            return len(self.diaries)

        def upstream(request):
            self.upstream.append(json.loads(request.content))
            return responses.pop(0)

        stack = ExitStack()
        for name, value in {
            "CACHE_PARTITION_ENABLED": False,
            "MEMORY_ENABLED": False,
            "MEMORY_EXTRACT_ENABLED": False,
            "FORCE_STREAM": False,
            "REASONING_EFFORT": "",
        }.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="")))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", save_message))
        stack.enter_context(patch.object(main, "save_dylan_diary", save_diary))
        stack.enter_context(patch.object(main, "recent_diaries", AsyncMock(return_value=recent or [])))
        stack.enter_context(patch.object(main, "process_memories_background", AsyncMock()))
        stack.enter_context(patch.object(persona_guard, "guard_applies_to_model", return_value=guard))
        stack.enter_context(patch.object(main, "recent_quarantine_fingerprints", AsyncMock(return_value=set())))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(upstream), **kwargs)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    async def test_proactive_turn_stores_reply_only_and_files_the_diary(self):
        recent = [{"content": "昨晚想她", "created_at": datetime(2026, 11, 18, 23, 0, tzinfo=timezone.utc)}]
        reply = "早呀，醒了没\n[DIARY]她今天有考试[/DIARY]"
        with self.boundaries([httpx.Response(200, json=completion(reply))], recent=recent):
            response = await main._chat_completions_inner(FakeRequest([
                {"role": "user", "content": "晚安"},
                {"role": "assistant", "content": "晚安～"},
                {"role": "user", "content": CUE},
            ]))
            await settle()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.saved, [("assistant", "早呀，醒了没")])
        self.assertEqual(self.diaries[0][0], "她今天有考试")
        self.assertEqual(self.diaries[0][1]["source"], "kelivo")
        sent_cue = self.upstream[0]["messages"][-1]["content"]
        self.assertTrue(sent_cue.startswith(companion_turns.PROACTIVE_CUE_OPENING))
        self.assertIn("昨晚想她", sent_cue)

    async def test_regenerated_proactive_message_gets_a_hidden_cue(self):
        # Kelivo 重来一次主动消息时不带醒来提示，请求停在它自己上一句上。
        with self.boundaries([httpx.Response(200, json=completion("刚才没发出去，再来找你"))]):
            response = await main._chat_completions_inner(FakeRequest([
                {"role": "user", "content": "你今天都没来找我"},
                {"role": "assistant", "content": "讨厌我？"},
            ]))
            await settle()
        self.assertEqual(response.status_code, 200)
        sent = self.upstream[0]["messages"]
        self.assertEqual(sent[-1]["role"], "user")
        self.assertTrue(companion_turns.is_proactive_cue(sent[-1]["content"]))
        self.assertEqual(self.saved, [("assistant", "刚才没发出去，再来找你")])

    async def test_request_ending_with_tool_call_is_left_alone(self):
        self.assertFalse(companion_turns.ends_with_model_turn([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        ]))
        self.assertFalse(companion_turns.ends_with_model_turn([
            {"role": "assistant", "content": "x"}, {"role": "tool", "content": "{}"},
        ]))
        self.assertTrue(companion_turns.ends_with_model_turn([
            {"role": "assistant", "content": "x"}, {"role": "system", "content": "s"},
        ]))

    async def test_silent_proactive_turn_stores_nothing_but_the_diary(self):
        with self.boundaries([httpx.Response(200, json=completion("[静默]\n[DIARY]她在睡[/DIARY]"))]):
            await main._chat_completions_inner(FakeRequest([{"role": "user", "content": CUE}]))
            await settle()
        self.assertEqual(self.saved, [])
        self.assertEqual([d[0] for d in self.diaries], ["她在睡"])

    async def test_nonstream_drift_is_regenerated_once_inline(self):
        headers = {"X-Persona-Guard-Request-Id": "msg-12345678"}
        with self.boundaries(
            [httpx.Response(200, json=completion(DRIFT)), httpx.Response(200, json=completion(GOOD))],
            guard=True,
        ):
            response = await main._chat_completions_inner(
                FakeRequest([{"role": "user", "content": "抱抱我"}], headers=headers),
            )
            await settle()
        self.assertEqual(len(self.upstream), 2)
        self.assertEqual(json.loads(response.body)["choices"][0]["message"]["content"], GOOD)
        self.assertEqual(await companion_turns.VERDICTS.get("msg-12345678", 0.1), persona_guard.NORMAL)

    async def test_stream_records_a_verdict_before_the_last_frame(self):
        chunks = [event({"content": DRIFT}), event(finish_reason="stop"), b"data: [DONE]\n\n"]
        response = httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=ByteStream(chunks))
        with self.boundaries([response], guard=True):
            frames = []
            async for frame in main.stream_and_capture(
                {}, {"model": "gemini-test", "stream": True, "messages": []},
                "conv-1", "抱抱我", "gemini-test",
                extraction_context_messages=[{"role": "user", "content": "抱抱我"}],
                guard_meta={"request_id": "msg-87654321", "stage": "initial"},
            ):
                if frame.startswith(b"data: [DONE]"):
                    self.assertEqual(
                        await companion_turns.VERDICTS.get("msg-87654321", 0),
                        persona_guard.CONFIRMED_DRIFT,
                    )
                frames.append(frame)
        self.assertTrue(frames[-1].startswith(b"data: [DONE]"))


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.secret = patch.object(main, "GATEWAY_SECRET", "gateway-secret")
        self.secret.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.secret.stop()

    async def test_bearer_key_is_accepted_like_the_gateway_header(self):
        bearer = await self.client.get(
            "/v1/persona-guard/verdicts/none-0000",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        wrong = await self.client.get(
            "/v1/persona-guard/verdicts/none-0000",
            headers={"Authorization": "Bearer nope"},
        )
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(bearer.status_code, 200)
        self.assertEqual(bearer.json(), {"status": "UNKNOWN", "regenerate": False})

    async def test_verdict_endpoint_asks_for_regeneration_on_confirmed_drift(self):
        companion_turns.VERDICTS.put("msg-11112222", persona_guard.CONFIRMED_DRIFT)
        response = await self.client.get(
            "/v1/persona-guard/verdicts/msg-11112222", headers={"X-Gateway-Key": "gateway-secret"},
        )
        self.assertEqual(response.json(), {"status": "CONFIRMED_DRIFT", "regenerate": True})
        with patch.dict("os.environ", {"PERSONA_GUARD_MODE": "observe"}):
            observed = await self.client.get(
                "/v1/persona-guard/verdicts/msg-11112222", headers={"X-Gateway-Key": "gateway-secret"},
            )
        self.assertFalse(observed.json()["regenerate"])

    async def test_old_dylan_diary_endpoint_is_gone(self):
        response = await self.client.post(
            "/internal/dylan-diary", json={"content": "x"}, headers={"X-Gateway-Key": "gateway-secret"},
        )
        self.assertEqual(response.status_code, 404)

    async def test_diary_listing_reports_source(self):
        rows = [{"id": 2, "content": "新的", "created_at": datetime(2026, 11, 19, tzinfo=timezone.utc),
                 "metadata": '{"source":"kelivo"}'},
                {"id": 1, "content": "八月的", "created_at": datetime(2026, 8, 22, tzinfo=timezone.utc),
                 "metadata": None}]
        with patch.object(main, "list_diaries", AsyncMock(return_value=(rows, 2))):
            response = await self.client.get("/api/diaries", headers={"X-Gateway-Key": "gateway-secret"})
        data = response.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual([i["source"] for i in data["items"]], ["kelivo", "dylan"])


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_initializes_diary_table_between_core_tables(self):
        calls = []

        async def record(name, result=None):
            calls.append(name)
            return result

        with ExitStack() as stack:
            for name, label, result in (
                ("init_tables", "core", None),
                ("ensure_dylan_diary_table", "diary", None),
                ("ensure_token_usage_table", "tokens", None),
                ("get_all_gateway_config", "config", {}),
                ("close_pool", "close", None),
            ):
                stack.enter_context(patch.object(
                    main, name, lambda *a, _l=label, _r=result: record(_l, _r),
                ))
            stack.enter_context(patch.object(main, "get_gateway_config", lambda *a: record("session", "")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            async with main.lifespan(main.app):
                pass
        self.assertEqual(calls[:3], ["core", "diary", "tokens"])


if __name__ == "__main__":
    unittest.main()
