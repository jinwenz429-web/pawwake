"""上游卡住/限流时换备用模型再试。"""

import asyncio
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx

import main

REAL_ASYNC_CLIENT = httpx.AsyncClient


def sse(text):
    frames = [
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return b"".join(("data: " + json.dumps(f, ensure_ascii=False) + "\n\n").encode() for f in frames) + b"data: [DONE]\n\n"


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved = []
        self.models = []

    def boundaries(self, handler, fallbacks="backup-a, backup-b", timeout=30):
        async def save_message(session_id, role, content, model, metadata=None):
            self.saved.append((role, content, model))
            return len(self.saved)

        async def upstream(request):
            model = json.loads(request.content)["model"]
            self.models.append(model)
            return await handler(model)

        stack = ExitStack()
        for name, value in {
            "CACHE_PARTITION_ENABLED": False,
            "MEMORY_ENABLED": False,
            "MEMORY_EXTRACT_ENABLED": False,
            "FORCE_STREAM": False,
            "REASONING_EFFORT": "",
            "FALLBACK_MODELS": fallbacks,
            "UPSTREAM_HEADER_TIMEOUT": timeout,
        }.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="")))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", save_message))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(upstream), **kwargs)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    async def run_stream(self):
        out = b"".join([chunk async for chunk in main.stream_and_capture(
            {}, {"model": "main-model", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]},
            "conv", "hi", "main-model",
        )])
        for _ in range(5):
            await asyncio.sleep(0)
        return out

    async def test_rate_limited_model_falls_back(self):
        async def handler(model):
            if model == "main-model":
                return httpx.Response(429, json={"error": {"message": "busy", "code": 429}})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse("在呢"))

        with self.boundaries(handler):
            out = await self.run_stream()
        self.assertEqual(self.models, ["main-model", "backup-a"])
        self.assertIn("在呢", out.decode())

    async def test_hanging_model_times_out_and_falls_back(self):
        async def handler(model):
            if model == "main-model":
                await asyncio.sleep(5)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse("来了"))

        with self.boundaries(handler, timeout=0.05):
            out = await self.run_stream()
        self.assertEqual(self.models, ["main-model", "backup-a"])
        self.assertIn("来了", out.decode())

    async def test_last_model_failure_is_reported_once_tried_everything(self):
        async def handler(model):
            return httpx.Response(502, text="bad gateway")

        with self.boundaries(handler):
            out = await self.run_stream()
        self.assertEqual(self.models, ["main-model", "backup-a", "backup-b"])
        self.assertIn('"error"', out.decode())
        self.assertEqual(self.saved, [])

    async def test_content_filter_is_not_retried_on_other_models(self):
        async def handler(model):
            return httpx.Response(400, json={"error": {"message": "blocked", "code": "content_filter"}})

        with self.boundaries(handler):
            await self.run_stream()
        self.assertEqual(self.models, ["main-model"])

    async def test_no_fallbacks_configured_behaves_as_before(self):
        async def handler(model):
            return httpx.Response(429, json={"error": {"message": "busy", "code": 429}})

        with self.boundaries(handler, fallbacks=""):
            out = await self.run_stream()
        self.assertEqual(self.models, ["main-model"])
        self.assertIn('"error"', out.decode())

    def test_chain_skips_duplicates_and_blanks(self):
        with patch.object(main, "FALLBACK_MODELS", " a, ,main , a,b "):
            self.assertEqual(main._fallback_chain("main"), ["main", "a", "b"])


if __name__ == "__main__":
    unittest.main()
