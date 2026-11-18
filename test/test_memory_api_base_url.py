"""记忆模型单独的 API 地址：留空回退主地址；提取、整理新碎片、按日整理都走它；面板能热更新。"""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx

import main
import memory_extractor
import memory_rebuilder

REAL_ASYNC_CLIENT = httpx.AsyncClient
MAIN_URL = "https://main.example/v1/chat/completions"
MEMORY_URL = "https://memory.example/v1/chat/completions"


def reply(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


class FallbackTests(unittest.TestCase):
    def test_empty_falls_back_to_main_url(self):
        with patch.object(memory_extractor, "API_BASE_URL", MAIN_URL), \
                patch.object(memory_extractor, "MEMORY_API_BASE_URL", "  "):
            self.assertEqual(memory_extractor.get_memory_api_base_url(), MAIN_URL)
        with patch.object(memory_extractor, "API_BASE_URL", MAIN_URL), \
                patch.object(memory_extractor, "MEMORY_API_BASE_URL", MEMORY_URL):
            self.assertEqual(memory_extractor.get_memory_api_base_url(), MEMORY_URL)


class CallSiteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.urls = []

    def client_factory(self, content):
        def handler(request):
            self.urls.append(str(request.url))
            return httpx.Response(200, json=reply(content))
        return lambda **kwargs: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    def url_patches(self):
        return [
            patch.object(memory_extractor, "API_BASE_URL", MAIN_URL),
            patch.object(memory_extractor, "MEMORY_API_BASE_URL", MEMORY_URL),
            patch.object(memory_extractor, "MEMORY_API_KEY", "k"),
            patch.object(main, "API_BASE_URL", MAIN_URL),
            patch.object(main, "MEMORY_API_BASE_URL", MEMORY_URL),
            patch.object(main, "MEMORY_API_KEY", "k"),
        ]

    async def run_with(self, factory, coro):
        managers = self.url_patches() + [factory]
        for manager in managers:
            manager.start()
        try:
            with redirect_stdout(io.StringIO()):
                return await coro()
        finally:
            for manager in reversed(managers):
                manager.stop()

    async def test_extraction_and_scoring(self):
        factory = patch.object(memory_extractor.httpx, "AsyncClient", self.client_factory("[]"))
        await self.run_with(factory, lambda: memory_extractor.extract_memories(
            [{"role": "user", "content": "x", "source_id": 1}]))
        await self.run_with(factory, lambda: memory_extractor.score_memories(["x"]))
        self.assertEqual(self.urls, [MEMORY_URL, MEMORY_URL])

    async def test_rebuild_and_fix_share_the_memory_url(self):
        factory = patch.object(memory_rebuilder.httpx, "AsyncClient", self.client_factory("[]"))
        await self.run_with(factory, lambda: memory_rebuilder._post_json_array("p", "t"))
        self.assertEqual(self.urls, [MEMORY_URL])

    async def test_daily_consolidation(self):
        async def call():
            async with self.client_factory(json.dumps([]))() as client:
                return await main._post_consolidation_completion(client, "p", "m", 100, "整理")
        await self.run_with(patch.object(main, "TIMEZONE_HOURS", 8), call)
        self.assertEqual(self.urls, [MEMORY_URL])

    async def test_chat_still_uses_main_url(self):
        self.assertEqual(main._memory_api_base_url.__module__, "main")
        with patch.object(main, "API_BASE_URL", MAIN_URL), patch.object(main, "MEMORY_API_BASE_URL", ""):
            self.assertEqual(main._memory_api_base_url(), MAIN_URL)


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_save_hot_updates_and_clearing_falls_back(self):
        saved = {}

        async def set_config(key, value):
            saved[key] = value

        class Req:
            def __init__(self, body):
                self.body = body

            async def json(self):
                return self.body

        with patch.object(main, "set_gateway_config", set_config), \
                patch.object(main, "API_BASE_URL", MAIN_URL), \
                patch.object(main, "MEMORY_API_BASE_URL", ""), \
                patch.object(memory_extractor, "MEMORY_API_BASE_URL", ""), \
                patch.dict("os.environ", {}), redirect_stdout(io.StringIO()):
            result = await main.save_settings(Req({"MEMORY_API_BASE_URL": f"  {MEMORY_URL} "}))
            self.assertIn("MEMORY_API_BASE_URL", result["updated"])
            self.assertEqual(main.MEMORY_API_BASE_URL, MEMORY_URL)
            self.assertEqual(memory_extractor.get_memory_api_base_url(), MEMORY_URL)
            await main.save_settings(Req({"MEMORY_API_BASE_URL": ""}))
            self.assertEqual(saved["MEMORY_API_BASE_URL"], "")
            self.assertEqual(memory_extractor.get_memory_api_base_url(), MAIN_URL)


if __name__ == "__main__":
    unittest.main()
