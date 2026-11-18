"""阶段 6：记忆 MCP 服务（协议、开关、鉴权）+ 一轮完整的工具调用往返。"""

import asyncio
import copy
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import httpx

import main
import memory_mcp
import persona_guard

REAL_ASYNC_CLIENT = httpx.AsyncClient
SECRET = "s" * 40

MEMORIES = [
    {"id": 12, "content": "她说想 2027 年春天去京都看樱花", "importance": 7,
     "event_date": None, "created_at": datetime(2026, 10, 1, 2, 0), "kind": "plan"},
    {"id": 13, "content": "2026-11-17 和妈妈吵了一架", "importance": 6,
     "event_date": date(2026, 11, 17), "created_at": datetime(2026, 11, 17, 2, 0), "kind": "event"},
]


def rpc(method, params=None, message_id=1):
    message = {"jsonrpc": "2.0", "method": method}
    if message_id is not None:
        message["id"] = message_id
    if params is not None:
        message["params"] = params
    return message


class McpEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(main, "GATEWAY_SECRET", SECRET))
        self.stack.enter_context(patch.object(main, "MEMORY_ENABLED", True))
        self.stack.enter_context(patch.object(main, "MEMORY_MCP_ENABLED", True))
        self.stack.enter_context(patch.object(main, "MEMORY_MCP_WRITE_ENABLED", False))
        self.stack.enter_context(patch.object(main._db_module, "CONVERSATION_RECALL_ENABLED", False))
        self.stack.enter_context(patch.object(main, "_local_today", return_value=date(2026, 11, 19)))
        self.search = AsyncMock(return_value=MEMORIES)
        self.stack.enter_context(patch.object(main, "search_memories", self.search))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.client = REAL_ASYNC_CLIENT(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.stack.close()

    async def post(self, payload, key=SECRET):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return await self.client.post("/mcp", json=payload, headers=headers)

    async def test_requires_the_gateway_secret(self):
        response = await self.post(rpc("tools/list"), key="wrong")
        self.assertEqual(response.status_code, 401)
        response = await self.post(rpc("tools/list"), key=None)
        self.assertEqual(response.status_code, 401)

    async def test_refuses_to_run_without_a_gateway_secret(self):
        with patch.object(main, "GATEWAY_SECRET", ""):
            response = await self.post(rpc("tools/list"), key=None)
        self.assertEqual(response.status_code, 503)

    async def test_switch_off_hides_the_endpoint(self):
        with patch.object(main, "MEMORY_MCP_ENABLED", False):
            response = await self.post(rpc("tools/list"))
        self.assertEqual(response.status_code, 404)

    async def test_initialize_handshake(self):
        response = await self.post(rpc("initialize", {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "kelivo", "version": "1"},
        }))
        result = response.json()["result"]
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "pawwake-memory")
        response = await self.post(rpc("notifications/initialized", message_id=None))
        self.assertEqual(response.status_code, 202)
        response = await self.post(rpc("initialize", {"protocolVersion": "1999-01-01"}))
        self.assertEqual(response.json()["result"]["protocolVersion"], memory_mcp.LATEST_PROTOCOL_VERSION)

    async def test_tool_list_follows_switches(self):
        names = [t["name"] for t in (await self.post(rpc("tools/list"))).json()["result"]["tools"]]
        self.assertEqual(names, ["search_memory"])
        with patch.object(main._db_module, "CONVERSATION_RECALL_ENABLED", True), \
                patch.object(main, "MEMORY_MCP_WRITE_ENABLED", True):
            names = [t["name"] for t in (await self.post(rpc("tools/list"))).json()["result"]["tools"]]
        self.assertEqual(names, ["search_memory", "recall_conversation", "remember", "update_memory"])

    async def test_search_memory_returns_injection_format(self):
        response = await self.post(rpc("tools/call", {"name": "search_memory",
                                                      "arguments": {"query": "旅行", "limit": 99}}))
        result = response.json()["result"]
        self.assertFalse(result["isError"])
        text = result["content"][0]["text"]
        self.assertIn("- [记于 2026-10-01｜49 天前｜计划] 她说想 2027 年春天去京都看樱花", text)
        self.assertIn("- [2026-11-17｜2 天前｜事件] 2026-11-17 和妈妈吵了一架", text)
        self.assertNotIn("#12", text)
        self.search.assert_awaited_once_with("旅行", limit=30)

    async def test_disabled_write_tool_is_an_error_result(self):
        response = await self.post(rpc("tools/call", {"name": "remember",
                                                      "arguments": {"content": "x", "kind": "fact"}}))
        result = response.json()["result"]
        self.assertTrue(result["isError"])

    async def test_remember_goes_through_dedup_with_mcp_source(self):
        save = AsyncMock(return_value={"action": "inserted", "memory_id": 77})
        with patch.object(main, "MEMORY_MCP_WRITE_ENABLED", True), \
                patch.object(main, "save_memory_deduplicated", save):
            response = await self.post(rpc("tools/call", {"name": "remember", "arguments": {
                "content": "她 2026-12-20 考研初试", "kind": "plan", "event_date": "2026-12-20"}}))
        self.assertIn("#77", response.json()["result"]["content"][0]["text"])
        kwargs = save.await_args.kwargs
        self.assertEqual(kwargs["source_session"], "mcp")
        self.assertEqual(kwargs["kind"], "plan")
        self.assertEqual(kwargs["valid_until"], date(2026, 12, 23))

    async def test_bad_arguments_are_reported_to_the_model(self):
        with patch.object(main, "MEMORY_MCP_WRITE_ENABLED", True):
            response = await self.post(rpc("tools/call", {"name": "remember", "arguments": {
                "content": "x", "kind": "gossip"}}))
        self.assertTrue(response.json()["result"]["isError"])
        response = await self.post(rpc("tools/call", {"name": "search_memory", "arguments": {}}))
        self.assertTrue(response.json()["result"]["isError"])

    async def test_batch_and_unknown_method(self):
        response = await self.post([rpc("ping", message_id=1), rpc("nope", message_id=2)])
        body = response.json()
        self.assertEqual(body[0]["result"], {})
        self.assertEqual(body[1]["error"]["code"], -32601)

    async def test_get_has_no_server_stream(self):
        response = await self.client.get("/mcp", headers={"Authorization": f"Bearer {SECRET}"})
        self.assertEqual(response.status_code, 405)


class LegacySseTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_messages_are_answered_on_the_stream(self):
        session_id, queue = memory_mcp.SSE_SESSIONS.open()
        toolset = memory_mcp.MemoryToolset(
            search=AsyncMock(return_value="- [记于 2026-11-01｜18 天前] x"),
            recall=AsyncMock(), remember=AsyncMock(), update=AsyncMock(),
        )
        stream = memory_mcp.sse_stream(session_id, queue, f"/mcp/messages?session_id={session_id}")
        first = await stream.__anext__()
        self.assertIn(b"event: endpoint", first)
        self.assertIn(session_id.encode(), first)
        response = await memory_mcp.handle_payload(toolset, rpc("tools/call", {
            "name": "search_memory", "arguments": {"query": "x"}}))
        await queue.put(response)
        frame = await stream.__anext__()
        self.assertTrue(frame.startswith(b"event: message\ndata: "))
        payload = json.loads(frame.decode().split("data: ", 1)[1])
        self.assertIn("18 天前", payload["result"]["content"][0]["text"])
        await queue.put(None)
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()
        self.assertIsNone(memory_mcp.SSE_SESSIONS.get(session_id))


def completion(message):
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


class FakeRequest:
    def __init__(self, messages, tools):
        self.headers = {"X-Conversation-Id": "conv-1"}
        self._body = {"model": "gemini-test", "stream": False, "messages": messages, "tools": tools}

    async def json(self):
        return copy.deepcopy(self._body)


TOOLS = [{"type": "function", "function": {
    "name": "search_memory", "description": "查记忆",
    "parameters": memory_mcp.TOOL_SEARCH_MEMORY["inputSchema"]}}]

TOOL_CALL = {"id": "call_1", "type": "function",
             "function": {"name": "search_memory", "arguments": json.dumps({"query": "旅行"})}}


class ToolRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """模型调用 search_memory → 拿到结果 → 再回复：存档完整，提取只看 user/assistant 文字。"""

    async def test_search_memory_round_trip_archives_and_extracts_normally(self):
        saved = []
        upstream_bodies = []
        extraction_windows = []

        async def save_message(session_id, role, content, model, metadata=None):
            saved.append({"role": role, "content": content, "metadata": metadata})
            return 100 + len(saved)

        async def extract(messages, existing_memories=None):
            extraction_windows.append(messages)
            return [{"content": "她想 2027 年春天去京都看樱花", "importance": 7,
                     "source_ids": [101], "kind": "plan"}]

        replies = [
            completion({"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}),
            completion({"role": "assistant", "content": "你说过想明年春天去京都看樱花。"}),
        ]

        def upstream(request):
            upstream_bodies.append(json.loads(request.content))
            return httpx.Response(200, json=replies.pop(0))

        stored = AsyncMock(return_value={"action": "inserted", "memory_id": 900})
        background = []
        real_create_task = asyncio.create_task

        def track(coro):
            task = real_create_task(coro)
            background.append(task)
            return task

        stack = ExitStack()
        for name, value in {
            "CACHE_PARTITION_ENABLED": False, "MEMORY_ENABLED": True,
            "MEMORY_EXTRACT_ENABLED": True, "MEMORY_EXTRACT_INTERVAL": 1,
            "FORCE_STREAM": False, "REASONING_EFFORT": "",
        }.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="persona")))
        stack.enter_context(patch.object(main, "search_memories", AsyncMock(return_value=[])))
        stack.enter_context(patch.object(main, "get_core_memories", AsyncMock(return_value=[])))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", save_message))
        stack.enter_context(patch.object(main, "get_last_user_message_id", AsyncMock(return_value=101)))
        stack.enter_context(patch.object(main, "get_recent_memories", AsyncMock(return_value=[])))
        stack.enter_context(patch.object(main, "extract_memories", extract))
        stack.enter_context(patch.object(main, "save_memory_deduplicated", stored))
        stack.enter_context(patch.object(main, "get_active_memories_count", AsyncMock(return_value=1)))
        stack.enter_context(patch.object(persona_guard, "guard_applies_to_model", return_value=False))
        stack.enter_context(patch.object(main, "recent_quarantine_fingerprints", AsyncMock(return_value=set())))
        stack.enter_context(patch.object(main.asyncio, "create_task", track))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(upstream), **kwargs)))
        stack.enter_context(redirect_stdout(io.StringIO()))

        with stack:
            user = {"role": "user", "content": "我之前说想去哪旅行来着？"}
            first = await main._chat_completions_inner(FakeRequest([user], TOOLS))
            await asyncio.gather(*background)
            self.assertEqual(json.loads(first.body)["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                             "search_memory")
            # 第一个请求：工具调用还没结束，只存档、不提取
            self.assertEqual([m["role"] for m in saved], ["user", "assistant"])
            self.assertEqual(json.loads(saved[1]["metadata"])["tool_calls"][0]["id"], "call_1")
            self.assertEqual(extraction_windows, [])
            self.assertIn("search_memory", upstream_bodies[0]["messages"][0]["content"])

            tool_result = "- [记于 2026-10-01｜49 天前｜计划] 她说想 2027 年春天去京都看樱花"
            second = await main._chat_completions_inner(FakeRequest([
                user,
                {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
                {"role": "tool", "tool_call_id": "call_1", "name": "search_memory", "content": tool_result},
            ], TOOLS))
            await asyncio.gather(*background)

        self.assertEqual(second.status_code, 200)
        # 第二个请求：tool 结果和最终回复照常存档
        self.assertEqual([m["role"] for m in saved], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(saved[2]["content"], tool_result)
        self.assertEqual(json.loads(saved[2]["metadata"])["tool_call_id"], "call_1")
        # 提取窗口里没有 tool 消息、没有召回的旧记忆原文
        self.assertEqual(len(extraction_windows), 1)
        window = extraction_windows[0]
        self.assertEqual([m["role"] for m in window], ["user", "assistant"])
        self.assertTrue(all(tool_result not in m["content"] for m in window))
        self.assertEqual(window[0]["source_id"], 101)
        # 来源核对通过（引用了 user 消息），照常入库
        stored.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
