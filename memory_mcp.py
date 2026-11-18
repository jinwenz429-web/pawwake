"""记忆的 MCP 服务：让模型自己查记忆（必要时自己记）。

传输：
- Streamable HTTP：POST /mcp，一次请求一个 JSON-RPC 消息（或一批），直接回 JSON；
  不开服务端推送流，GET /mcp 回 405（规范允许）。
- 旧版 HTTP+SSE（兜底，给只支持 SSE 的客户端）：GET /mcp/sse 建流，第一条
  `endpoint` 事件给出 POST 地址 /mcp/messages?session_id=…，响应走 SSE 流回去。

鉴权由网关中间件统一做（Authorization: Bearer <GATEWAY_SECRET>）。

这个模块只管协议；具体的工具实现由 main.py 注入（main 以 `python main.py`
启动时模块名是 __main__，这里不能反过来 import main）。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

SERVER_NAME = "pawwake-memory"
SERVER_VERSION = "1.0.0"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

MEMORY_KINDS = ("fact", "state", "event", "plan", "promise")

SERVER_INSTRUCTIONS = (
    "这是你和对方之间的长期记忆库。网关每轮会自动带上一部分记忆，但只是一部分；"
    "想不起来、或需要确认某件事、某个日期、某个约定时，用 search_memory 查。"
)

TOOL_SEARCH_MEMORY = {
    "name": "search_memory",
    "description": (
        "查你和对方之间的长期记忆。每轮自动带上的记忆只是一部分，想不起来、"
        "或者需要确认某件事、日期、约定时调用。返回的每条都带日期、隔了几天和类型；"
        "已归档、已被新记忆取代的不会返回。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "想查的内容，用自然语言写，比如「她说过想去哪里旅行」"},
            "limit": {"type": "integer", "description": "最多返回几条，默认 8", "default": 8,
                      "minimum": 1, "maximum": 30},
        },
        "required": ["query"],
    },
}

TOOL_RECALL_CONVERSATION = {
    "name": "recall_conversation",
    "description": (
        "按内容检索过去的原始聊天片段（不是整理过的记忆，而是当时的原话）。"
        "需要确认对方当时具体怎么说的时候用。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "想找的对话内容"},
            "limit": {"type": "integer", "description": "最多返回几段，默认 3", "default": 3,
                      "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
}

TOOL_REMEMBER = {
    "name": "remember",
    "description": (
        "把一件值得长期记住的事写进记忆库。只记关于对方、或你们之间的约定；"
        "相对时间（明天、下周三）要换算成具体日期写进 content。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "一句完整、能独立看懂的话，保留具体细节"},
            "kind": {"type": "string", "enum": list(MEMORY_KINDS),
                     "description": "fact=长期事实/偏好, state=临时状态, event=发生过的事, plan=计划/将来的事, promise=你对对方的承诺"},
            "event_date": {"type": "string", "description": "事情发生或计划发生的日期 YYYY-MM-DD，可不填"},
        },
        "required": ["content", "kind"],
    },
}

TOOL_UPDATE_MEMORY = {
    "name": "update_memory",
    "description": (
        "用新的完整现状取代一条旧记忆（target_id 是 search_memory 结果里的记忆编号）。"
        "content 要写成完整的现状，例如「以前喜欢喝咖啡，2026-11 起改喝茶」。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "target_id": {"type": "integer", "description": "要取代的旧记忆编号"},
            "content": {"type": "string", "description": "新的完整现状"},
        },
        "required": ["target_id", "content"],
    },
}


class ToolInputError(ValueError):
    """参数不对：作为工具错误（isError）回给模型，而不是协议错误。"""


@dataclass
class MemoryToolset:
    """main.py 注入的工具实现和开关。"""

    search: Callable[[str, int], Awaitable[str]]
    recall: Callable[[str, int], Awaitable[str]]
    remember: Callable[[str, str, str | None], Awaitable[str]]
    update: Callable[[int, str], Awaitable[str]]
    recall_enabled: Callable[[], bool] = lambda: False
    write_enabled: Callable[[], bool] = lambda: False

    def tool_definitions(self) -> list:
        tools = [TOOL_SEARCH_MEMORY]
        if self.recall_enabled():
            tools.append(TOOL_RECALL_CONVERSATION)
        if self.write_enabled():
            tools.extend([TOOL_REMEMBER, TOOL_UPDATE_MEMORY])
        return tools


def _int_arg(arguments: dict, name: str, default: int, lower: int, upper: int) -> int:
    value = arguments.get(name, default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolInputError(f"{name} 必须是整数")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{name} 必须是整数")
    return max(lower, min(upper, value))


def _str_arg(arguments: dict, name: str, required: bool = True) -> str:
    value = arguments.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ToolInputError(f"缺少参数 {name}")
        return ""
    if not isinstance(value, str):
        raise ToolInputError(f"{name} 必须是字符串")
    return value.strip()


async def call_tool(toolset: MemoryToolset, name: str, arguments: dict) -> dict:
    """执行一个工具，返回 MCP 的 CallToolResult。"""
    if not isinstance(arguments, dict):
        arguments = {}
    available = {tool["name"] for tool in toolset.tool_definitions()}
    try:
        if name not in available:
            raise ToolInputError(f"工具 {name} 不存在或没有开启")
        if name == "search_memory":
            text = await toolset.search(
                _str_arg(arguments, "query"), _int_arg(arguments, "limit", 8, 1, 30),
            )
        elif name == "recall_conversation":
            text = await toolset.recall(
                _str_arg(arguments, "query"), _int_arg(arguments, "limit", 3, 1, 10),
            )
        elif name == "remember":
            kind = _str_arg(arguments, "kind").lower()
            if kind not in MEMORY_KINDS:
                raise ToolInputError(f"kind 只能是 {', '.join(MEMORY_KINDS)}")
            text = await toolset.remember(
                _str_arg(arguments, "content"), kind, _str_arg(arguments, "event_date", required=False) or None,
            )
        elif name == "update_memory":
            target_id = arguments.get("target_id")
            if isinstance(target_id, bool):
                raise ToolInputError("target_id 必须是整数")
            try:
                target_id = int(target_id)
            except (TypeError, ValueError):
                raise ToolInputError("target_id 必须是整数")
            text = await toolset.update(target_id, _str_arg(arguments, "content"))
        else:  # pragma: no cover - available 已经挡住
            raise ToolInputError(f"工具 {name} 不存在")
        return {"content": [{"type": "text", "text": text}], "isError": False}
    except ToolInputError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    except Exception as exc:
        print(f"⚠️ MCP 工具 {name} 执行失败: {type(exc).__name__}: {exc}", flush=True)
        return {"content": [{"type": "text", "text": f"{name} 执行失败，请稍后再试"}], "isError": True}


def _error(message_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


async def handle_message(toolset: MemoryToolset, message) -> dict | None:
    """处理一条 JSON-RPC 消息；通知（没有 id）返回 None。"""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Invalid Request")
    method = message.get("method")
    is_request = "id" in message
    message_id = message.get("id")
    if not isinstance(method, str):
        # 客户端发来的 response（我们不发请求，忽略）
        return None
    if not is_request:
        return None  # notifications/initialized、notifications/cancelled 等
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        return {"jsonrpc": "2.0", "id": message_id, "result": {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": SERVER_INSTRUCTIONS,
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": message_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": message_id, "result": {"tools": toolset.tool_definitions()}}
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return _error(message_id, -32602, "Invalid params")
        result = await call_tool(toolset, params["name"], params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": message_id, "result": result}
    if method in ("resources/list", "prompts/list"):
        key = "resources" if method.startswith("resources") else "prompts"
        return {"jsonrpc": "2.0", "id": message_id, "result": {key: []}}
    return _error(message_id, -32601, f"Method not found: {method}")


async def handle_payload(toolset: MemoryToolset, payload):
    """处理一次 POST 的内容（单条或批量）。返回要回给客户端的 JSON；全是通知时返回 None。"""
    if isinstance(payload, list):
        if not payload:
            return _error(None, -32600, "Invalid Request")
        responses = []
        for item in payload:
            response = await handle_message(toolset, item)
            if response is not None:
                responses.append(response)
        return responses or None
    return await handle_message(toolset, payload)


# ---------------- 旧版 HTTP+SSE 传输 ----------------

@dataclass
class SseSessions:
    queues: dict = field(default_factory=dict)

    def open(self) -> tuple[str, asyncio.Queue]:
        session_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        self.queues[session_id] = queue
        return session_id, queue

    def close(self, session_id: str) -> None:
        self.queues.pop(session_id, None)

    def get(self, session_id: str):
        return self.queues.get(session_id)


SSE_SESSIONS = SseSessions()
SSE_KEEPALIVE_SECONDS = 25


def sse_event(event: str, data: str) -> bytes:
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n".encode("utf-8")


async def sse_stream(session_id: str, queue: asyncio.Queue, endpoint: str, is_disconnected=None):
    try:
        yield sse_event("endpoint", endpoint)
        while True:
            if is_disconnected is not None and await is_disconnected():
                break
            try:
                message = await asyncio.wait_for(queue.get(), SSE_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
                continue
            if message is None:
                break
            yield sse_event("message", json.dumps(message, ensure_ascii=False))
    finally:
        SSE_SESSIONS.close(session_id)
