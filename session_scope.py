"""对话线分区：主线 / 剧情 / 工具。

- 主线（默认）：所有主线对话共用一个记忆池。
- 剧情：能读主线记忆；剧情里提取的记忆只记在这条线自己名下（memories.scope_session），
  只在这条线里自动召回。别的线不自动注入，但 MCP 搜索能搜到，并标明出自哪段剧情。
- 工具：完全不提取记忆。

用户在一条对话的消息开头写 `#剧情 猫妖篇`（或 `#工具`）就能给这条线打标签；
标记会从消息里剥掉，不发给模型、不存进对话记录。
"""
from __future__ import annotations

import contextvars
import re
import time

MAIN = "main"
STORY = "story"
TOOL = "tool"
KINDS = (MAIN, STORY, TOOL)
KIND_LABELS = {MAIN: "主线", STORY: "剧情", TOOL: "工具"}

# 召回范围：None = 只看主线；某个 session_id = 主线 + 这条剧情线；ALL = 全部（MCP 搜索用，结果带标签）
ALL = object()
CURRENT_SCOPE: contextvars.ContextVar = contextvars.ContextVar("memory_scope", default=None)

_MARKER = re.compile(
    r"^\s*[#＃]\s*(剧情|工具)[ \t　]*([^\n]{0,40}?)[ \t　]*(?:\n|$)"
)
_SAFE_SESSION = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


def parse_marker(text):
    """消息开头的 #剧情 名字 / #工具。返回 (kind, name, 去掉标记后的文字)；没有标记返回 None。"""
    if not isinstance(text, str):
        return None
    match = _MARKER.match(text)
    if not match:
        return None
    kind = STORY if match.group(1) == "剧情" else TOOL
    name = match.group(2).strip() or None
    return kind, name, text[match.end():].lstrip("\n")


def strip_marker_from_message(message: dict):
    """就地剥掉一条 user 消息开头的标记（字符串或多段内容都处理）。返回解析结果或 None。"""
    content = message.get("content")
    if isinstance(content, str):
        parsed = parse_marker(content)
        if parsed:
            message["content"] = parsed[2]
        return parsed
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parsed = parse_marker(part.get("text"))
                if parsed:
                    part["text"] = parsed[2]
                return parsed
    return None


def safe_session_literal(session_id):
    """把 session_id 变成能直接拼进 SQL 的字面量；不合规就返回 None（按只看主线处理）。"""
    if not isinstance(session_id, str) or not _SAFE_SESSION.match(session_id):
        return None
    return "'" + session_id.replace("'", "''") + "'"


def scope_sql(column: str = "scope_session") -> str:
    """按当前召回范围拼一段 ` AND ...`，直接接在 WHERE 后面。"""
    scope = CURRENT_SCOPE.get()
    if scope is ALL:
        return ""
    literal = safe_session_literal(scope) if scope else None
    if literal:
        return f" AND ({column} IS NULL OR {column} = {literal})"
    return f" AND {column} IS NULL"


def scope_for(tag) -> object:
    """根据一条线的标签决定召回范围。"""
    if tag and tag.get("kind") == STORY:
        return tag["session_id"]
    return None


# 最近一次聊天请求属于哪条线：MCP 工具调用不带对话编号，靠它推断「记一条」该记在哪
_last_chat = {"session_id": None, "at": 0.0}
LAST_CHAT_WINDOW_SECONDS = 300


def note_chat_session(session_id):
    _last_chat["session_id"] = session_id
    _last_chat["at"] = time.monotonic()


def recent_chat_session():
    if _last_chat["session_id"] and time.monotonic() - _last_chat["at"] <= LAST_CHAT_WINDOW_SECONDS:
        return _last_chat["session_id"]
    return None
