"""Voice calls from the Kelivo companion fork.

During a call Kelivo sends every turn with an ``X-Call-Id`` header. Such turns
are answered but never stored and never mined for memories:

* The first turn of a call builds a lean base context once (persona, core
  memories, a handful of relevant memories, the session summary and the last
  few chat rounds) and keeps it in memory for the rest of the call.
* Every turn sends that base plus the call so far (Kelivo carries the call
  transcript itself) and adds up to ``TURN_MEMORIES`` memories searched with
  what the user just said.

When the call ends Kelivo posts the whole transcript once. It is stored as a
single round (the transcript as the user message, a short marker as the
assistant reply) so later chats see the call, and memories are extracted from
it in one go. Each spoken line becomes its own extraction message, so only the
user's own lines can back a fact about the user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

CALL_ID_HEADER = "X-Call-Id"
RECENT_ROUNDS = 4
BASE_MEMORIES = 10
TURN_MEMORIES = 3
CONTEXT_TTL_SECONDS = 3 * 3600
MAX_CONTEXTS = 8
MAX_CALL_ID_LENGTH = 80
MAX_TRANSCRIPT_LINES = 600
MAX_LINE_CHARS = 2000

# Shown in history after the transcript, in place of a reply.
CALL_ENDED_REPLY = "（电话挂断了）"
# Extraction message ids for transcript lines. They never collide with real
# conversation row ids and are mapped back before anything is saved.
SYNTHETIC_ID_BASE = 2_000_000_000


@dataclass
class CallContext:
    session_id: str
    system_text: str
    summary_parts: list = field(default_factory=list)
    recent_messages: list = field(default_factory=list)
    memory_ids: set = field(default_factory=set)
    created_at: float = field(default_factory=time.monotonic)


class CallContextStore:
    def __init__(self, ttl_seconds: float = CONTEXT_TTL_SECONDS, max_entries: int = MAX_CONTEXTS):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: dict[str, CallContext] = {}

    def get(self, call_id: str, now: Optional[float] = None) -> Optional[CallContext]:
        now = time.monotonic() if now is None else now
        self._evict(now)
        return self._items.get(call_id)

    def put(self, call_id: str, context: CallContext, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._evict(now)
        self._items[call_id] = context
        while len(self._items) > self.max_entries:
            oldest = min(self._items, key=lambda key: self._items[key].created_at)
            del self._items[oldest]

    def pop(self, call_id: str) -> Optional[CallContext]:
        return self._items.pop(call_id, None)

    def _evict(self, now: float) -> None:
        expired = [
            key for key, ctx in self._items.items()
            if now - ctx.created_at > self.ttl_seconds
        ]
        for key in expired:
            del self._items[key]


CONTEXTS = CallContextStore()


def call_id_from_headers(headers) -> str:
    raw = (headers.get(CALL_ID_HEADER) or "").strip()
    if not raw or len(raw) > MAX_CALL_ID_LENGTH:
        return ""
    return raw


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def recent_history(messages: Iterable[dict], rounds: int = RECENT_ROUNDS) -> list:
    """The last ``rounds`` chat rounds as plain user/assistant text messages."""
    plain = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_of(message.get("content")).strip()
        if not text:
            continue
        plain.append({"role": role, "content": text})
    if rounds <= 0:
        return []
    starts = [i for i, m in enumerate(plain) if m["role"] == "user"]
    if not starts:
        return plain[-2 * rounds:]
    cut = starts[-rounds] if len(starts) >= rounds else starts[0]
    return plain[cut:]


def call_messages(messages: Iterable[dict]) -> list:
    """Non-system messages of the call.

    Neighbouring plain user/assistant lines are merged. A memory lookup during
    the call (assistant tool_calls + tool results) is kept exactly as sent.
    """
    merged: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "tool" or (role == "assistant" and message.get("tool_calls")):
            merged.append(dict(message))
            continue
        if role not in ("user", "assistant"):
            continue
        text = _text_of(message.get("content")).strip()
        if not text:
            continue
        last = merged[-1] if merged else None
        if last and last["role"] == role and not last.get("tool_calls"):
            last["content"] = f"{last['content']}\n{text}"
        else:
            merged.append({"role": role, "content": text})
    return merged


def ends_awaiting_reply(messages: list) -> bool:
    """A call turn ends with the user (or a hidden cue) or with tool results."""
    return bool(messages) and messages[-1].get("role") in ("user", "tool")


def latest_user_text(messages: Iterable[dict]) -> str:
    last = ""
    for message in messages:
        if message.get("role") == "user":
            last = _text_of(message.get("content"))
    return last.strip()


def memory_block(lines: list, heading: str) -> str:
    if not lines:
        return ""
    return f"<{heading}>\n" + "\n".join(lines) + f"\n</{heading}>"


def build_messages(
    context: CallContext,
    call_msgs: list,
    time_text: str,
    turn_memory_text: str = "",
) -> list:
    """Upstream messages for one call turn."""
    result = []
    if context.system_text:
        result.append({"role": "system", "content": context.system_text})
    if context.summary_parts:
        summary = "\n\n".join(["[以下是之前对话的摘要，帮助你回忆上下文]", *context.summary_parts])
        result.append({"role": "user", "content": summary})
        result.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})
    history = [dict(m) for m in context.recent_messages]
    # The call always opens with a user turn; keep roles alternating.
    if history and call_msgs and history[-1]["role"] == call_msgs[0]["role"] == "user":
        history.pop()
    result.extend(history)
    body = [dict(m) for m in call_msgs]
    last_user = max((i for i, m in enumerate(body) if m["role"] == "user"), default=None)
    if last_user is not None and last_user == len(body) - 1:
        parts = [part for part in (time_text, turn_memory_text) if part]
        if parts:
            body[last_user]["content"] = "\n\n".join(parts + [body[last_user]["content"]])
    result.extend(body)
    return result


CALL_SEARCH_HINT = (
    "# 电话里查记忆\n"
    "- 每句话网关已经自动带上了几条相关记忆，大多数时候直接聊就好。\n"
    "- 真想不起来、或者要确认某个日期、某个约定时，可以调用 search_memory 去查；"
    "查的时候对方会先听到一句垫话，所以别查个没完，一轮最多查一次。"
)


# ---------- transcript ----------

def _parse_time(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class TranscriptError(ValueError):
    pass


@dataclass
class Transcript:
    lines: list  # [{"role", "text", "at": datetime | None}]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    user_label: str
    assistant_label: str


def parse_transcript(payload: dict) -> Transcript:
    if not isinstance(payload, dict):
        raise TranscriptError("body must be an object")
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise TranscriptError("lines must be a non-empty list")
    if len(raw_lines) > MAX_TRANSCRIPT_LINES:
        raise TranscriptError("too many lines")
    lines = []
    for item in raw_lines:
        if not isinstance(item, dict):
            raise TranscriptError("each line must be an object")
        role = item.get("role")
        if role not in ("user", "assistant"):
            raise TranscriptError("role must be user or assistant")
        text = str(item.get("text") or "").strip()[:MAX_LINE_CHARS]
        if text:
            lines.append({"role": role, "text": text, "at": _parse_time(item.get("at"))})
    if not any(line["role"] == "user" for line in lines) and not any(
        line["role"] == "assistant" for line in lines
    ):
        raise TranscriptError("transcript is empty")
    user_label = str(payload.get("user_label") or "").strip()[:20] or "我"
    assistant_label = str(payload.get("assistant_label") or "").strip()[:20] or "你"
    return Transcript(
        lines=lines,
        started_at=_parse_time(payload.get("started_at")),
        ended_at=_parse_time(payload.get("ended_at")),
        user_label=user_label,
        assistant_label=assistant_label,
    )


def _local(moment: datetime, tz_hours: float) -> datetime:
    return moment.astimezone(timezone(timedelta(hours=tz_hours)))


def transcript_header(transcript: Transcript, tz_hours: float) -> str:
    start = transcript.started_at
    end = transcript.ended_at
    if start and end and end >= start:
        minutes = max(1, round((end - start).total_seconds() / 60))
        s, e = _local(start, tz_hours), _local(end, tz_hours)
        span = f"{s:%Y-%m-%d %H:%M}–{e:%H:%M}" if s.date() == e.date() else f"{s:%Y-%m-%d %H:%M}–{e:%Y-%m-%d %H:%M}"
        return f"[语音通话 {span}，{minutes} 分钟]"
    if start:
        return f"[语音通话 {_local(start, tz_hours):%Y-%m-%d %H:%M}]"
    return "[语音通话]"


def stored_round(transcript: Transcript, tz_hours: float) -> tuple[str, str]:
    """(user content, assistant content) stored for the call."""
    body = "\n".join(
        f"{transcript.user_label if line['role'] == 'user' else transcript.assistant_label}：{line['text']}"
        for line in transcript.lines
    )
    return f"{transcript_header(transcript, tz_hours)}\n{body}", CALL_ENDED_REPLY


def extraction_messages(transcript: Transcript) -> tuple[list, dict]:
    """One extraction message per spoken line with its own synthetic id.

    Returns the messages and a map from synthetic id to the role that spoke,
    so verified source ids can be mapped back to the stored rows.
    """
    messages = []
    roles = {}
    for index, line in enumerate(transcript.lines):
        synthetic_id = SYNTHETIC_ID_BASE + index
        roles[synthetic_id] = line["role"]
        message = {"role": line["role"], "content": line["text"], "source_id": synthetic_id}
        if line["at"] is not None:
            message["created_at"] = line["at"]
        messages.append(message)
    return messages, roles


def map_source_ids(ids: Iterable[int], roles: dict, user_row_id, assistant_row_id) -> list:
    mapped = set()
    for source_id in ids:
        role = roles.get(source_id)
        if role == "user" and user_row_id is not None:
            mapped.add(int(user_row_id))
        elif role == "assistant" and assistant_row_id is not None:
            mapped.add(int(assistant_row_id))
        elif role is None:
            mapped.add(int(source_id))
    return sorted(mapped)

