"""Kelivo companion fork: proactive turns, diaries and persona-guard verdicts.

Kelivo (jinwenz429-web/kelivo, branch companion-apk) talks to Pawwake
directly. When it wakes on its own it ends the request with a hidden user
turn that starts with ``PROACTIVE_CUE_OPENING``; that cue is never something
the user said, so it must not be stored or mined for memories. The reply may
be only ``[静默]`` (stay quiet) and may carry ``[DIARY]...[/DIARY]`` blocks,
which belong in the diary table rather than the chat log.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

# Must match `proactiveCueOpening` in Kelivo's message_builder_service.dart.
PROACTIVE_CUE_OPENING = "（这一条不是对方发来的消息，是只有你能看到的提示。）"

_SILENCE = re.compile(r"^\s*[\[【](静默|silent|silence)[\]】]", re.IGNORECASE)
_DIARY = re.compile(r"\[DIARY\](.*?)\[/DIARY\]", re.IGNORECASE | re.DOTALL)

DIARY_CONTEXT_ENTRIES = 3
DIARY_CONTEXT_DAYS = 7
DIARY_CONTEXT_CHARS = 300


def is_proactive_cue(text: str) -> bool:
    return isinstance(text, str) and text.lstrip().startswith(PROACTIVE_CUE_OPENING)


# Kelivo only appends its wake cue on the first try. Regenerating a failed or
# unwanted proactive message rebuilds the request without it, so the request
# ends with the model's own previous reply, which Gemini rejects with
# "Requests ending with a model turn are not supported".
CONTINUATION_CUE = (
    PROACTIVE_CUE_OPENING
    + "对方没有发新消息。这一轮还是你主动开口（多半是刚才那次没发出去，重来一次）。"
    "顺着上面的对话自然地说，别提这条提示。"
)


def ends_with_model_turn(messages: list) -> bool:
    """True when the last non-system message is a finished assistant reply."""
    for message in reversed(messages or []):
        role = message.get("role")
        if role == "system":
            continue
        return role == "assistant" and not message.get("tool_calls")
    return False


def is_silence(text: str) -> bool:
    return bool(_SILENCE.match(text or ""))


def split_diary(text: str) -> tuple[str, list[str]]:
    """Return the reply without diary blocks, and the non-empty diary bodies."""
    diaries = [body.strip() for body in _DIARY.findall(text or "") if body.strip()]
    rest = _DIARY.sub("", text or "").strip()
    return rest, diaries


def diary_context(entries: list[dict]) -> str:
    """Recent diary entries, oldest first, for a wake cue."""
    if not entries:
        return ""
    lines = ["你最近写下的日记（只有你自己看得到，不用复述给对方）："]
    for entry in sorted(entries, key=lambda e: e.get("created_at") or 0):
        created = entry.get("created_at")
        stamp = created.strftime("%m月%d日 %H:%M") if hasattr(created, "strftime") else ""
        body = str(entry.get("content") or "").strip().replace("\n", " ")
        if len(body) > DIARY_CONTEXT_CHARS:
            body = body[:DIARY_CONTEXT_CHARS] + "…"
        lines.append(f"- {stamp} {body}".rstrip())
    return "\n".join(lines)


class GuardVerdicts:
    """Short-lived persona-guard verdicts keyed by the client's request id.

    Kelivo asks after a streamed reply whether it broke character. The verdict
    is recorded before the stream's final frame, so it is normally already
    there; ``get`` still waits briefly to absorb ordering jitter.
    """

    TTL_SECONDS = 600

    def __init__(self) -> None:
        self._items: dict[str, tuple[float, str]] = {}
        self._events: dict[str, asyncio.Event] = {}

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.TTL_SECONDS
        for key in [k for k, (at, _) in self._items.items() if at < cutoff]:
            self._items.pop(key, None)
            self._events.pop(key, None)

    def put(self, request_id: str, status: str) -> None:
        if not request_id:
            return
        self._prune()
        self._items[request_id] = (time.monotonic(), status)
        event = self._events.setdefault(request_id, asyncio.Event())
        event.set()

    async def get(self, request_id: str, timeout: float = 3.0) -> Optional[str]:
        if request_id in self._items:
            return self._items[request_id][1]
        event = self._events.setdefault(request_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        item = self._items.get(request_id)
        return item[1] if item else None


VERDICTS = GuardVerdicts()


def auto_regenerate_enabled() -> bool:
    """Whether a confirmed drift may be answered with one regeneration.

    ``PERSONA_GUARD_MODE=observe`` keeps the guard to logging and quarantine.
    """
    import os

    mode = os.getenv("PERSONA_GUARD_MODE", "conservative").strip().lower()
    flag = os.getenv("PERSONA_GUARD_AUTO_REGENERATE", "true").strip().lower()
    return mode != "observe" and flag not in {"0", "false", "no", "off"}
