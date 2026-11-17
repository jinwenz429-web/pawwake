"""Persona Continuity Guard（Pawwake 侧）。

与 Dylan 的 persona_guard.js 共用 persona_guard_rules.json 和 test/fixtures/persona_guard_cases.json，
判定逻辑一一对应。这里只负责“记忆写入闸门”：
  - CONFIRMED_DRIFT：不写入 conversations（因而不进摘要/召回/日后的上下文），只在隔离表留指纹；不提取记忆
  - SUSPECTED_DRIFT / SAFETY_REFUSAL：对话照常保存，但本轮不做长期记忆提取
  - 恢复请求（Dylan 发来的 X-Persona-Guard-Stage≠initial）：等首轮落库后，把被采纳的回复接到同一条 user 消息后面
纯代码，无任何模型调用。总开关 PERSONA_GUARD_ENABLED 默认关闭。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

_RULES_PATH = Path(__file__).with_name("persona_guard_rules.json")
RULES = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
RULES_VERSION = RULES.get("version")

NORMAL = "NORMAL"
SUSPECTED_DRIFT = "SUSPECTED_DRIFT"
CONFIRMED_DRIFT = "CONFIRMED_DRIFT"
SAFETY_REFUSAL = "SAFETY_REFUSAL"
UNKNOWN = "UNKNOWN"


def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns or []]


_IDENTITY = [dict(rule, regexes=_compile(rule["patterns"])) for rule in RULES["identity"]]
_MECHANICAL = [dict(rule, regexes=_compile(rule["patterns"])) for rule in RULES["mechanical"]]
_REFUSAL = _compile(RULES["refusal"])
_AI_TOPIC = _compile(RULES["userExemption"]["aiTopic"])
_BREAK = _compile(RULES["userExemption"]["breakCharacter"])
_TECH = _compile(RULES["userExemption"]["technical"])
_SAFETY_FINISH = {v.upper() for v in RULES["safetyFinishReasons"]}


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def guard_enabled() -> bool:
    return _env_bool("PERSONA_GUARD_ENABLED", False)


def guard_applies_to_model(model: str) -> bool:
    if not guard_enabled():
        return False
    if not _env_bool("PERSONA_GUARD_GEMINI_ONLY", True):
        return True
    return "gemini" in str(model or "").lower()


def ai_self_aware_override():
    raw = os.getenv("PERSONA_GUARD_AI_SELF_AWARE_PERSONA", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(str(part.get("text") or part.get("content") or ""))
        return "\n".join(out)
    return ""


def _any(regexes, text: str) -> bool:
    return any(r.search(text) for r in regexes)


def _ai_self_aware(recent_assistant_texts, override):
    if override is True or override is False:
        return override
    if not recent_assistant_texts:
        return False
    rules = [r for r in _IDENTITY if r.get("discountWhenAiSelfAware")]
    hits = [t for t in recent_assistant_texts if any(_any(r["regexes"], t) for r in rules)]
    return len(hits) >= 2 or (len(hits) >= 1 and len(recent_assistant_texts) <= 2)


_STRUCTURE = re.compile(r"(^|\n)\s*(#{1,4}\s|\d+[.、]\s|[-*]\s)")


def _structured(text: str) -> bool:
    return bool(_STRUCTURE.search(text)) and len(text.split("\n")) >= 4


def detect_drift(
    reply: str,
    user_message: str = "",
    recent_messages=None,
    finish_reason: str = "",
    upstream_blocked: bool = False,
    has_tool_calls: bool = False,
    roleplay: bool = True,
    ai_self_aware=None,
) -> dict:
    started = time.perf_counter()
    reply = str(reply or "").strip()
    user_message = str(user_message or "")
    rules: list[str] = []

    def done(status, score=0):
        return {
            "status": status,
            "score": score,
            "rules": rules,
            "allow_auto_recovery": status == CONFIRMED_DRIFT,
            "allow_conversation_log": status != CONFIRMED_DRIFT,
            "allow_memory": status == NORMAL,
            "allow_context": status != CONFIRMED_DRIFT,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    if upstream_blocked or str(finish_reason or "").upper() in _SAFETY_FINISH:
        rules.append("S0_upstream_safety_signal")
        return done(SAFETY_REFUSAL)
    if has_tool_calls:
        rules.append("U0_tool_call_turn")
        return done(UNKNOWN)
    if not reply:
        rules.append("U1_empty_reply")
        return done(UNKNOWN)
    if roleplay is False:
        rules.append("U2_not_roleplay")
        return done(NORMAL)
    if _any(_REFUSAL, reply):
        rules.append("S1_request_refusal_language")
        return done(SAFETY_REFUSAL)
    if _any(_BREAK, user_message):
        rules.append("X1_user_break_character")
        return done(NORMAL)
    user_ai_topic = _any(_AI_TOPIC, user_message)
    user_technical = _any(_TECH, user_message)

    recent = [m for m in (recent_messages or []) if isinstance(m, dict)]
    recent_assistant = [
        text_of(m.get("content")) for m in recent
        if m.get("role") == "assistant" and not m.get("tool_calls")
    ]
    recent_assistant = [t for t in recent_assistant if t][-6:]
    aware = _ai_self_aware(recent_assistant, ai_self_aware)

    identity_score = 0
    mechanical_score = 0
    identity_hits = set()
    for rule in _IDENTITY:
        if not _any(rule["regexes"], reply):
            continue
        weight = rule["weight"]
        if user_ai_topic:
            weight = 0
        elif aware and rule.get("discountWhenAiSelfAware"):
            weight = 0
        if weight > 0:
            identity_score += weight
            identity_hits.add(rule["id"])
            rules.append(rule["id"])
        else:
            rules.append(f"{rule['id']}:discounted")

    if not user_technical:
        for rule in _MECHANICAL:
            if _any(rule["regexes"], reply):
                mechanical_score += rule["weight"]
                rules.append(rule["id"])
        if len(recent_assistant) >= 2:
            if "您" in reply and not any("您" in t for t in recent_assistant) and "您" not in user_message:
                mechanical_score += 1
                rules.append("B3_register_shift_formal_you")
            if _structured(reply) and not any(_structured(t) for t in recent_assistant) and len(user_message) < 120:
                mechanical_score += 1
                rules.append("B4_structure_shift")

    score = identity_score + mechanical_score
    strong = "A1_ai_self_label" in identity_hits or "A3_relationship_denial" in identity_hits
    other = len(identity_hits) + (1 if mechanical_score > 0 else 0)
    if strong and (other >= 2 or identity_score >= 5):
        rules.append("C1_continuity_break")
        return done(CONFIRMED_DRIFT, score)
    if identity_score >= 2 or (identity_score >= 1 and mechanical_score >= 1) or mechanical_score >= 3:
        return done(SUSPECTED_DRIFT, score)
    return done(NORMAL, score)


_TS_PREFIX = re.compile(r"^（?\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]?)\d{1,2}[:：]\d{2}[）\s]*")


def fingerprint_text(text: str) -> str:
    """与 Dylan persona_quarantine.js 的 fingerprintText 完全一致。"""
    normalized = re.sub(r"\s+", "", _TS_PREFIX.sub("", str(text or ""))).strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def filter_extraction_messages(messages: list) -> list:
    """记忆提取窗口里可能带着之前轮次的可疑回复：非 NORMAL 的 assistant 消息不交给提取模型。"""
    kept = []
    override = ai_self_aware_override()
    for index, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            before = messages[:index]
            last_user = next((m.get("content", "") for m in reversed(before) if m.get("role") == "user"), "")
            verdict = detect_drift(msg.get("content", ""), last_user, before, ai_self_aware=override)
            if verdict["status"] in {CONFIRMED_DRIFT, SUSPECTED_DRIFT}:
                continue
        kept.append(msg)
    return kept


def filter_confirmed_drift(messages: list) -> list:
    """摘要输入：排除 guard 启用前就已经落库的高置信漂移回复。"""
    kept = []
    override = ai_self_aware_override()
    for index, msg in enumerate(messages):
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            before = messages[:index]
            last_user = next((text_of(m.get("content")) for m in reversed(before) if m.get("role") == "user"), "")
            verdict = detect_drift(text_of(msg.get("content")), last_user, before, ai_self_aware=override)
            if verdict["status"] == CONFIRMED_DRIFT:
                continue
        kept.append(msg)
    return kept


class AttemptRegistry:
    """同一个 Dylan request_id 的首轮与恢复轮之间的协调（单进程内）。

    首轮的后台落库可能晚于恢复请求到达，恢复轮必须等首轮落库后再接续，
    否则会重复保存 user 消息或写反顺序。条目 10 分钟过期。
    """

    TTL_SECONDS = 600

    def __init__(self):
        self._entries: dict[str, dict] = {}

    def _prune(self):
        cutoff = time.monotonic() - self.TTL_SECONDS
        for key in [k for k, v in self._entries.items() if v["created"] < cutoff]:
            self._entries.pop(key, None)

    def _entry(self, request_id: str) -> dict:
        self._prune()
        entry = self._entries.get(request_id)
        if entry is None:
            entry = {"created": time.monotonic(), "event": asyncio.Event(), "result": None}
            self._entries[request_id] = entry
        return entry

    def expect(self, request_id: str):
        if request_id:
            self._entry(request_id)

    def resolve(self, request_id: str, result: dict):
        if not request_id:
            return
        entry = self._entry(request_id)
        entry["result"] = result
        entry["event"].set()

    async def wait(self, request_id: str, timeout: float = 30.0):
        if not request_id:
            return None
        entry = self._entry(request_id)
        try:
            await asyncio.wait_for(entry["event"].wait(), timeout)
        except asyncio.TimeoutError:
            return None
        return entry["result"]


ATTEMPTS = AttemptRegistry()


def guard_request_meta(headers) -> dict:
    request_id = str(headers.get("X-Persona-Guard-Request-Id", "") or "").strip()[:64]
    stage = str(headers.get("X-Persona-Guard-Stage", "") or "").strip().lower()[:32]
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", request_id or ""):
        request_id = ""
    if stage not in {"initial", "regenerate", "context_repair", "fallback"}:
        stage = "initial" if request_id else ""
    return {"request_id": request_id, "stage": stage}


def log_event(**fields):
    """只打印状态与规则，不打印正文。"""
    allowed = {"status", "rules", "latency_ms", "action", "stage", "request_id", "session_hash", "rules_version"}
    clean = {k: v for k, v in fields.items() if k in allowed}
    clean["event"] = "persona_guard"
    print(json.dumps(clean, ensure_ascii=False), flush=True)
