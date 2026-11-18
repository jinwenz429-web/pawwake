"""修补旧记忆：先预览，确认后才写，写了能撤回。

两种模式：
- time：活跃记忆里含相对时间词（昨天、下周、最近……）的，按每条的 created_at
  （本地时区）让模型改写成具体日期，顺带补上缺的 kind / event_date。
- kind：老记忆里没有 kind 的，批量补 kind（顺带补 event_date），不改内容。

应用时原地更新，旧值记进 memory_edit_log，能整份撤回；内容改了的重算向量。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import date, datetime

import database as _db
import memory_extractor as _me
from database import get_pool
from memory_rebuilder import _jsonb_object, _post_json_array

FIX_BATCH_SIZE = 20
FIX_MODES = ("time", "kind")

RELATIVE_TIME_WORDS = (
    "大前天", "大后天", "前天", "后天", "昨天", "今天", "明天",
    "昨晚", "今晚", "明晚", "今早", "明早", "昨早", "今天早上", "明天早上",
    "上周", "下周", "这周", "本周", "上礼拜", "下礼拜", "这礼拜", "上星期", "下星期", "这星期",
    "上个月", "下个月", "这个月", "本月", "周末",
    "最近", "刚才", "刚刚", "前几天", "这几天", "过几天", "前阵子", "这阵子",
    "去年", "今年", "明年",
)
RELATIVE_TIME_RE = re.compile("|".join(sorted(map(re.escape, RELATIVE_TIME_WORDS), key=len, reverse=True)))

_status = {"running": False, "mode": None, "phase": None, "processed": 0, "total": 0,
           "plan_id": None, "error": None}


class FixConflict(Exception):
    """方案已经应用 / 撤回过，或者不能再应用。"""


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _as_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def has_relative_time(text: str) -> bool:
    return bool(RELATIVE_TIME_RE.search(text or ""))


async def ensure_fix_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_fix_plans (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'preview',
                items JSONB NOT NULL,
                apply_result JSONB,
                applied_at TIMESTAMPTZ,
                reverted_at TIMESTAMPTZ
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_edit_log (
                id BIGSERIAL PRIMARY KEY,
                plan_id BIGINT NOT NULL,
                memory_id INTEGER NOT NULL,
                old_content TEXT,
                new_content TEXT,
                old_kind TEXT,
                new_kind TEXT,
                old_event_date DATE,
                new_event_date DATE,
                old_valid_until DATE,
                new_valid_until DATE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reverted_at TIMESTAMPTZ
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_edit_log_plan ON memory_edit_log (plan_id)"
        )


TIME_FIX_PROMPT = """你在修补一个记忆库里的时间写法。每条记忆都带 recorded_at：记下这条记忆时的本地时间。

任务：
1. 把 content 里的相对时间（今天、昨天、前天、明天、后天、下周三、这周末、上个月、最近、刚才、今晚、去年……）
   按 recorded_at 换算成具体日期写进去，例如 recorded_at 是 2026-11-18 周三，「昨天和妈妈吵架」→
   「2026-11-17 和妈妈吵架」；「下周三考试」→「2026-11-25 考试」；「最近在赶论文」→「截至 2026-11-18 在赶论文」。
2. 除了时间写法，其他文字一字不改：不要润色、不要压缩、不要补充信息。
3. 换算不出来的（比如「去年」只能确定到年份）就写到能确定的程度（「2025 年」）。
4. 顺带给出 kind 和 event_date：
   kind：fact（长期事实/偏好）| state（临时状态）| event（发生过的事）| plan（计划/将来的事）| promise（AI 对对方的承诺）
   event_date：事情发生或计划发生的日期 YYYY-MM-DD，说不清就填 null。

每个输入 id 输出且只输出一次。只输出 JSON 数组：
[{"id": 1, "content": "改写后的完整内容", "kind": "event", "event_date": "2026-11-17"}]

输入：
{items}
"""

KIND_FIX_PROMPT = """你在给一个记忆库里的老记忆补类型。每条记忆带 recorded_at（记录时的本地时间）。
不要改 content，只判断：
- kind：fact（长期事实/偏好）| state（临时状态）| event（发生过的事）| plan（计划/将来的事）| promise（AI 对对方的承诺）
- event_date：事情发生或计划发生的日期 YYYY-MM-DD；内容里写了具体日期就用它，说不清就填 null

每个输入 id 输出且只输出一次。只输出 JSON 数组：
[{"id": 1, "kind": "fact", "event_date": null}]

输入：
{items}
"""


async def _load_candidates(mode: str) -> list:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if mode == "kind":
            rows = await conn.fetch("""
                SELECT id, content, created_at, kind, event_date, valid_until
                FROM memories
                WHERE is_active = TRUE AND kind IS NULL
                ORDER BY id
            """)
        else:
            rows = await conn.fetch("""
                SELECT id, content, created_at, kind, event_date, valid_until
                FROM memories
                WHERE is_active = TRUE
                ORDER BY id
            """)
    items = [dict(row) for row in rows]
    if mode == "time":
        items = [item for item in items if has_relative_time(item["content"])]
    return items


def build_plan_items(mode: str, sources: list, results: list) -> list:
    """把模型结果和原记忆对齐，只留下真的有变化的条目。"""
    by_id = {int(item["id"]): item for item in sources}
    items = []
    seen = set()
    for raw in results:
        if not isinstance(raw, dict):
            continue
        try:
            memory_id = int(raw.get("id"))
        except (TypeError, ValueError):
            continue
        source = by_id.get(memory_id)
        if source is None or memory_id in seen:
            continue
        seen.add(memory_id)
        old_content = source["content"] or ""
        new_content = old_content
        if mode == "time":
            proposed = str(raw.get("content") or "").strip()
            if proposed:
                new_content = proposed
        kind = str(raw.get("kind") or "").strip().lower()
        kind = kind if kind in _db.MEMORY_KINDS else None
        old_kind = source.get("kind")
        old_event = _as_date(source.get("event_date"))
        new_kind = old_kind or kind              # 只补空的，不覆盖已有的
        new_event = old_event or _as_date(raw.get("event_date"))
        if new_content == old_content and new_kind == old_kind and new_event == old_event:
            continue
        items.append({
            "id": memory_id,
            "content_hash": _hash(old_content),
            "recorded_at": _me.format_local_stamp(source["created_at"]) if source.get("created_at") else None,
            "old_content": old_content,
            "new_content": new_content,
            "old_kind": old_kind,
            "new_kind": new_kind,
            "old_event_date": old_event.isoformat() if old_event else None,
            "new_event_date": new_event.isoformat() if new_event else None,
        })
    return items


async def _ask_model(mode: str, batch: list) -> list:
    payload = [{
        "id": int(item["id"]),
        "recorded_at": _me.format_local_stamp(item["created_at"]) if item.get("created_at") else "未知",
        "content": item["content"],
    } for item in batch]
    prompt = (TIME_FIX_PROMPT if mode == "time" else KIND_FIX_PROMPT).replace("{items}", _json_text(payload))
    return await _post_json_array(prompt, "修补时间" if mode == "time" else "补 kind")


async def run_fix_preview(mode: str):
    global _status
    if _status["running"]:
        return {"status": "already_running", **_status}
    _status = {"running": True, "mode": mode, "phase": "loading", "processed": 0, "total": 0,
               "plan_id": None, "error": None}
    try:
        await ensure_fix_tables()
        sources = await _load_candidates(mode)
        _status["total"] = len(sources)
        if not sources:
            _status.update(running=False, phase="empty")
            return {"status": "empty", **_status}
        _status["phase"] = "rewriting"
        items = []
        for index in range(0, len(sources), FIX_BATCH_SIZE):
            batch = sources[index:index + FIX_BATCH_SIZE]
            items.extend(build_plan_items(mode, batch, await _ask_model(mode, batch)))
            _status["processed"] = min(index + len(batch), len(sources))
            print(f"🕰️ 修补预览({mode}) 进度: {_status['processed']}/{len(sources)}", flush=True)
        pool = await get_pool()
        async with pool.acquire() as conn:
            plan_id = await conn.fetchval(
                "INSERT INTO memory_fix_plans (mode, items) VALUES ($1, $2::jsonb) RETURNING id",
                mode, _json_text(items),
            )
        _status.update(running=False, phase="done", plan_id=int(plan_id))
        print(f"✅ 修补预览({mode}) 完成: plan_id={plan_id}, 扫描 {len(sources)} 条，"
              f"有改动 {len(items)} 条", flush=True)
        return {"status": "done", **_status}
    except Exception as exc:
        _status.update(running=False, phase="error", error=str(exc))
        print(f"⚠️ 修补预览({mode}) 失败: {exc}", flush=True)
        return {"status": "error", **_status}


def start_fix_preview(mode: str):
    if mode not in FIX_MODES:
        raise ValueError("mode 只能是 time 或 kind")
    if _status["running"]:
        return {"status": "already_running", **_status}
    asyncio.create_task(run_fix_preview(mode))
    return {"status": "started", "mode": mode}


def get_fix_status():
    return dict(_status)


async def get_fix_plan(plan_id: int):
    await ensure_fix_tables()
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, created_at, mode, status, items, apply_result, applied_at, reverted_at
               FROM memory_fix_plans WHERE id = $1""",
            int(plan_id),
        )
    if not row:
        return None
    items = row["items"]
    if isinstance(items, str):
        items = json.loads(items)
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"].isoformat(),
        "mode": row["mode"],
        "status": row["status"],
        "items": items,
        "apply_result": _jsonb_object(row["apply_result"]) if row["apply_result"] else None,
        "applied_at": row["applied_at"].isoformat() if row["applied_at"] else None,
        "reverted_at": row["reverted_at"].isoformat() if row["reverted_at"] else None,
    }


def _record_date(created_at):
    if not created_at:
        return None
    try:
        return _me._local_dt(created_at).date()
    except Exception:
        return None


async def _embed(pairs):
    if not _db.MEMORY_VECTOR_ENABLED or not pairs:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        for memory_id, content in pairs:
            try:
                embedding = await _db.compute_embedding(content)
                if embedding:
                    await _db.save_memory_embedding(conn, memory_id, embedding)
            except Exception as exc:
                print(f"⚠️ 修补后重算向量失败（id={memory_id}）: {exc}", flush=True)


async def apply_fix_plan(plan_id: int, confirmed_plan_id: int, excluded_ids=None) -> dict:
    """原地更新；旧值进 memory_edit_log。期间被改过的条目跳过（不整批失败）。"""
    if int(plan_id) != int(confirmed_plan_id):
        raise ValueError("确认的方案编号不匹配")
    excluded = {int(value) for value in (excluded_ids or [])}
    await ensure_fix_tables()
    pool = await get_pool()
    reembed = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            plan = await conn.fetchrow(
                "SELECT id, status, items FROM memory_fix_plans WHERE id = $1 FOR UPDATE",
                int(plan_id),
            )
            if not plan:
                raise ValueError("修补方案不存在")
            if plan["status"] != "preview":
                raise FixConflict("这份方案已经应用或撤回过了")
            items = plan["items"]
            if isinstance(items, str):
                items = json.loads(items)
            applied = 0
            skipped_changed = 0
            for item in items:
                memory_id = int(item["id"])
                if memory_id in excluded:
                    continue
                row = await conn.fetchrow(
                    """SELECT id, content, kind, event_date, valid_until, created_at, is_active
                       FROM memories WHERE id = $1 FOR UPDATE""",
                    memory_id,
                )
                if not row or not row["is_active"] or _hash(row["content"]) != item["content_hash"]:
                    skipped_changed += 1
                    continue
                new_kind = item.get("new_kind")
                new_event = _as_date(item.get("new_event_date"))
                old_valid = row["valid_until"]
                new_valid = old_valid if old_valid is not None else _db.compute_valid_until(
                    new_kind, new_event, _record_date(row["created_at"]),
                )
                await conn.execute(
                    """UPDATE memories SET content = $2, kind = $3, event_date = $4, valid_until = $5
                       WHERE id = $1""",
                    memory_id, item["new_content"], new_kind, new_event, new_valid,
                )
                await conn.execute(
                    """INSERT INTO memory_edit_log
                       (plan_id, memory_id, old_content, new_content, old_kind, new_kind,
                        old_event_date, new_event_date, old_valid_until, new_valid_until)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                    int(plan_id), memory_id, row["content"], item["new_content"],
                    row["kind"], new_kind, row["event_date"], new_event, old_valid, new_valid,
                )
                applied += 1
                if item["new_content"] != row["content"]:
                    reembed.append((memory_id, item["new_content"]))
            result = {"status": "applied", "plan_id": int(plan_id), "applied": applied,
                      "excluded": len(excluded), "skipped_changed": skipped_changed}
            await conn.execute(
                """UPDATE memory_fix_plans SET status = 'applied', applied_at = NOW(),
                   apply_result = $2::jsonb WHERE id = $1""",
                int(plan_id), _json_text(result),
            )
    await _embed(reembed)
    return result


async def revert_fix_plan(plan_id: int) -> dict:
    """撤回：只恢复应用之后没再被改过的条目。"""
    await ensure_fix_tables()
    pool = await get_pool()
    reembed = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            plan = await conn.fetchrow(
                "SELECT id, status FROM memory_fix_plans WHERE id = $1 FOR UPDATE", int(plan_id),
            )
            if not plan:
                raise ValueError("修补方案不存在")
            if plan["status"] != "applied":
                raise FixConflict("只有已应用的方案能撤回")
            logs = await conn.fetch(
                """SELECT id, memory_id, old_content, new_content, old_kind, old_event_date,
                          old_valid_until
                   FROM memory_edit_log WHERE plan_id = $1 AND reverted_at IS NULL
                   ORDER BY id""",
                int(plan_id),
            )
            reverted = 0
            skipped_changed = 0
            for log in logs:
                current = await conn.fetchval(
                    "SELECT content FROM memories WHERE id = $1 FOR UPDATE", log["memory_id"],
                )
                if current != log["new_content"]:
                    skipped_changed += 1
                    continue
                await conn.execute(
                    """UPDATE memories SET content = $2, kind = $3, event_date = $4, valid_until = $5
                       WHERE id = $1""",
                    log["memory_id"], log["old_content"], log["old_kind"],
                    log["old_event_date"], log["old_valid_until"],
                )
                await conn.execute(
                    "UPDATE memory_edit_log SET reverted_at = NOW() WHERE id = $1", log["id"],
                )
                reverted += 1
                if log["old_content"] != log["new_content"]:
                    reembed.append((log["memory_id"], log["old_content"]))
            await conn.execute(
                "UPDATE memory_fix_plans SET status = 'reverted', reverted_at = NOW() WHERE id = $1",
                int(plan_id),
            )
    await _embed(reembed)
    return {"status": "reverted", "plan_id": int(plan_id), "reverted": reverted,
            "skipped_changed": skipped_changed}
