"""
Pawwake memory rebuild planner.
Creates a persistent, reviewable dry-run plan. It never mutates memories.
"""

import asyncio
import hashlib
import json
import os
from collections import defaultdict

import httpx

import memory_extractor as _me
from database import get_pool

REBUILD_BATCH_SIZE = int(os.getenv("MEMORY_REBUILD_BATCH_SIZE", "25"))
REBUILD_GROUP_SIZE = int(os.getenv("MEMORY_REBUILD_GROUP_SIZE", "45"))
REBUILD_MAX_TOKENS = int(os.getenv("MEMORY_REBUILD_MAX_TOKENS", "8000"))

DOMAINS = (
    "identity",
    "preferences",
    "relationships",
    "goals",
    "ongoing_state",
    "projects_tech",
    "health",
    "fandom_travel",
    "values_emotions",
    "significant_events",
    "other",
)

_status = {
    "running": False,
    "phase": None,
    "processed": 0,
    "total": 0,
    "plan_id": None,
    "summary": None,
    "error": None,
}


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _parse_json_array(text: str):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1:] if first_newline >= 0 else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end < start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, list):
        raise ValueError("模型输出顶层不是 JSON 数组")
    return value


async def _post_json_array(prompt: str, label: str, max_tokens: int = None):
    api_key = _me.get_memory_api_key()
    if not api_key:
        raise RuntimeError("MEMORY_API_KEY / API_KEY 未配置")
    max_tokens = max_tokens or REBUILD_MAX_TOKENS
    last_error = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    _me.API_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _me.MEMORY_MODEL,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
            if response.status_code != 200:
                raise RuntimeError(
                    f"{label} HTTP {response.status_code}: {response.text[:300]}"
                )
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            text = (choice.get("message") or {}).get("content") or ""
            if finish_reason == "length":
                raise RuntimeError(f"{label} 输出被截断")
            return _parse_json_array(text)
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(2)
    raise RuntimeError(f"{label} 失败: {last_error}")


async def ensure_rebuild_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_rebuild_plans (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status TEXT NOT NULL DEFAULT 'preview',
                source_count INTEGER NOT NULL,
                summary JSONB NOT NULL,
                plan JSONB NOT NULL,
                applied_at TIMESTAMPTZ DEFAULT NULL
            )
        """)


async def _load_active_memories():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance, layer, title,
                   created_at, event_date, merged_from
            FROM memories
            WHERE is_active = TRUE
            ORDER BY id
        """)
    result = []
    for row in rows:
        result.append({
            "id": int(row["id"]),
            "content": row["content"] or "",
            "importance": int(row["importance"] or 5),
            "layer": int(row["layer"] or 1),
            "title": row["title"] or "",
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "event_date": str(row["event_date"]) if row["event_date"] else None,
            "merged_from": list(row["merged_from"] or []),
        })
    return result


CLASSIFY_PROMPT = """你正在审查一个长期记忆库。不是聊天摘要库。
请逐条判断输入记忆是否值得继续占据长期记忆。

长期价值标准：
- 三个月后仍能帮助理解用户、维持关系连续性或继续长期任务；
- 稳定身份/偏好/关系/目标/持续状态/价值观；
- 真正改变项目后续状态的里程碑或稳定结论；
- 有显著情绪、关系或人生叙事意义的重要事件。

应丢弃：
- 普通吃饭、出行、天气、临时日程、一次性琐事；
- 技术调试流水、已经被最终结论取代的中间步骤；
- 没有未来用途的随口细节；
- 明显只是另一条长期事实的低信息量改写。

decision 只能是 core / event / discard。
core = 适合作为稳定语义记忆；event = 值得保留的情景记忆；discard = 应归档。
domain 必须从以下值中选：
identity, preferences, relationships, goals, ongoing_state, projects_tech,
health, fandom_travel, values_emotions, significant_events, other

topic 要短且稳定，用于把同一概念聚到一起，例如“考研目标”“母女关系”“Pawwake部署”。
normalized_content 对 core/event 写成高度凝练、可独立理解的版本；discard 可为空。
reason 用一句话解释判断，不要写泛泛的“重要/不重要”。

必须为每个输入 id 输出且只输出一次，不能新增不存在的 id。
只输出 JSON 数组：
[{"id":1,"decision":"core","domain":"goals","topic":"考研目标",
"normalized_content":"...","importance":8,"reason":"..."}]

输入：
{items}
"""


def _validate_classification(items, result):
    expected = {int(item["id"]) for item in items}
    seen = set()
    normalized = []
    for raw in result:
        if not isinstance(raw, dict):
            raise ValueError("分类结果包含非对象")
        memory_id = int(raw.get("id"))
        if memory_id not in expected or memory_id in seen:
            raise ValueError("分类结果 id 覆盖异常")
        decision = str(raw.get("decision", "")).strip().lower()
        if decision not in {"core", "event", "discard"}:
            raise ValueError(f"非法 decision: {decision}")
        domain = str(raw.get("domain", "other")).strip()
        if domain not in DOMAINS:
            domain = "other"
        seen.add(memory_id)
        normalized.append({
            "id": memory_id,
            "decision": decision,
            "domain": domain,
            "topic": str(raw.get("topic", "")).strip()[:40] or "未分类",
            "normalized_content": str(raw.get("normalized_content", "")).strip(),
            "importance": max(1, min(10, int(raw.get("importance", 5)))),
            "reason": str(raw.get("reason", "")).strip()[:300],
        })
    if seen != expected:
        raise ValueError(f"分类遗漏 {sorted(expected - seen)}")
    return normalized


async def _classify_batch(items):
    payload = [{
        "id": item["id"],
        "layer": item["layer"],
        "title": item["title"],
        "content": item["content"],
        "importance": item["importance"],
        "event_date": item["event_date"],
    } for item in items]
    result = await _post_json_array(
        CLASSIFY_PROMPT.replace("{items}", _json_text(payload)),
        "记忆分类",
    )
    return _validate_classification(items, result)


SYNTHESIZE_PROMPT = """你正在把同一领域的长期记忆候选整理成最终记忆方案。
输入中的每个 id 必须且只能被一个输出动作覆盖。

动作：
- KEEP：单条已经足够完整，source_ids 必须只有一个；
- MERGE：多条描述同一稳定事实/目标/偏好/项目状态，合成一条 layer=3；
- EVENT：一条或多条属于同一个有长期叙事价值的重要事件，合成 layer=2；
- DISCARD：结合上下文后确认没有长期价值，允许一条或多条。

关键规则：
1. 同一广泛主题下的不同事实不要硬合并；只有真正属于同一概念/同一事件才合并。
2. 稳定事实默认 layer=3；情景事件 layer=2。
3. 输出 content 必须自包含、凝练，但不要把有情绪意义的事件压成空洞标签。
4. 不要保留调试流水和普通日常；保留最终根因、决定、稳定结果。
5. reason 要具体说明为什么保留/合并/事件化/归档。
6. 不得编造输入中没有的信息。

只输出 JSON 数组：
[{"action":"MERGE","source_ids":[1,2],"target_layer":3,"title":"考研目标",
"content":"...","importance":8,"reason":"两条描述同一长期目标，合并后信息更完整"}]

输入：
{items}
"""


def _validate_actions(source_ids, actions):
    expected = {int(value) for value in source_ids}
    seen = set()
    normalized = []
    for raw in actions:
        if not isinstance(raw, dict):
            raise ValueError("整理结果包含非对象")
        action = str(raw.get("action", "")).strip().upper()
        if action not in {"KEEP", "MERGE", "EVENT", "DISCARD"}:
            raise ValueError(f"非法 action: {action}")
        ids = [int(value) for value in (raw.get("source_ids") or [])]
        if not ids:
            raise ValueError("action 缺少 source_ids")
        if action == "KEEP" and len(ids) != 1:
            raise ValueError("KEEP 只能覆盖一条来源")
        if action == "MERGE" and len(ids) < 2:
            raise ValueError("MERGE 至少需要两条来源")
        if action != "DISCARD" and not str(raw.get("content", "")).strip():
            raise ValueError(f"{action} 缺少 content")
        for memory_id in ids:
            if memory_id not in expected or memory_id in seen:
                raise ValueError("整理结果 source_ids 覆盖异常")
            seen.add(memory_id)
        target_layer = int(raw.get("target_layer", 0) or 0)
        if action == "DISCARD":
            target_layer = 0
        elif action == "EVENT":
            target_layer = 2
        elif action == "MERGE":
            target_layer = 3
        elif target_layer not in {2, 3}:
            target_layer = 3
        normalized.append({
            "action": action,
            "source_ids": ids,
            "target_layer": target_layer,
            "title": str(raw.get("title", "")).strip()[:80],
            "content": str(raw.get("content", "")).strip(),
            "importance": max(1, min(10, int(raw.get("importance", 5)))),
            "reason": str(raw.get("reason", "")).strip()[:500],
        })
    if seen != expected:
        raise ValueError(f"整理遗漏 {sorted(expected - seen)}")
    return normalized


async def _synthesize_once(candidates, label):
    payload = [{
        "id": item["id"],
        "current_layer": item["source"]["layer"],
        "topic": item["topic"],
        "suggested_kind": item["decision"],
        "normalized_content": item["normalized_content"],
        "original_content": item["source"]["content"],
        "importance": item["importance"],
    } for item in candidates]
    source_ids = [item["id"] for item in candidates]
    result = await _post_json_array(
        SYNTHESIZE_PROMPT.replace("{items}", _json_text(payload)),
        label,
    )
    return _validate_actions(source_ids, result)


async def _synthesize_domain(candidates, domain):
    if len(candidates) <= REBUILD_GROUP_SIZE:
        return await _synthesize_once(candidates, f"领域整理:{domain}")

    provisional = []
    for index in range(0, len(candidates), REBUILD_GROUP_SIZE):
        chunk = candidates[index:index + REBUILD_GROUP_SIZE]
        provisional.extend(
            await _synthesize_once(chunk, f"领域整理:{domain}:{index // REBUILD_GROUP_SIZE + 1}")
        )

    # 再把跨批次形成的保留候选做一次归并；已判定 DISCARD 的候选直接保留，
    # 避免空 content 被下一轮误当成 core。
    final = [action for action in provisional if action["action"] == "DISCARD"]
    pseudo = []
    for index, action in enumerate(
        [item for item in provisional if item["action"] != "DISCARD"],
        start=1,
    ):
        synthetic_id = 1000000000 + index
        pseudo.append({
            "id": synthetic_id,
            "decision": "event" if action["target_layer"] == 2 else "core",
            "topic": action["title"] or domain,
            "normalized_content": action["content"],
            "importance": action["importance"],
            "source": {
                "layer": action["target_layer"] or 1,
                "content": action["content"],
            },
            "_original_source_ids": action["source_ids"],
        })

    if pseudo:
        reconciled = await _synthesize_once(pseudo, f"跨批次归并:{domain}")
        for action in reconciled:
            original_ids = []
            for synthetic_id in action["source_ids"]:
                item = next(p for p in pseudo if p["id"] == synthetic_id)
                original_ids.extend(item["_original_source_ids"])
            action["source_ids"] = sorted(set(original_ids))
            final.append(action)

    all_ids = [item["id"] for item in candidates]
    return _validate_actions(all_ids, final)


async def _save_plan(source_memories, actions):
    summary = {
        "source_count": len(source_memories),
        "action_count": len(actions),
        "discarded_sources": sum(
            len(action["source_ids"]) for action in actions if action["action"] == "DISCARD"
        ),
        "merge_actions": sum(1 for action in actions if action["action"] == "MERGE"),
        "event_actions": sum(1 for action in actions if action["action"] == "EVENT"),
        "core_actions": sum(
            1 for action in actions
            if action["action"] in {"KEEP", "MERGE"} and action["target_layer"] == 3
        ),
        "estimated_active_after": sum(
            1 for action in actions if action["action"] != "DISCARD"
        ),
    }
    plan = {
        "version": 1,
        "source_hashes": {
            str(item["id"]): _content_hash(item["content"])
            for item in source_memories
        },
        "actions": actions,
    }
    pool = await get_pool()
    async with pool.acquire() as conn:
        plan_id = await conn.fetchval(
            """INSERT INTO memory_rebuild_plans (source_count, summary, plan)
               VALUES ($1, $2::jsonb, $3::jsonb)
               RETURNING id""",
            len(source_memories),
            _json_text(summary),
            _json_text(plan),
        )
    return int(plan_id), summary


async def run_memory_rebuild_preview():
    global _status
    if _status["running"]:
        return {"status": "already_running", **_status}

    _status = {
        "running": True,
        "phase": "loading",
        "processed": 0,
        "total": 0,
        "plan_id": None,
        "summary": None,
        "error": None,
    }
    try:
        await ensure_rebuild_table()
        source_memories = await _load_active_memories()
        _status["total"] = len(source_memories)
        if not source_memories:
            raise RuntimeError("没有活跃记忆可整理")

        source_map = {item["id"]: item for item in source_memories}
        classified = []
        _status["phase"] = "classifying"
        for index in range(0, len(source_memories), REBUILD_BATCH_SIZE):
            batch = source_memories[index:index + REBUILD_BATCH_SIZE]
            result = await _classify_batch(batch)
            for item in result:
                item["source"] = source_map[item["id"]]
            classified.extend(result)
            _status["processed"] = min(index + len(batch), len(source_memories))
            print(
                f"🧠 整库重整分类进度: {_status['processed']}/{len(source_memories)}",
                flush=True,
            )

        actions = []
        for item in classified:
            if item["decision"] == "discard":
                actions.append({
                    "action": "DISCARD",
                    "source_ids": [item["id"]],
                    "target_layer": 0,
                    "title": "",
                    "content": "",
                    "importance": item["importance"],
                    "reason": item["reason"],
                })

        domains = defaultdict(list)
        for item in classified:
            if item["decision"] != "discard":
                domains[item["domain"]].append(item)

        _status["phase"] = "synthesizing"
        for domain in DOMAINS:
            candidates = domains.get(domain, [])
            if not candidates:
                continue
            actions.extend(await _synthesize_domain(candidates, domain))
            print(
                f"🧩 整库重整领域完成: {domain} ({len(candidates)} 条候选)",
                flush=True,
            )

        all_ids = [item["id"] for item in source_memories]
        actions = _validate_actions(all_ids, actions)
        plan_id, summary = await _save_plan(source_memories, actions)
        _status.update({
            "running": False,
            "phase": "done",
            "processed": len(source_memories),
            "plan_id": plan_id,
            "summary": summary,
            "error": None,
        })
        print(
            "✅ 整库重整 dry-run 完成: "
            f"plan_id={plan_id}, source={summary['source_count']}, "
            f"actions={summary['action_count']}, "
            f"discard={summary['discarded_sources']}, "
            f"estimated_active={summary['estimated_active_after']}",
            flush=True,
        )
        return {"status": "done", **_status}
    except Exception as exc:
        _status.update({
            "running": False,
            "phase": "error",
            "error": str(exc),
        })
        print(f"⚠️ 整库重整 dry-run 失败: {exc}", flush=True)
        return {"status": "error", **_status}


def start_memory_rebuild_preview():
    if _status["running"]:
        return {"status": "already_running", **_status}
    asyncio.create_task(run_memory_rebuild_preview())
    return {"status": "started"}


def _jsonb_object(value):
    """asyncpg 默认会把 JSON/JSONB 返回为字符串；统一还原为 dict。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"JSONB 字段不是对象: {type(value).__name__}")


async def get_memory_rebuild_status():
    """返回当前任务状态；进程重启后自动回退到 Neon 中最近一次持久化方案。"""
    result = dict(_status)
    if result.get("running") or result.get("plan_id") or result.get("error"):
        return result

    await ensure_rebuild_table()
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, summary
               FROM memory_rebuild_plans
               ORDER BY id DESC
               LIMIT 1"""
        )
    if not row:
        return result

    summary = _jsonb_object(row["summary"])
    source_count = int(summary.get("source_count") or 0)
    result.update({
        "phase": "done",
        "processed": source_count,
        "total": source_count,
        "plan_id": int(row["id"]),
        "summary": summary,
    })
    return result


async def get_memory_rebuild_plan(plan_id: int):
    await ensure_rebuild_table()
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, created_at, status, source_count, summary, plan, applied_at
               FROM memory_rebuild_plans WHERE id = $1""",
            int(plan_id),
        )
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"].isoformat(),
        "status": row["status"],
        "source_count": int(row["source_count"]),
        "summary": _jsonb_object(row["summary"]),
        "plan": _jsonb_object(row["plan"]),
        "applied_at": row["applied_at"].isoformat() if row["applied_at"] else None,
    }
