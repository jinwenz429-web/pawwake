"""Persistent memory rebuild previews and explicitly confirmed application."""

import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime

import httpx

import memory_extractor as _me
import database as _db
from database import get_pool

REBUILD_BATCH_SIZE = int(os.getenv("MEMORY_REBUILD_BATCH_SIZE", "25"))
REBUILD_GROUP_SIZE = int(os.getenv("MEMORY_REBUILD_GROUP_SIZE", "45"))
REBUILD_MAX_TOKENS = int(os.getenv("MEMORY_REBUILD_MAX_TOKENS", "8000"))
REBUILD_EXISTING_CONTEXT = int(os.getenv("MEMORY_REBUILD_EXISTING_CONTEXT", "10"))
REBUILD_SCOPE = "unprocessed_fragments"

# Applied plans also cover unchanged KEEP sources and created memories whose
# merged_from links may later be cleared by archive cleanup. Previews do not.
_ACTIVE_MEMORIES_SQL = """
    SELECT m.id, m.content, m.importance, m.layer, m.title,
           m.created_at, m.event_date, m.merged_from, m.kind, m.valid_until,
           m.superseded_by,
           EXISTS (
               SELECT 1 FROM memory_rebuild_plans p
               WHERE p.status = 'applied' AND (
                   (p.plan->'source_hashes') ? m.id::text
                   OR COALESCE(p.apply_result->'created_ids', '[]'::jsonb)
                      @> jsonb_build_array(m.id)
               )
           ) AS was_rebuilt
    FROM memories m
    WHERE m.is_active = TRUE
    ORDER BY m.id
"""


def _is_unprocessed_fragment(memory) -> bool:
    return (int(memory.get("layer") or 1) == 1
            and not memory.get("merged_from")
            and not memory.get("was_rebuilt", False))


def _is_fragment_plan(plan) -> bool:
    return plan.get("version") == 2 and plan.get("scope") == REBUILD_SCOPE

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


class RebuildApplyConflict(Exception):
    """A rebuild plan was already applied or no longer matches active memories."""


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
                    _me.get_memory_api_base_url(),
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
        await conn.execute("""
            ALTER TABLE memory_rebuild_plans
            ADD COLUMN IF NOT EXISTS apply_result JSONB
        """)


async def _load_active_memories():
    """Load only active fragments that have never been successfully organized."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(_ACTIVE_MEMORIES_SQL)
    result = []
    for row in rows:
        if not _is_unprocessed_fragment(row):
            continue
        result.append({
            "id": int(row["id"]),
            "content": row["content"] or "",
            "importance": int(row["importance"] or 5),
            "layer": int(row["layer"] or 1),
            "title": row["title"] or "",
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "event_date": str(row["event_date"]) if row["event_date"] else None,
            "merged_from": list(row["merged_from"] or []),
            "kind": row.get("kind"),
            "valid_until": str(row["valid_until"]) if row.get("valid_until") else None,
        })
    return result


def _local_created(value) -> str | None:
    """created_at（ISO 字符串或 datetime）转成本地日期时间文字，给模型看。"""
    if not value:
        return None
    try:
        return _me.format_local_stamp(value)
    except Exception:
        return str(value)[:16]


CLASSIFY_PROMPT = """你在帮一个陪伴型 AI 整理它关于对方的记忆。目标是让它像一个认识对方很久的人，
不是做知识库，所以生活里的事、情绪、关系、约定都值得留。请逐条判断输入的记忆碎片。

值得留的（和平时提取记忆的标准一致）：
- 稳定事实、偏好、关系、目标和计划、正在发生的状态、健康
- 生活里值得回忆的细节（去了哪、吃了什么特别的、遇到的事），即使是日常
- 情绪上重要的时刻、双方的约定和承诺

应丢弃的（discard）：
- 寒暄、纯知识问答或翻译、调试过程（只留最终结论那条）
- 关于记忆系统本身的讨论、AI 的思考过程
- 和另一条完全重复、没有多出任何信息的

decision 只能是 core / event / discard：
core = 稳定的事实/偏好/关系/目标；event = 发生过的事、某一天的经历；discard = 应归档。
domain 必须从以下值中选：
identity, preferences, relationships, goals, ongoing_state, projects_tech,
health, fandom_travel, values_emotions, significant_events, other

topic 要短且稳定，用于把同一概念聚到一起，例如「考研目标」「母女关系」「Pawwake部署」。
normalized_content 一般留空；只有原文看不懂（缺主语、指代不清）时才写一个补全后的版本，
补全时保留原文的细节、原话和日期，不许压缩。
reason 用一句话解释判断，不要写泛泛的「重要/不重要」。
每条输入带 created_at（记录时间）、event_date、kind，判断时参考，不要改动日期。

必须为每个输入 id 输出且只输出一次，不能新增不存在的 id。
只输出 JSON 数组：
[{"id":1,"decision":"core","domain":"goals","topic":"考研目标",
"normalized_content":"","importance":8,"reason":"..."}]

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
        "created_at": _local_created(item.get("created_at")),
        "event_date": item["event_date"],
        "kind": item.get("kind"),
    } for item in items]
    result = await _post_json_array(
        CLASSIFY_PROMPT.replace("{items}", _json_text(payload)),
        "记忆分类",
    )
    return _validate_classification(items, result)


SYNTHESIZE_PROMPT = """你在把同一领域的记忆碎片整理成最终方案。只合并，不压缩。
输入分两部分：
- fragments：待整理的碎片。每个 id 必须且只能被一个输出动作的 source_ids 覆盖；
- existing：库里已有的相关核心记忆/事件记忆（「已有记忆，id=…」），只供参考，
  除非用 UPDATE，否则不要把它们放进 source_ids。

动作：
- KEEP：原样保留那一条碎片，source_ids 只有一个。不许改写 content（输出的 content 会被忽略），
  只能调整 title、importance、target_layer；
- MERGE：多条碎片确认是同一个稳定事实/目标/偏好/状态，合成一条 layer=3；
- EVENT：一条或多条碎片确认是同一件事（同一天、同一个经历），合成 layer=2；
- DISCARD：结合上下文后确认没有保留价值；
- UPDATE：碎片说明某条已有记忆的情况变了或需要补充。target_id 填那条已有记忆的 id，
  source_ids 填提供新信息的碎片，content 写成完整的现状（例如「以前喜欢喝咖啡，2026-11 起改喝茶」）。
  每条已有记忆最多被 UPDATE 一次。

关键规则：
1. 只合并确认是同一个概念或同一件事的条目；同一大主题下的不同事实不要硬合并。
2. 不同日期发生的不同事件不能合并成一条。
3. 合并后要保留原话（谁说的）、具体细节（人名、地点、原因、数字）和具体日期；
   不要写成概括性的标签，宁可长一点。
4. 不得编造输入中没有的信息，日期以输入为准。
5. 稳定事实默认 layer=3；具体经历 layer=2。
6. reason 要具体说明为什么保留/合并/事件化/归档/更新。

只输出 JSON 数组：
[{"action":"MERGE","source_ids":[1,2],"target_layer":3,"title":"考研目标",
"content":"...","importance":8,"reason":"两条描述同一长期目标，合并后信息更完整"},
{"action":"UPDATE","target_id":88,"source_ids":[3],"target_layer":3,"title":"饮品偏好",
"content":"以前喜欢喝咖啡，2026-11 起改喝茶","importance":6,"reason":"碎片 3 说明口味变了"}]

输入：
{items}
"""


def _validate_actions(source_ids, actions, existing_ids=None):
    """校验整理动作：每个碎片恰好被覆盖一次；UPDATE 只能指向给模型看过的已有记忆，各一次。"""
    expected = {int(value) for value in source_ids}
    allowed_targets = {int(value) for value in (existing_ids or [])}
    seen = set()
    seen_targets = set()
    normalized = []
    for raw in actions:
        if not isinstance(raw, dict):
            raise ValueError("整理结果包含非对象")
        action = str(raw.get("action", "")).strip().upper()
        if action not in {"KEEP", "MERGE", "EVENT", "DISCARD", "UPDATE"}:
            raise ValueError(f"非法 action: {action}")
        ids = [int(value) for value in (raw.get("source_ids") or [])]
        if not ids:
            raise ValueError("action 缺少 source_ids")
        if action == "MERGE" and len(ids) == 1:
            # 模型常把「单条升为核心记忆」写成只有一条来源的 MERGE。以前直接报错，
            # 整个领域的整理预览都会失败；按「只合并不压缩」改成 KEEP 升到 layer 3，
            # 保留原文，不用模型的改写版本。
            action = "KEEP"
            raw = dict(raw, target_layer=3)
        if action == "KEEP" and len(ids) != 1:
            raise ValueError("KEEP 只能覆盖一条来源")
        if action == "MERGE" and len(ids) < 2:
            raise ValueError("MERGE 至少需要两条来源")
        if action in {"MERGE", "EVENT", "UPDATE"} and not str(raw.get("content", "")).strip():
            raise ValueError(f"{action} 缺少 content")
        target_id = None
        if action == "UPDATE":
            try:
                target_id = int(raw.get("target_id"))
            except (TypeError, ValueError):
                raise ValueError("UPDATE 缺少 target_id")
            if target_id not in allowed_targets:
                raise ValueError(f"UPDATE 指向了没给出的已有记忆 #{target_id}")
            if target_id in seen_targets:
                raise ValueError(f"已有记忆 #{target_id} 被 UPDATE 了不止一次")
            seen_targets.add(target_id)
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
        elif target_layer not in {1, 2, 3}:
            target_layer = 3
        item = {
            "action": action,
            "source_ids": ids,
            "target_layer": target_layer,
            "title": str(raw.get("title", "")).strip()[:80],
            "content": str(raw.get("content", "")).strip(),
            "importance": max(1, min(10, int(raw.get("importance", 5)))),
            "reason": str(raw.get("reason", "")).strip()[:500],
        }
        if action == "UPDATE":
            item["target_id"] = target_id
            for key in ("target_content", "target_hash", "target_layer_before", "target_kind"):
                if key in raw:
                    item[key] = raw[key]
        normalized.append(item)
    if seen != expected:
        raise ValueError(f"整理遗漏 {sorted(expected - seen)}")
    return normalized


def _candidate_payload(item):
    source = item["source"]
    payload = {
        "id": item["id"],
        "current_layer": source["layer"],
        "topic": item["topic"],
        "suggested_kind": item["decision"],
        "original_content": source["content"],
        "created_at": _local_created(source.get("created_at")),
        "event_date": source.get("event_date"),
        "kind": source.get("kind"),
        "importance": item["importance"],
    }
    if item.get("normalized_content"):
        payload["normalized_content"] = item["normalized_content"]
    if item.get("_original_contents"):
        # 跨批次归并：带上原始碎片原文，不能只给上一轮压过的版本
        payload["original_fragments"] = item["_original_contents"]
    return payload


def _existing_payload(existing):
    return [{
        "existing_id": mem["id"],
        "note": f"已有记忆，id={mem['id']}",
        "layer": mem.get("layer"),
        "kind": mem.get("kind"),
        "event_date": str(mem["event_date"]) if mem.get("event_date") else None,
        "created_at": _local_created(mem.get("created_at")),
        "content": mem.get("content") or "",
    } for mem in (existing or [])]


async def _synthesize_once(candidates, label, existing=None):
    payload = {
        "fragments": [_candidate_payload(item) for item in candidates],
        "existing": _existing_payload(existing),
    }
    source_ids = [item["id"] for item in candidates]
    result = await _post_json_array(
        SYNTHESIZE_PROMPT.replace("{items}", _json_text(payload)),
        label,
    )
    existing_map = {int(mem["id"]): mem for mem in (existing or [])}
    actions = _validate_actions(source_ids, result, existing_map.keys())
    by_id = {item["id"]: item for item in candidates}
    for action in actions:
        if action["action"] == "KEEP":
            # KEEP 不许改写：预览里显示的就是库里那条原文，免得看着像改过
            action["content"] = by_id[action["source_ids"][0]]["source"]["content"]
        elif action["action"] == "UPDATE":
            target = existing_map[action["target_id"]]
            action["target_content"] = target.get("content") or ""
            action["target_hash"] = _content_hash(target.get("content") or "")
            action["target_layer_before"] = int(target.get("layer") or 1)
            action["target_kind"] = target.get("kind")
    return actions


async def _related_existing(candidates, exclude_ids):
    """和这批碎片相关的已有核心/事件记忆（混合检索），给合成那一步参考。"""
    if REBUILD_EXISTING_CONTEXT <= 0:
        return []
    text = "\n".join(item["source"]["content"] for item in candidates)[:4000]
    try:
        rows = await _db.search_memories(text, limit=REBUILD_EXISTING_CONTEXT * 3)
    except Exception as exc:
        print(f"⚠️ 碎片整理：检索已有记忆失败，本领域不带参考: {exc}", flush=True)
        return []
    exclude = {int(value) for value in exclude_ids}
    related = []
    for row in rows:
        mem = dict(row)
        if int(mem["id"]) in exclude or int(mem.get("layer") or 1) not in (2, 3):
            continue
        related.append(mem)
        if len(related) >= REBUILD_EXISTING_CONTEXT:
            break
    return related


async def _synthesize_domain(candidates, domain, existing=None):
    if len(candidates) <= REBUILD_GROUP_SIZE:
        return await _synthesize_once(candidates, f"领域整理:{domain}", existing)

    # 分批时，第一轮不给已有记忆（避免几批各自 UPDATE 同一条），跨批次归并那一轮再给
    provisional = []
    for index in range(0, len(candidates), REBUILD_GROUP_SIZE):
        chunk = candidates[index:index + REBUILD_GROUP_SIZE]
        provisional.extend(
            await _synthesize_once(chunk, f"领域整理:{domain}:{index // REBUILD_GROUP_SIZE + 1}")
        )

    # 再把跨批次形成的保留候选做一次归并；已判定 DISCARD 的候选直接保留，
    # 避免空 content 被下一轮误当成 core。
    by_id = {item["id"]: item for item in candidates}
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
            "normalized_content": "",
            "importance": action["importance"],
            "source": {
                "layer": action["target_layer"] or 1,
                "content": action["content"],
                "created_at": min(
                    (by_id[i]["source"].get("created_at") or "" for i in action["source_ids"]),
                    default=None,
                ) or None,
                "event_date": min(
                    (by_id[i]["source"].get("event_date") for i in action["source_ids"]
                     if by_id[i]["source"].get("event_date")),
                    default=None,
                ),
                "kind": by_id[action["source_ids"][0]]["source"].get("kind"),
            },
            "_original_contents": [by_id[i]["source"]["content"] for i in action["source_ids"]],
            "_original_source_ids": action["source_ids"],
            "_provisional_action": action,
        })

    if pseudo:
        reconciled = await _synthesize_once(pseudo, f"跨批次归并:{domain}", existing)
        for action in reconciled:
            if action["action"] == "KEEP":
                # 跨批次 KEEP = 保留上一轮那个动作本身（可能是 MERGE/EVENT），只采纳新的标题和分数
                item = next(p for p in pseudo if p["id"] == action["source_ids"][0])
                provisional = dict(item["_provisional_action"])
                provisional["title"] = action["title"] or provisional["title"]
                provisional["importance"] = action["importance"]
                final.append(provisional)
                continue
            original_ids = []
            for synthetic_id in action["source_ids"]:
                item = next(p for p in pseudo if p["id"] == synthetic_id)
                original_ids.extend(item["_original_source_ids"])
            action["source_ids"] = sorted(set(original_ids))
            final.append(action)

    all_ids = [item["id"] for item in candidates]
    return _validate_actions(all_ids, final, [mem["id"] for mem in (existing or [])])


async def _save_plan(source_memories, actions):
    summary = {
        "scope": REBUILD_SCOPE,
        "source_count": len(source_memories),
        "action_count": len(actions),
        "discarded_sources": sum(
            len(action["source_ids"]) for action in actions if action["action"] == "DISCARD"
        ),
        "merge_actions": sum(1 for action in actions if action["action"] == "MERGE"),
        "event_actions": sum(1 for action in actions if action["action"] == "EVENT"),
        "update_actions": sum(1 for action in actions if action["action"] == "UPDATE"),
        "core_actions": sum(
            1 for action in actions
            if action["action"] in {"KEEP", "MERGE"} and action["target_layer"] == 3
        ),
        "estimated_active_after": sum(
            1 for action in actions if action["action"] != "DISCARD"
        ),
    }
    plan = {
        "version": 2,
        "scope": REBUILD_SCOPE,
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
            _status.update(running=False, phase="empty")
            return {"status": "empty", **_status}

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
                f"🧠 碎片重整分类进度: {_status['processed']}/{len(source_memories)}",
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
            existing = await _related_existing(candidates, source_map.keys())
            actions.extend(await _synthesize_domain(candidates, domain, existing))
            print(
                f"🧩 碎片重整领域完成: {domain} ({len(candidates)} 条候选)",
                flush=True,
            )

        all_ids = [item["id"] for item in source_memories]
        actions = _validate_actions(
            all_ids, actions,
            [action["target_id"] for action in actions if action["action"] == "UPDATE"],
        )
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
            "✅ 碎片重整 dry-run 完成: "
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
        print(f"⚠️ 碎片重整 dry-run 失败: {exc}", flush=True)
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
    if (result.get("running") or result.get("plan_id") or result.get("error")
            or result.get("phase") == "empty"):
        return result

    await ensure_rebuild_table()
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, status, summary
               FROM memory_rebuild_plans
               ORDER BY id DESC
               LIMIT 1"""
        )
    if not row:
        return result

    summary = _jsonb_object(row["summary"])
    source_count = int(summary.get("source_count") or 0)
    result.update({
        "phase": "applied" if row["status"] == "applied" else "done",
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
            """SELECT id, created_at, status, source_count, summary, plan,
                      applied_at, apply_result
               FROM memory_rebuild_plans WHERE id = $1""",
            int(plan_id),
        )
    if not row:
        return None
    plan = _jsonb_object(row["plan"])
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"].isoformat(),
        "status": row["status"],
        "source_count": int(row["source_count"]),
        "summary": _jsonb_object(row["summary"]),
        "plan": plan,
        "can_apply": row["status"] == "preview" and _is_fragment_plan(plan),
        "applied_at": row["applied_at"].isoformat() if row["applied_at"] else None,
        "apply_result": _jsonb_object(row["apply_result"]) if row["apply_result"] else None,
    }


def _as_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _inherited_fields(sources, preferred_kind=None):
    """新建记忆继承来源的 kind / event_date / valid_until。

    kind：优先用被更新那条的；否则取来源里最多的那种。
    event_date：来源里最早的事件日（没有就用记录日），和以前一致。
    valid_until：取最晚的那个；有任何一条为空就为空（说明里面有长期有效的内容）。
    """
    kinds = [str(src.get("kind")) for src in sources if src.get("kind")]
    kind = preferred_kind or (Counter(kinds).most_common(1)[0][0] if kinds else None)
    dates = []
    for src in sources:
        value = _as_date(src.get("event_date")) or _as_date(src.get("created_at"))
        if value is not None:
            dates.append(value)
    valid = [_as_date(src.get("valid_until")) for src in sources]
    valid_until = max(valid) if valid and all(v is not None for v in valid) else None
    return {
        "kind": kind,
        "event_date": min(dates) if dates else None,
        "valid_until": valid_until,
    }


def _prepare_rebuild_apply(plan: dict, active_sources: list, source_count: int,
                           update_mode: str = "observe") -> dict:
    """Validate only the saved fragment sources against the locked snapshot."""
    if not _is_fragment_plan(plan):
        raise RebuildApplyConflict("旧整库方案已停用，请重新生成仅整理新碎片的方案")
    source_hashes = plan.get("source_hashes")
    if not isinstance(source_hashes, dict) or len(source_hashes) != source_count:
        raise ValueError("方案来源数量不一致")
    try:
        expected_ids = {int(value) for value in source_hashes}
    except (TypeError, ValueError) as exc:
        raise ValueError("方案来源 ID 无效") from exc
    if len(expected_ids) != len(source_hashes):
        raise ValueError("方案来源 ID 重复")
    source_map = {int(item["id"]): item for item in active_sources
                  if int(item["id"]) in expected_ids and _is_unprocessed_fragment(item)}
    if set(source_map) != expected_ids:
        raise RebuildApplyConflict("来源碎片已变化或已整理，请重新生成并审阅方案")
    for source_id, source in source_map.items():
        if _content_hash(source["content"]) != source_hashes[str(source_id)]:
            raise RebuildApplyConflict(
                f"来源记忆 #{source_id} 已变化，请重新生成并审阅方案"
            )

    raw_actions = plan.get("actions") or []
    target_ids = [raw.get("target_id") for raw in raw_actions
                  if str(raw.get("action", "")).upper() == "UPDATE"]
    actions = _validate_actions(expected_ids, raw_actions, [int(t) for t in target_ids if t is not None])
    active_map = {int(item["id"]): item for item in active_sources}
    archive_ids = []
    kept_ids = []
    keep_updates = []
    creates = []
    proposals = []
    for action in actions:
        ids = action["source_ids"]
        if action["action"] == "DISCARD":
            archive_ids.extend(ids)
            continue
        if action["action"] == "KEEP":
            # KEEP 一律原地保留那一行：id、created_at、content、embedding 都不动，
            # 最多更新 title / importance / layer
            source = source_map[ids[0]]
            kept_ids.append(ids[0])
            if ((source["title"] or "") != action["title"]
                    or int(source["importance"] or 5) != action["importance"]
                    or int(source["layer"] or 1) != action["target_layer"]):
                keep_updates.append({
                    "id": ids[0],
                    "title": action["title"] or (source["title"] or ""),
                    "importance": action["importance"],
                    "layer": action["target_layer"],
                })
            continue
        sources = [source_map[source_id] for source_id in ids]
        if action["action"] == "UPDATE":
            target = active_map.get(action["target_id"])
            if (target is None or target.get("superseded_by")
                    or _content_hash(target["content"]) != action.get("target_hash")):
                raise RebuildApplyConflict(
                    f"要更新的已有记忆 #{action['target_id']} 已变化或已归档，请重新生成并审阅方案"
                )
            inherited = _inherited_fields(sources + [target], target.get("kind"))
            item = {
                "content": action["content"],
                "importance": action["importance"],
                "layer": action["target_layer"],
                "title": action["title"] or (target.get("title") or ""),
                "merged_from": ids,
                **inherited,
            }
            if update_mode == "apply":
                archive_ids.extend(ids)
                creates.append({**item, "supersede": int(target["id"])})
            else:
                # observe：新旧都不动、碎片也先留着，只记一条提议；采纳时才新建、归档
                proposals.append({
                    "old_id": int(target["id"]),
                    "old_content": target["content"],
                    "payload": {**item, "event_date": str(item["event_date"]) if item["event_date"] else None,
                                "valid_until": str(item["valid_until"]) if item["valid_until"] else None},
                })
            continue
        archive_ids.extend(ids)
        creates.append({
            "content": action["content"],
            "importance": action["importance"],
            "layer": action["target_layer"],
            "title": action["title"],
            "merged_from": ids,
            **_inherited_fields(sources),
        })
    return {
        "archive_ids": sorted(archive_ids),
        "kept_ids": sorted(kept_ids),
        "keep_updates": keep_updates,
        "creates": creates,
        "proposals": proposals,
    }


async def apply_memory_rebuild_plan(plan_id: int, confirmed_plan_id: int) -> dict:
    """Apply one reviewed plan atomically; fail closed on any concurrent change."""
    if plan_id != confirmed_plan_id:
        raise ValueError("确认的方案编号不匹配")
    await ensure_rebuild_table()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """SELECT id, status, source_count, plan
                   FROM memory_rebuild_plans WHERE id = $1 FOR UPDATE""",
                plan_id,
            )
            if not row:
                raise ValueError("重整方案不存在")
            if row["status"] != "preview":
                raise RebuildApplyConflict("这份方案已应用或不再可应用")

            # Exclude concurrent INSERT/UPDATE while comparing the saved snapshot.
            await conn.execute("LOCK TABLE memories IN SHARE ROW EXCLUSIVE MODE")
            active_rows = await conn.fetch(_ACTIVE_MEMORIES_SQL)
            update_mode = _db.current_update_mode()
            prepared = _prepare_rebuild_apply(
                _jsonb_object(row["plan"]), active_rows, int(row["source_count"]),
                update_mode,
            )
            archive_ids = prepared["archive_ids"]
            if archive_ids:
                command = await conn.execute(
                    """UPDATE memories SET is_active = FALSE
                       WHERE id = ANY($1::int[]) AND is_active = TRUE""",
                    archive_ids,
                )
                if int(command.split()[-1]) != len(archive_ids):
                    raise RebuildApplyConflict("归档数量不一致，整批操作已回滚")

            for item in prepared["keep_updates"]:
                await conn.execute(
                    """UPDATE memories SET title = $2, importance = $3, layer = $4
                       WHERE id = $1 AND is_active = TRUE""",
                    item["id"], item["title"], item["importance"], item["layer"],
                )

            created_ids = []
            created_contents = []
            superseded = 0
            for item in prepared["creates"]:
                new_id = await conn.fetchval(
                    """INSERT INTO memories
                       (content, importance, layer, title, is_active,
                        merged_from, event_date, kind, valid_until)
                       VALUES ($1, $2, $3, $4, TRUE, $5, $6, $7, $8)
                       RETURNING id""",
                    item["content"], item["importance"], item["layer"],
                    item["title"], item["merged_from"], item["event_date"],
                    item.get("kind"), item.get("valid_until"),
                )
                if new_id is None:
                    raise RuntimeError("创建整理记忆失败，整批操作已回滚")
                created_ids.append(int(new_id))
                created_contents.append(item["content"])
                if item.get("supersede"):
                    command = await conn.execute(
                        """UPDATE memories SET is_active = FALSE, superseded_by = $1
                           WHERE id = $2 AND is_active = TRUE""",
                        int(new_id), item["supersede"],
                    )
                    if int(command.split()[-1]) != 1:
                        raise RebuildApplyConflict("要更新的已有记忆已变化，整批操作已回滚")
                    superseded += 1

            proposal_ids = []
            for item in prepared["proposals"]:
                proposal_id = await conn.fetchval(
                    """INSERT INTO memory_update_proposals
                       (new_memory_id, old_memory_id, new_content, old_content, source, payload)
                       VALUES (NULL, $1, $2, $3, 'rebuild', $4::jsonb) RETURNING id""",
                    item["old_id"], item["payload"]["content"], item["old_content"],
                    _json_text(item["payload"]),
                )
                proposal_ids.append(int(proposal_id))

            result = {
                "status": "applied",
                "plan_id": plan_id,
                "archived": len(archive_ids),
                "kept": len(prepared["kept_ids"]),
                "kept_updated": len(prepared["keep_updates"]),
                "created": len(created_ids),
                "created_ids": created_ids,
                "update_mode": update_mode,
                "superseded": superseded,
                "proposals": len(proposal_ids),
                "proposal_ids": proposal_ids,
            }
            await conn.execute(
                """UPDATE memory_rebuild_plans
                   SET status = 'applied', applied_at = NOW(),
                       apply_result = $2::jsonb WHERE id = $1""",
                plan_id, _json_text(result),
            )
    if _status.get("plan_id") == plan_id:
        _status["phase"] = "applied"
    await _embed_created_memories(list(zip(created_ids, created_contents)))
    return result


async def _embed_created_memories(items):
    """事务提交后再算向量：和 create_consolidated_events / merge_memories 一样，
    失败只记日志，不回滚已经应用的方案（漏掉的可以用「补算向量」补上）。"""
    if not _db.MEMORY_VECTOR_ENABLED or not items:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        for memory_id, content in items:
            try:
                embedding = await _db.compute_embedding(content)
                if embedding:
                    await _db.save_memory_embedding(conn, memory_id, embedding)
            except Exception as exc:
                print(f"⚠️ 整理新建记忆 embedding 计算失败（id={memory_id}）: {exc}", flush=True)
