"""
记忆提取模块 —— 用 LLM 从对话中提炼关键记忆
=============================================
每次对话结束后，把最近的对话内容发给一个便宜的模型，
让它提取出值得记住的信息，存到数据库里。

v2.3 改进：提取时注入已有记忆，让模型对比后只提取全新信息。
"""

import os
import json
import httpx
from typing import List, Dict

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 记忆模型专用 API Key（不设则回退到主 API_KEY）
# 适用于中转站按模型分组、不同模型需要不同 Key 的场景
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")

# 用来提取记忆的模型（便宜的就行）
# 面板上这项写着"留空用默认"，清空会写进一个空串，os.getenv 的默认值这时不生效，
# 所以默认值单独拎出来用 or 兜，热更新和重启后行为才一致
DEFAULT_MEMORY_MODEL = "anthropic/claude-haiku-4.5"
MEMORY_MODEL = os.getenv("MEMORY_MODEL") or DEFAULT_MEMORY_MODEL

# 记忆提取的输出上限，原先硬编码 1000。部分上游会把 reasoning token
# 也算进这条额度，JSON 可能在收尾前被截断，表面只报"未找到JSON数组"
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "4000"))

def get_memory_api_key() -> str:
    return MEMORY_API_KEY or API_KEY


def _diagnose_incomplete(finish_reason, completion_tokens, reasoning_tokens) -> str:
    """JSON 收不了尾时，判断是截断还是格式不符。证据不足就返回"无法判定"，不硬猜"""
    if finish_reason == "length":
        return (
            f"输出被上限切断（上游明确报 finish_reason=length，当前 MEMORY_MAX_TOKENS={MEMORY_MAX_TOKENS}）。"
            "调高该值；模型若带推理模式，推理 token 也占这条额度"
        )

    if finish_reason == "stop":
        # stop 说明上游认为输出完整，就算推理 token 顶满 usage 也不是截断
        return "上游报正常结束，是模型没按 JSON 格式输出。检查提示词，或换一个更听话的模型"

    if isinstance(completion_tokens, int) and completion_tokens >= MEMORY_MAX_TOKENS:
        extra = f"，其中推理 {reasoning_tokens}" if reasoning_tokens is not None else ""
        return (
            f"输出很可能被切断（completion_tokens={completion_tokens}{extra}，已顶到上限 {MEMORY_MAX_TOKENS}）。"
            "先调高 MEMORY_MAX_TOKENS 再看。注意各家 usage 口径不一，这条是强证据但不是铁证"
        )

    return (
        f"原因无法判定：上游没给 finish_reason（={finish_reason}），usage 也证明不了是否触顶。"
        "先确认中转站是否返回这两个字段，再谈是截断还是格式问题"
    )


EXTRACTION_PROMPT = """你负责给一个陪伴型 AI 记事。目标不是做知识库，而是让它下次聊天时像一个认识对方很久的人：
记得对方是谁、在乎什么、最近在经历什么、两个人之间说过什么、约好了什么。

# 现在
现在是 {now_text}（对方所在时区）。对话里每条消息前面都标了发送时间。

# 要记的
- 稳定事实：身份、生日、职业、学校、住在哪、养的宠物、家里的人等
- 偏好：喜欢和讨厌的东西、习惯、口味、审美、希望被怎样对待
- 关系：对方生活里的重要的人，和他们之间发生的事、关系的变化
- 目标和计划：考试、项目、旅行、打算做的事，带上具体日期
- 正在发生的状态：最近在忙什么、在烦什么、身体和情绪状况（写明「截至哪天」）
- 健康：身体状况、病、过敏、吃的药、饮食禁忌、作息和睡眠问题
- 生活里值得回忆的细节：去了哪、吃了什么特别的、看了什么、遇到的有意思的事
- 情绪上重要的时刻：难过、开心、崩溃、被安慰到、吵架、和好
- 两个人之间的约定和承诺，包括 AI 答应对方的事
- 事件类要保留双方关键的原话，并标明是谁说的，例如：她说「我真的撑不下去了」，AI 回「我一直在」

# 不记的
- 寒暄（你好、在吗、晚安这种本身没有内容的话）
- 纯知识问答、翻译、百科、代码讲解（除非里面透露了对方自己的事）
- 调试、排错的过程；只记最终结论（比如「2026-11-18 把网关部署到了 Render」）
- 关于记忆系统本身的讨论（某条记忆有没有被记住、记忆遗漏之类）
- AI 的思考过程、思维链

# 只记对方的事
- 来源以对方自己说的话为准。AI 回复里转述、猜测、总结的对方信息，不能单独当作来源
- AI 对对方做出的承诺、约定是例外，标 "kind": "promise"，可以只引用 AI 的那条消息

# 时间（很重要）
- 所有相对时间（今天、昨天、前天、明天、后天、下周三、这周末、上个月、最近、刚才、今晚）
  都要按那条消息的发送时间换算成具体日期写进 content，不能留「昨天」「下周」这种说法
- 例子：「2026-11-18 和妈妈吵了一架」「计划 2026-12-20 考研初试」
- 状态类写成「截至 2026-11-18 在赶论文」
- event_date 填事情发生或计划发生的那一天（YYYY-MM-DD），说不清是哪天就不填

# 写法
- 写成一个人直接能看懂的完整句子，保留具体细节：人名、地名、原因、原话、数字
- 不要压缩成标签式的短语（不要写「用户-咖啡-喜欢」「情绪低落」这种）
- 一条记忆讲一件事；同一件事在多条消息里补充，就合成一条完整的

# 已知记忆
下面是库里和这段对话可能相关的记忆，每条前面是编号和记录日期：
<已知记忆>
{existing_memories}
</已知记忆>

- 已经记过、只是换了说法的，不要再记
- 对方的情况变了（以前喜欢咖啡、现在改喝茶；计划改期了；状态结束了），
  输出 "action": "update"，target_id 填被取代的那条的编号，content 写成完整的现状，
  例如「以前喜欢喝咖啡，2026-11 起改喝茶」
- 其他新内容 "action": "new"
- 没有值得记的就返回 []

# 输出
只返回 JSON 数组，每条：
- content：完整的一句话
- importance：1-10，越影响以后的相处越高
- source_ids：支撑这条的消息编号（至少一条是对方说的，promise 除外）
- kind：fact（长期事实/偏好）| state（临时状态）| event（发生过的事）| plan（计划、约定、将来的事）| promise（AI 对对方的承诺）
- event_date：YYYY-MM-DD，可以不填
- action：new | update
- target_id：action 为 update 时填旧记忆编号

[
  {{"content": "2026-11-18 晚上和妈妈因为考研的事吵了一架，她说「我妈根本不懂我」", "importance": 7,
    "source_ids": [12], "kind": "event", "event_date": "2026-11-18", "action": "new"}},
  {{"content": "以前喜欢喝咖啡，2026-11 起改喝茶", "importance": 5,
    "source_ids": [14], "kind": "fact", "action": "update", "target_id": 123}}
]
"""


WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
MEMORY_KINDS = ("fact", "state", "event", "plan", "promise")


def _local_dt(value=None, timezone_hours=None):
    """转成本地时间（按 TIMEZONE_HOURS）。value 为空表示现在；不带时区的按 UTC。"""
    from datetime import datetime, timedelta, timezone
    hours = timezone_hours if timezone_hours is not None else int(os.getenv("TIMEZONE_HOURS", "8"))
    if value is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=hours)))


def format_local_stamp(value=None, timezone_hours=None) -> str:
    """2026-11-18 周二 23:40"""
    dt = _local_dt(value, timezone_hours)
    return f"{dt.strftime('%Y-%m-%d')} {WEEKDAYS[dt.weekday()]} {dt.strftime('%H:%M')}"


def _parse_date(value):
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _format_existing(existing_memories) -> str:
    if not existing_memories:
        return "（暂无）"
    lines = []
    for mem in existing_memories:
        if isinstance(mem, str):
            lines.append(f"- {mem}")
            continue
        recorded = mem.get("event_date") or mem.get("created_at")
        if recorded is not None and not isinstance(recorded, str) and hasattr(recorded, "hour"):
            recorded = _local_dt(recorded).date()
        date_text = str(recorded)[:10] if recorded else "日期不明"
        lines.append(f"[#{mem['id']}｜{date_text}] {mem.get('content', '')}")
    return "\n".join(lines)


async def extract_memories(messages: List[Dict[str, str]], existing_memories: list = None,
                          now=None) -> List[Dict]:
    """
    从对话消息中提取记忆

    参数：
        messages: [{"role", "content", "source_id", "created_at"}, ...]；created_at 为空的按现在算
        existing_memories: 给模型对比的已知记忆，[{"id", "content", "created_at", "event_date"}]
                           （也兼容纯字符串列表）
        now: 「现在」，默认取当前时间

    返回：
        [{"content", "importance", "source_ids", "kind", "event_date", "action", "target_id"}, ...]
    """
    if not get_memory_api_key():
        print("⚠️  API_KEY 和 MEMORY_API_KEY 都未设置，跳过记忆提取")
        return []

    if not messages:
        return []

    # 把对话格式化成文本：[消息 12｜2026-11-18 周二 23:40] 用户：……
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        if role not in ("user", "assistant"):
            continue  # tool 消息不进提取
        content = msg.get("content", "")
        source_id = msg.get("source_id")
        stamp = format_local_stamp(msg.get("created_at") or now)
        number = f"消息 {source_id}" if isinstance(source_id, int) else "无编号"
        speaker = "用户" if role == "user" else "AI"
        conversation_text += f"[{number}｜{stamp}] {speaker}：{content}\n"

    if not conversation_text.strip():
        return []

    prompt = EXTRACTION_PROMPT.format(
        existing_memories=_format_existing(existing_memories),
        now_text=format_local_stamp(now),
    )

    # 调用 LLM 提取记忆
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {get_memory_api_key()}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://midsummer-gateway.local",
                    "X-Title": "Midsummer Memory Extraction",
                },
                json={
                    "model": MEMORY_MODEL,
                    "max_tokens": MEMORY_MAX_TOKENS,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"请从以下对话中提取新的记忆：\n\n{conversation_text}"},
                    ],
                },
            )

            if response.status_code != 200:
                print(f"⚠️  记忆提取请求失败: {response.status_code}, model={MEMORY_MODEL}: {response.text[:500]}")
                return []

            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            finish_reason = choice.get("finish_reason")

            # usage 各家口径不同（推理 token 有的单列有的算进 completion），只当佐证
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

            # 正文截断防刷屏，但长度和停止原因要给全，否则分不清是日志截断还是真截断
            usage_part = f"，completion_tokens={completion_tokens}/{MEMORY_MAX_TOKENS}" if completion_tokens is not None else "，usage 未提供"
            if reasoning_tokens is not None:
                usage_part += f"（其中推理 {reasoning_tokens}）"
            print(
                f"📝 记忆模型原始返回（{len(text)} 字符，finish_reason={finish_reason}{usage_part}）:\n{text[:500]}",
                flush=True,
            )

            # 清理可能的 markdown 格式（原始长度留给报错用，免得日志里两个数对不上）
            raw_len = len(text)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # 强力JSON提取：如果上面清理后仍然解析失败，用正则兜底
            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                # 尝试从文本中提取第一个 [...] 结构
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                        print(f"📝 JSON正则兜底提取成功")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  记忆提取结果解析失败: {e}")
                        return []
                else:
                    # 不要补上收尾的 ]：断掉的可能是半个字符串，
                    # 补完只会把残缺内容伪装成一条有效记忆存进库
                    print(
                        f"⚠️  记忆提取结果中未找到完整 JSON 数组（共 {raw_len} 字符），本轮跳过\n"
                        f"    {_diagnose_incomplete(finish_reason, completion_tokens, reasoning_tokens)}"
                    )
                    return []

            if not isinstance(memories, list):
                return []

            # 模型可能先吐完一个完整数组再被切断，解析成功也未必没丢东西。
            # 只认上游明确报 length；finish_reason 缺失时才退回 token 计数兜底，
            # 报 stop 的完整回复不警告（推理 token 会把 completion_tokens 顶过上限）
            if finish_reason == "length" or (
                finish_reason is None
                and isinstance(completion_tokens, int)
                and completion_tokens >= MEMORY_MAX_TOKENS
            ):
                print(
                    f"⚠️  本次解析成功，但上游显示输出已顶到上限 {MEMORY_MAX_TOKENS}，"
                    "后面可能还有没写完的记忆。建议调高 MEMORY_MAX_TOKENS"
                )

            # 验证格式
            valid_memories = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    source_ids = mem.get("source_ids")
                    try:
                        importance = int(mem.get("importance", 5))
                    except (TypeError, ValueError):
                        importance = 5
                    kind = str(mem.get("kind") or "").strip().lower()
                    action = str(mem.get("action") or "new").strip().lower()
                    target_id = mem.get("target_id")
                    try:
                        target_id = int(target_id) if target_id is not None and not isinstance(target_id, bool) else None
                    except (TypeError, ValueError):
                        target_id = None
                    valid_memories.append({
                        "content": str(mem["content"]),
                        "importance": max(1, min(10, importance)),
                        "source_ids": source_ids if isinstance(source_ids, list) else [],
                        "kind": kind if kind in MEMORY_KINDS else None,
                        "event_date": _parse_date(mem.get("event_date")),
                        "action": "update" if action == "update" and target_id else "new",
                        "target_id": target_id if action == "update" else None,
                    })

            print(f"📝 从对话中提取了 {len(valid_memories)} 条新记忆（已对比 {len(existing_memories or [])} 条已有记忆）")
            return valid_memories

    except json.JSONDecodeError as e:
        print(f"⚠️  记忆提取结果解析失败: {e}")
        return []
    except Exception as e:
        print(f"⚠️  记忆提取出错: {e}")
        return []


SCORING_PROMPT = """你是记忆重要性评分专家。请对以下记忆条目逐条评分。

# 评分规则（1-10）
- 9-10：核心身份信息（名字、生日、职业、重要关系）
- 7-8：重要偏好、重大事件、深层情感
- 5-6：日常习惯、一般偏好
- 3-4：临时状态、偶然提及
- 1-2：琐碎信息

# 输入记忆
{memories_text}

# 输出格式
返回 JSON 数组，每条包含原文和评分：
[{{"content": "原文", "importance": 评分数字}}]

只返回 JSON，不要其他文字。"""


async def score_memories(texts: List[str]) -> List[Dict]:
    """对纯文本记忆条目批量评分"""
    if not texts:
        return []

    memories_text = "\n".join(f"- {t}" for t in texts)
    prompt = SCORING_PROMPT.format(memories_text=memories_text)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {get_memory_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MEMORY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    # 跟提取同一个模型同一类活，跟着同一个配置走；写死会让用户调了也不生效
                    "max_tokens": MEMORY_MAX_TOKENS,
                },
            )

            if response.status_code != 200:
                print(f"⚠️  记忆评分请求失败: {response.status_code}, model={MEMORY_MODEL}: {response.text[:500]}")
                # 失败时返回默认分数
                return [{"content": t, "importance": 5} for t in texts]

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                    except json.JSONDecodeError:
                        return [{"content": t, "importance": 5} for t in texts]
                else:
                    return [{"content": t, "importance": 5} for t in texts]

            if not isinstance(memories, list):
                return [{"content": t, "importance": 5} for t in texts]

            valid = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    valid.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
                    })

            print(f"📝 为 {len(valid)} 条记忆完成自动评分")
            return valid

    except Exception as e:
        print(f"⚠️  记忆评分出错: {e}")
        return [{"content": t, "importance": 5} for t in texts]
