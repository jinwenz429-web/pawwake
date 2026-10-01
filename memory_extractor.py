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


EXTRACTION_PROMPT = """你是长期记忆筛选器。你的任务不是记录聊天流水账，而是只保留未来仍能帮助 AI 理解用户、维持关系连续性或继续长期任务的信息。

# 核心判断
提取前先问：如果三个月后再次聊天，这条信息仍会帮助 AI 更好地理解用户或延续上下文吗？
只有答案明确为“会”时才进入长期记忆。宁可少记，不要为了避免遗漏而保存普通日常。

# 应当提取
- 稳定事实：身份、生日、职业、学历、长期居住信息等
- 稳定偏好：反复或明确表达的喜好、厌恶、习惯、审美与使用偏好
- 长期目标与持续状态：备考、长期计划、正在持续的项目、重要限制
- 关系信息：重要人物、长期关系模式、关系变化
- 价值观与长期需求：反复影响选择、互动方式或情绪反应的偏好和原则
- 项目里程碑：真正改变后续状态的决定、完成、迁移、故障根因与最终解决方案
- 重要情景记忆：里程碑、显著关系变化、强烈情绪事件、双方重要约定，或未来很可能再次提及的经历
- AI 与用户之间会影响未来互动的承诺、约定或边界

# 事件门槛
普通“发生过”不等于值得长期保存。事件至少满足一项才可提取：
1. 明显改变了关系、计划、项目状态或用户对事情的理解；
2. 带来持续或强烈的情绪意义；
3. 是生日、毕业、重要考试、旅行、重大购买/故障/完成等里程碑；
4. 未来再次提及时，知道这件事能显著改善连续性。
事件只有在原话本身具有关系或情绪意义时才保留少量关键原话，不要机械抄录对话。

# 不要提取
- 普通吃饭、普通出行、拿快递、天气、一次性日程等生活流水账
- 没有长期意义的一次性临时状态
- 普通寒暄、玩笑、随口提及
- 技术调试的逐步过程、无后续价值的报错细节；只记最终根因、关键决定和稳定结果
- AI 的纯知识性回答、翻译、百科、代码讲解
- 关于记忆系统本身“有没有记住某句话”的元讨论
- AI 的思考过程、思维链内容
- 已经能被更稳定、更完整的一条长期记忆覆盖的细碎改写

# 凝练原则
- 一条记忆只表达一个稳定概念或一个完整的重要事件
- 多轮对话都在补充同一事实时，输出一条整合后的完整表述，不要拆成多条近义碎片
- 临时细节只有在它解释长期事实或重要事件时才保留
- 项目记忆优先写“当前稳定状态 + 关键原因/决定”，不要写操作流水

# 已知信息处理【最重要】
<已知信息>
{existing_memories}
</已知信息>

- 新信息必须与已知信息逐条比对
- 相同、相似、语义重复或只是换一种说法的信息必须忽略
- 已知信息得到实质补充时，可以提取“整合后的最新完整版本”
- 与已知信息矛盾且明显代表状态变化的新信息可以提取为更新
- 如果没有达到长期记忆门槛的新信息，返回空数组 []

# 输出格式
只返回 JSON 数组：
[
  {{"content": "高度凝练、可独立理解的长期记忆", "importance": 分数}},
  {{"content": "完整的重要事件或长期事实", "importance": 分数}}
]

importance 1-10：长期影响越大、未来越常用，分数越高。普通日常即使真实发生也不应靠低分进入记忆库，而应直接不提取。
"""


async def extract_memories(messages: List[Dict[str, str]], existing_memories: List[str] = None) -> List[Dict]:
    """
    从对话消息中提取记忆

    参数：
        messages: 对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
        existing_memories: 已有记忆内容列表，用于去重对比

    返回：
        记忆列表，格式 [{"content": "...", "importance": N}, ...]
    """
    if not get_memory_api_key():
        print("⚠️  API_KEY 和 MEMORY_API_KEY 都未设置，跳过记忆提取")
        return []

    if not messages:
        return []

    # 把对话格式化成文本
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"用户: {content}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content}\n"

    if not conversation_text.strip():
        return []

    # 格式化已有记忆
    if existing_memories:
        memories_text = "\n".join(f"- {m}" for m in existing_memories)
    else:
        memories_text = "（暂无已知信息）"

    # 把已有记忆填入prompt
    prompt = EXTRACTION_PROMPT.format(existing_memories=memories_text)

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
                    valid_memories.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
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
