"""Persona Guard 的最小化存储：一张只存指纹和规则的隔离表。

不改 conversations 表结构，不删除任何已有记录。被确认漂移的回复不写入 conversations，
所以摘要、对话召回、分区上下文、记忆提取这些读 conversations 的路径天然看不到它。
正文默认不保存（PERSONA_GUARD_KEEP_QUARANTINED_TEXT=true 时才保存前 2000 字用于排查）。
"""

import json
import os
from typing import Optional

from database import get_pool


async def ensure_persona_guard_table() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_guard_quarantine (
                id           BIGSERIAL PRIMARY KEY,
                session_id   TEXT NOT NULL,
                request_id   TEXT,
                stage        TEXT,
                fingerprint  TEXT NOT NULL,
                status       TEXT NOT NULL,
                rules        TEXT,
                content      TEXT,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_pg_quarantine_fp ON persona_guard_quarantine (fingerprint);
            CREATE INDEX IF NOT EXISTS idx_pg_quarantine_session ON persona_guard_quarantine (session_id, id DESC);
            """
        )


async def save_quarantine(
    session_id: str,
    fingerprint: str,
    status: str,
    rules: list,
    request_id: str = "",
    stage: str = "",
    content: Optional[str] = None,
) -> Optional[int]:
    if not fingerprint:
        return None
    keep_text = os.getenv("PERSONA_GUARD_KEEP_QUARANTINED_TEXT", "false").strip().lower() == "true"
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT id FROM persona_guard_quarantine WHERE fingerprint = $1 LIMIT 1",
            fingerprint,
        )
        if existing:
            return existing
        return await conn.fetchval(
            """
            INSERT INTO persona_guard_quarantine
                (session_id, request_id, stage, fingerprint, status, rules, content)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            session_id, request_id or None, stage or None, fingerprint, status,
            json.dumps(list(rules or [])[:12], ensure_ascii=False),
            (content or "")[:2000] if keep_text else None,
        )


async def recent_quarantine_fingerprints(limit: int = 300) -> set:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT fingerprint FROM persona_guard_quarantine ORDER BY id DESC LIMIT $1",
            limit,
        )
    return {row["fingerprint"] for row in rows}
