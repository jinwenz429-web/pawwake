"""Diary entries the companion writes when it wakes (table name kept from the Dylan era)."""

import json
from typing import Any, Optional

from database import get_pool


async def ensure_dylan_diary_table() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dylan_diary (
                id          BIGSERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata    TEXT
            );
            """
        )


async def save_dylan_diary(content: str, metadata: Optional[dict[str, Any]] = None) -> int:
    clean_content = str(content or "").strip()
    if not clean_content:
        raise ValueError("content is required")

    encoded_metadata = (
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        if metadata
        else None
    )

    pool = await get_pool()
    async with pool.acquire() as conn:
        diary_id = await conn.fetchval(
            """
            INSERT INTO dylan_diary (content, metadata)
            VALUES ($1, $2)
            RETURNING id;
            """,
            clean_content,
            encoded_metadata,
        )
    return int(diary_id)


async def recent_diaries(limit: int = 3, days: int = 7) -> list[dict]:
    """Newest entries within ``days``, newest first."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, created_at
            FROM dylan_diary
            WHERE created_at >= NOW() - make_interval(days => $2)
            ORDER BY created_at DESC
            LIMIT $1;
            """,
            int(limit),
            int(days),
        )
    return [dict(row) for row in rows]


async def list_diaries(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, created_at, metadata
            FROM dylan_diary
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2;
            """,
            int(limit),
            int(offset),
        )
        total = await conn.fetchval("SELECT COUNT(*) FROM dylan_diary;")
    return [dict(row) for row in rows], int(total or 0)


async def delete_diary(diary_id: int) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM dylan_diary WHERE id = $1;", int(diary_id))
    return result.endswith(" 1")
