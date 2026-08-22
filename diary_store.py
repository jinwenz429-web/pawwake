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
