"""真数据库上的集成测试。只在设置了 PAWWAKE_TEST_DATABASE_URL 时运行，会清空那个库里的表。

    PAWWAKE_TEST_DATABASE_URL=postgresql://postgres@/pw?host=/tmp/pgtest&port=55432 \\
        python -m pytest test/test_db_integration.py
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import database

TEST_URL = os.getenv("PAWWAKE_TEST_DATABASE_URL", "")


@unittest.skipUnless(TEST_URL, "需要 PAWWAKE_TEST_DATABASE_URL")
class RealDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            patch.object(database, "DATABASE_URL", TEST_URL),
            patch.object(database, "_pool", None),
            patch.object(database, "MEMORY_VECTOR_ENABLED", False),
            patch.object(database, "CONVERSATION_RECALL_ENABLED", False),
        ]
        for p in self.patches:
            p.start()
        with redirect_stdout(io.StringIO()):
            pool = await database.get_pool()
            async with pool.acquire() as conn:
                await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            await database.init_tables()
            await database.init_tables()  # 迁移要能重复跑
        self.pool = pool

    async def asyncTearDown(self):
        await database.close_pool()
        for p in reversed(self.patches):
            p.stop()

    async def say(self, session, role, content):
        return await database.save_message(session, role, content, "m")

    async def remember(self, session, content, sources, round_id=None):
        with redirect_stdout(io.StringIO()):
            return await database.save_memory_deduplicated(
                content=content, importance=6, source_session=session,
                source_message_ids=sources, round_assistant_id=round_id,
            )

    async def active(self, memory_id):
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT is_active FROM memories WHERE id = $1", memory_id)

    async def test_reroll_archives_every_memory_born_in_that_round(self):
        s = "conv-1"
        u1 = await self.say(s, "user", "我养了一只叫团子的橘猫")
        a1 = await self.say(s, "assistant", "团子好可爱")
        older = (await self.remember(s, "她养了一只叫团子的橘猫", [u1], a1))["memory_id"]
        u2 = await self.say(s, "user", "明天想去吃老码头火锅")
        a2 = await self.say(s, "assistant", "我陪你去，记得点毛肚")
        user_only = (await self.remember(s, "她 2026-11-20 想去吃老码头火锅", [u2], a2))["memory_id"]
        cites_reply = (await self.remember(s, "AI 答应陪她去吃火锅", [u2, a2], a2))["memory_id"]
        spans_rounds = (await self.remember(s, "她和团子都爱吃，明天打算去吃火锅", [u1, u2], a2))["memory_id"]

        with redirect_stdout(io.StringIO()):
            await database.archive_replaced_answer_memories(s, u2, a2)

        self.assertFalse(await self.active(user_only))
        self.assertFalse(await self.active(cites_reply))
        self.assertTrue(await self.active(spans_rounds))
        self.assertTrue(await self.active(older))

        # 上一次提取还在后台跑，晚于归档才想写入：整批不入库
        late = await self.remember(s, "她想吃火锅想了很久", [u2], a2)
        self.assertEqual((late["action"], late["reason"]), ("skipped", "round_regenerated"))

        # 新回复替换上去以后，这一轮重新提取照常入库
        replaced = await database.replace_last_conversation_round(
            s, "明天想去吃老码头火锅", "好呀，几点出发？", "m", expected_assistant_id=a2,
        )
        fresh = await self.remember(s, "她 2026-11-20 想去吃老码头火锅", [u2], replaced["assistant_id"])
        self.assertEqual(fresh["action"], "inserted")

    async def test_conversation_tail_and_core_and_search_filters(self):
        s = "conv-2"
        for i in range(5):
            await self.say(s, "user", f"消息{i}")
        tail = await database.get_conversation_tail(s, 3)
        self.assertEqual([row["content"] for row in tail], ["消息2", "消息3", "消息4"])

        async with self.pool.acquire() as conn:
            core_id = await conn.fetchval(
                "INSERT INTO memories (content, importance, layer) VALUES ('她叫阿锦，喜欢猫', 10, 3) RETURNING id")
            old_id = await conn.fetchval(
                "INSERT INTO memories (content, importance, layer) VALUES ('她喜欢喝咖啡', 9, 3) RETURNING id")
            await conn.execute("UPDATE memories SET superseded_by = $1 WHERE id = $2", core_id, old_id)
        core = await database.get_core_memories(8)
        self.assertEqual([row["id"] for row in core], [core_id])
        with redirect_stdout(io.StringIO()):
            results = await database.search_memories("咖啡 猫", limit=10)
        self.assertNotIn(old_id, [row["id"] for row in results])


if __name__ == "__main__":
    unittest.main()
