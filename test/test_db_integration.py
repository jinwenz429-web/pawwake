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

    async def test_update_proposal_observe_then_accept_and_apply_mode(self):
        s = "conv-3"
        u = await self.say(s, "user", "我现在不喝咖啡了，改喝茶")
        a = await self.say(s, "assistant", "好，那我记住")
        async with self.pool.acquire() as conn:
            old_id = await conn.fetchval(
                "INSERT INTO memories (content, importance, layer) VALUES ('她喜欢喝咖啡', 6, 3) RETURNING id")
        with redirect_stdout(io.StringIO()):
            observed = await database.save_memory_deduplicated(
                content="以前喜欢喝咖啡，2026-11 起改喝茶", importance=6, source_session=s,
                source_message_ids=[u], kind="fact", supersede_target_id=old_id,
                update_mode="observe", round_assistant_id=a,
            )
        self.assertEqual((observed["action"], observed["update_mode"]), ("inserted", "observe"))
        self.assertTrue(await self.active(old_id))
        items, pending = await database.list_update_proposals("pending")
        self.assertEqual(pending, 1)
        self.assertEqual((items[0]["old_memory_id"], items[0]["new_memory_id"]),
                         (old_id, observed["memory_id"]))

        result = await database.resolve_update_proposal(items[0]["id"], accept=True)
        self.assertEqual(result["archived"], 1)
        self.assertFalse(await self.active(old_id))
        async with self.pool.acquire() as conn:
            self.assertEqual(await conn.fetchval("SELECT superseded_by FROM memories WHERE id = $1", old_id),
                             observed["memory_id"])
        with self.assertRaises(database.ProposalConflict):
            await database.resolve_update_proposal(items[0]["id"], accept=False)

        # 恢复被取代的记忆：清掉 superseded_by，能重新被召回
        await database.restore_archived_memories_batch([old_id])
        async with self.pool.acquire() as conn:
            self.assertIsNone(await conn.fetchval("SELECT superseded_by FROM memories WHERE id = $1", old_id))

        # apply 模式：直接归档旧条
        async with self.pool.acquire() as conn:
            core_id = await conn.fetchval(
                "INSERT INTO memories (content, importance, layer) VALUES ('她住在杭州西湖区', 7, 3) RETURNING id")
        with redirect_stdout(io.StringIO()):
            applied = await database.save_memory_deduplicated(
                content="她住在杭州西湖区，2026-11 搬到了滨江区", importance=7, source_session=s,
                source_message_ids=[u], update_mode="apply", round_assistant_id=a,
            )
        # 比核心记忆更完整（containment_update）→ 按更新处理，而不是丢掉
        self.assertEqual((applied["action"], applied["supersedes"]), ("inserted", core_id))
        self.assertFalse(await self.active(core_id))

    async def test_kind_and_valid_until_are_stored(self):
        from datetime import date
        s = "conv-4"
        u = await self.say(s, "user", "我在赶论文")
        with redirect_stdout(io.StringIO()):
            saved = await database.save_memory_deduplicated(
                content="截至 2026-11-18 在赶毕业论文", importance=6, source_session=s,
                source_message_ids=[u], kind="state", valid_until=date(2026, 12, 2),
            )
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT kind, valid_until FROM memories WHERE id = $1",
                                      saved["memory_id"])
        self.assertEqual((row["kind"], row["valid_until"]), ("state", date(2026, 12, 2)))


if __name__ == "__main__":
    unittest.main()
