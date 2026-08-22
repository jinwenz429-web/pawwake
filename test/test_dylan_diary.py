import json
import sys
import types
import unittest


database_stub = types.ModuleType("database")


async def unavailable_pool():
    raise RuntimeError("test must replace get_pool")


database_stub.get_pool = unavailable_pool
sys.modules["database"] = database_stub

import diary_store


class FakeConnection:
    def __init__(self):
        self.execute_calls = []
        self.fetchval_calls = []

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))

    async def fetchval(self, sql, *args):
        self.fetchval_calls.append((sql, args))
        return 42


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


class DylanDiaryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = FakeConnection()
        self.pool = FakePool(self.conn)
        self.original_get_pool = diary_store.get_pool

        async def fake_get_pool():
            return self.pool

        diary_store.get_pool = fake_get_pool

    async def asyncTearDown(self):
        diary_store.get_pool = self.original_get_pool

    async def test_ensure_table_creates_dedicated_diary_table(self):
        await diary_store.ensure_dylan_diary_table()
        self.assertEqual(len(self.conn.execute_calls), 1)
        self.assertIn("CREATE TABLE IF NOT EXISTS dylan_diary", self.conn.execute_calls[0][0])

    async def test_save_diary_persists_content_and_metadata(self):
        diary_id = await diary_store.save_dylan_diary(
            "今天想她了。",
            {"source": "dylan-heartbeat", "mood": "soft"},
        )
        self.assertEqual(diary_id, 42)
        self.assertEqual(len(self.conn.fetchval_calls), 1)
        _, args = self.conn.fetchval_calls[0]
        self.assertEqual(args[0], "今天想她了。")
        self.assertEqual(
            json.loads(args[1]),
            {"source": "dylan-heartbeat", "mood": "soft"},
        )

    async def test_save_diary_rejects_blank_content(self):
        with self.assertRaises(ValueError):
            await diary_store.save_dylan_diary("   ")
