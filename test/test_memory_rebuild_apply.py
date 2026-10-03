import unittest
from unittest.mock import patch
from contextlib import asynccontextmanager
from copy import deepcopy

from memory_rebuilder import (
    RebuildApplyConflict,
    _prepare_rebuild_apply,
    _content_hash,
    apply_memory_rebuild_plan,
)


class MemoryRebuildApplyTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"id": 1, "content": "old fact", "importance": 5, "layer": 1,
             "title": "", "event_date": None, "created_at": None},
            {"id": 2, "content": "another fact", "importance": 5, "layer": 1,
             "title": "", "event_date": None, "created_at": None},
            {"id": 3, "content": "daily detail", "importance": 5, "layer": 1,
             "title": "", "event_date": None, "created_at": None},
        ]
        self.plan = {
            "version": 2,
            "scope": "unprocessed_fragments",
            "source_hashes": {str(item["id"]): _content_hash(item["content"])
                              for item in self.sources},
            "actions": [
                {"action": "MERGE", "source_ids": [1, 2], "target_layer": 3,
                 "title": "fact", "content": "combined fact", "importance": 8, "reason": "same"},
                {"action": "DISCARD", "source_ids": [3], "target_layer": 0,
                 "title": "", "content": "", "importance": 5, "reason": "ephemeral"},
            ],
        }

    def test_merge_and_discard_archive_all_sources_and_create_one_memory(self):
        result = _prepare_rebuild_apply(self.plan, self.sources, 3)
        self.assertEqual(result["archive_ids"], [1, 2, 3])
        self.assertEqual(len(result["creates"]), 1)
        self.assertEqual(result["creates"][0]["merged_from"], [1, 2])
        self.assertEqual(result["creates"][0]["layer"], 3)

    def test_changed_or_missing_source_rejects_stale_plan(self):
        changed = [dict(item) for item in self.sources]
        changed[0]["content"] = "edited after preview"
        with self.assertRaises(RebuildApplyConflict):
            _prepare_rebuild_apply(self.plan, changed, 3)
        with self.assertRaises(RebuildApplyConflict):
            _prepare_rebuild_apply(self.plan, self.sources[:2], 3)

    def test_new_memories_after_preview_are_left_for_the_next_run(self):
        result = _prepare_rebuild_apply(self.plan, self.sources + [
            {"id": 4, "content": "new", "importance": 5, "layer": 1,
             "title": "", "event_date": None, "created_at": None},
            {"id": 5, "content": "existing core", "importance": 8, "layer": 3,
             "title": "core", "event_date": None, "created_at": None}], 3)
        self.assertEqual(result["archive_ids"], [1, 2, 3])

    def test_old_whole_library_plan_cannot_be_applied(self):
        self.plan["version"] = 1
        self.plan.pop("scope")
        with self.assertRaises(RebuildApplyConflict):
            _prepare_rebuild_apply(self.plan, self.sources, 3)

    def test_plan_cannot_touch_event_core_or_previously_processed_sources(self):
        for changes in ({"layer": 2}, {"layer": 3}, {"merged_from": [99]},
                        {"was_rebuilt": True}):
            with self.subTest(changes=changes):
                sources = [dict(item) for item in self.sources]
                sources[0].update(changes)
                with self.assertRaises(RebuildApplyConflict):
                    _prepare_rebuild_apply(self.plan, sources, 3)

    def test_keep_unchanged_is_retained_but_normalized_keep_is_replaced(self):
        self.plan["actions"] = [
            {"action": "KEEP", "source_ids": [1], "target_layer": 1,
             "title": "", "content": "old fact", "importance": 5, "reason": "keep"},
            {"action": "KEEP", "source_ids": [2], "target_layer": 3,
             "title": "fact", "content": "better fact", "importance": 7, "reason": "normalize"},
            {"action": "DISCARD", "source_ids": [3], "target_layer": 0,
             "title": "", "content": "", "importance": 5, "reason": "ephemeral"},
        ]
        result = _prepare_rebuild_apply(self.plan, self.sources, 3)
        self.assertEqual(result["archive_ids"], [2, 3])
        self.assertEqual(result["kept_ids"], [1])
        self.assertEqual(result["creates"][0]["merged_from"], [2])

    def test_invalid_action_coverage_rejects_entire_plan(self):
        self.plan["actions"] = self.plan["actions"][:1]
        with self.assertRaises(ValueError):
            _prepare_rebuild_apply(self.plan, self.sources, 3)


class FakeConnection:
    def __init__(self, plan, sources):
        self.plan = {"id": 1, "status": "preview", "source_count": len(sources), "plan": plan}
        self.sources = {row["id"]: {**row, "is_active": True} for row in sources}
        self.next_id = 100
        self.fail_insert = False

    @asynccontextmanager
    async def transaction(self):
        snapshot = deepcopy((self.plan, self.sources, self.next_id))
        try:
            yield
        except Exception:
            self.plan, self.sources, self.next_id = snapshot
            raise

    async def fetchrow(self, sql, *args):
        if "memory_rebuild_plans" in sql:
            return self.plan if args[0] == 1 else None
        raise AssertionError(sql)

    async def fetch(self, sql, *args):
        if "FROM memories" in sql:
            applied = self.plan["status"] == "applied"
            source_ids = self.plan["plan"].get("source_hashes", {})
            created_ids = self.plan.get("apply_result", {}).get("created_ids", [])
            return [dict(row, was_rebuilt=applied and (
                str(row["id"]) in source_ids or row["id"] in created_ids
            )) for row in self.sources.values() if row["is_active"]]
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        if sql.startswith("LOCK TABLE"):
            return "LOCK TABLE"
        if "UPDATE memories SET is_active = FALSE" in sql:
            changed = 0
            for source_id in args[0]:
                if self.sources[source_id]["is_active"]:
                    self.sources[source_id]["is_active"] = False
                    changed += 1
            return f"UPDATE {changed}"
        if "UPDATE memory_rebuild_plans" in sql:
            self.plan["status"] = "applied"
            import json
            self.plan["apply_result"] = json.loads(args[1])
            return "UPDATE 1"
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        if self.fail_insert:
            raise RuntimeError("simulated insert failure")
        assert "INSERT INTO memories" in sql
        self.next_id += 1
        self.sources[self.next_id] = {
            "id": self.next_id, "content": args[0], "importance": args[1],
            "layer": args[2], "title": args[3], "is_active": True,
            "merged_from": args[4], "event_date": args[5], "created_at": None,
        }
        return self.next_id


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class MemoryRebuildTransactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        sources = [{"id": 1, "content": "a", "importance": 5, "layer": 1,
                    "title": "", "event_date": None, "created_at": None}]
        plan = {"version": 2, "scope": "unprocessed_fragments",
                "source_hashes": {"1": _content_hash("a")},
                "actions": [{"action": "KEEP", "source_ids": [1],
                             "target_layer": 3, "title": "a", "content": "A",
                             "importance": 8, "reason": "normalize"}]}
        self.conn = FakeConnection(plan, sources)

    async def _apply(self):
        async def ensure():
            pass
        async def pool():
            return FakePool(self.conn)
        with patch("memory_rebuilder.ensure_rebuild_table", ensure), \
             patch("memory_rebuilder.get_pool", pool):
            return await apply_memory_rebuild_plan(1, 1)

    async def test_apply_is_atomic_and_rejects_second_execution(self):
        result = await self._apply()
        self.assertEqual(result["archived"], 1)
        self.assertEqual(result["created"], 1)
        self.assertEqual(self.conn.sources[101]["merged_from"], [1])
        self.assertFalse(self.conn.sources[1]["is_active"])
        with self.assertRaises(RebuildApplyConflict):
            await self._apply()
        self.assertEqual(len(self.conn.sources), 2)

    async def test_insert_failure_rolls_back_archiving_and_plan_status(self):
        self.conn.fail_insert = True
        with self.assertRaises(RuntimeError):
            await self._apply()
        self.assertTrue(self.conn.sources[1]["is_active"])
        self.assertEqual(self.conn.plan["status"], "preview")


if __name__ == "__main__":
    unittest.main()
