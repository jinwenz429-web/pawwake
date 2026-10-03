import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import memory_rebuilder as rebuilder
from test.test_memory_rebuild_apply import FakeConnection, FakePool


def fragment(memory_id, **overrides):
    return dict(id=memory_id, content=f"detail {memory_id}", importance=5,
                layer=1, title="", event_date=None, created_at=None,
                merged_from=[], **overrides)


class MemoryRebuildScopeTests(unittest.IsolatedAsyncioTestCase):
    async def load(self, conn):
        with patch.object(rebuilder, "get_pool", AsyncMock(return_value=FakePool(conn))):
            return await rebuilder._load_active_memories()

    async def test_preview_selects_only_unprocessed_fragments(self):
        rows = [fragment(1), {**fragment(2), "layer": 2},
                {**fragment(3), "layer": 3},
                {**fragment(4), "merged_from": [8]}, fragment(5), fragment(6)]
        conn = FakeConnection({"version": 1, "source_hashes": {"5": "hash"}}, rows)
        conn.plan.update(status="applied", apply_result={"created_ids": [6]})
        selected = await self.load(conn)
        self.assertEqual([item["id"] for item in selected], [1])

    async def test_preview_does_not_mark_fragments_as_processed(self):
        conn = FakeConnection({"source_hashes": {"1": "hash"}}, [fragment(1)])
        self.assertEqual([item["id"] for item in await self.load(conn)], [1])
        self.assertEqual([item["id"] for item in await self.load(conn)], [1])

    async def test_keep_as_fragment_is_not_selected_again_after_apply(self):
        row = fragment(1)
        plan = {"version": 2, "scope": "unprocessed_fragments",
                "source_hashes": {"1": rebuilder._content_hash(row["content"])},
                "actions": [{"action": "KEEP", "source_ids": [1],
                             "target_layer": 1, "title": "", "content": row["content"],
                             "importance": 5, "reason": "retain exact detail"}]}
        conn = FakeConnection(plan, [row])
        with patch.object(rebuilder, "ensure_rebuild_table", AsyncMock()), \
             patch.object(rebuilder, "get_pool", AsyncMock(return_value=FakePool(conn))):
            result = await rebuilder.apply_memory_rebuild_plan(1, 1)
        self.assertEqual(result["kept"], 1)
        self.assertTrue(conn.sources[1]["is_active"])
        self.assertEqual(await self.load(conn), [])

    async def test_new_fragment_is_not_archived_by_an_earlier_preview(self):
        row = fragment(1)
        plan = {"version": 2, "scope": "unprocessed_fragments",
                "source_hashes": {"1": rebuilder._content_hash(row["content"])},
                "actions": [{"action": "DISCARD", "source_ids": [1],
                             "target_layer": 0, "content": "", "title": "",
                             "importance": 5, "reason": "transient"}]}
        conn = FakeConnection(plan, [row, fragment(2)])
        conn.plan["source_count"] = 1
        with patch.object(rebuilder, "ensure_rebuild_table", AsyncMock()), \
             patch.object(rebuilder, "get_pool", AsyncMock(return_value=FakePool(conn))):
            await rebuilder.apply_memory_rebuild_plan(1, 1)
        self.assertFalse(conn.sources[1]["is_active"])
        self.assertTrue(conn.sources[2]["is_active"])
        self.assertEqual([item["id"] for item in await self.load(conn)], [2])

    async def test_empty_run_uses_no_model_and_does_not_restore_an_old_plan(self):
        status = {"running": False, "phase": None, "processed": 0, "total": 0,
                  "plan_id": None, "summary": None, "error": None}
        with patch.object(rebuilder, "_status", status), \
             patch.object(rebuilder, "ensure_rebuild_table", AsyncMock()), \
             patch.object(rebuilder, "_load_active_memories", AsyncMock(return_value=[])), \
             patch.object(rebuilder, "_classify_batch", AsyncMock()) as classify, \
             patch.object(rebuilder, "_save_plan", AsyncMock()) as save, \
             patch.object(rebuilder, "get_pool", side_effect=AssertionError("old plan loaded")):
            result = await rebuilder.run_memory_rebuild_preview()
            self.assertEqual(result["phase"], "empty")
            self.assertIsNone(result["error"])
            classify.assert_not_awaited()
            save.assert_not_awaited()
            self.assertEqual((await rebuilder.get_memory_rebuild_status())["phase"], "empty")


if __name__ == "__main__":
    unittest.main()
