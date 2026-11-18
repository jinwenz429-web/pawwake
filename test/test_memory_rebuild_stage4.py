"""阶段 4：「整理新碎片」只合并不压缩、带日期和类型、可以 UPDATE 已有记忆、继承字段。"""

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import date
from unittest.mock import AsyncMock, patch

import database
import memory_rebuilder as rebuilder
from test.test_memory_rebuild_apply import FakeConnection, FakePool


def frag(memory_id, content, **extra):
    return {"id": memory_id, "content": content, "importance": 5, "layer": 1, "title": "",
            "event_date": None, "created_at": None, "merged_from": [], "kind": None,
            "valid_until": None, **extra}


class PromptTests(unittest.TestCase):
    def test_prompts_no_longer_compress(self):
        self.assertNotIn("高度凝练", rebuilder.CLASSIFY_PROMPT)
        self.assertIn("即使是日常", rebuilder.CLASSIFY_PROMPT)
        self.assertIn("不同日期发生的不同事件不能合并", rebuilder.SYNTHESIZE_PROMPT)
        self.assertIn("保留原话", rebuilder.SYNTHESIZE_PROMPT)
        self.assertIn("不得编造", rebuilder.SYNTHESIZE_PROMPT)
        self.assertIn("UPDATE", rebuilder.SYNTHESIZE_PROMPT)


class ValidationTests(unittest.TestCase):
    def test_update_must_target_a_shown_memory_once(self):
        update = {"action": "UPDATE", "source_ids": [1], "target_id": 88, "content": "现状"}
        self.assertEqual(rebuilder._validate_actions([1], [update], [88])[0]["target_id"], 88)
        with self.assertRaises(ValueError):
            rebuilder._validate_actions([1], [update], [77])
        with self.assertRaises(ValueError):
            rebuilder._validate_actions([1, 2], [update, dict(update, source_ids=[2])], [88])


class InheritanceTests(unittest.TestCase):
    def test_valid_until_is_latest_or_empty(self):
        sources = [{"kind": "state", "valid_until": date(2026, 12, 1), "event_date": None,
                    "created_at": None},
                   {"kind": "state", "valid_until": "2026-12-05", "event_date": "2026-11-01",
                    "created_at": None}]
        fields = rebuilder._inherited_fields(sources)
        self.assertEqual(fields, {"kind": "state", "event_date": date(2026, 11, 1),
                                  "valid_until": date(2026, 12, 5)})
        sources.append({"kind": "fact", "valid_until": None, "event_date": None, "created_at": None})
        self.assertIsNone(rebuilder._inherited_fields(sources)["valid_until"])

    def test_consolidation_inheritance(self):
        kind, valid = database.inherit_kind_and_valid_until([
            {"kind": "event", "valid_until": None}, {"kind": "event", "valid_until": None}])
        self.assertEqual((kind, valid), ("event", None))


class SynthesisInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_payload_carries_dates_kinds_and_existing_memories(self):
        sent = []

        async def fake_post(prompt, label, max_tokens=None):
            sent.append(json.loads(prompt.split("输入：\n", 1)[1]))
            return [{"action": "UPDATE", "source_ids": [1], "target_id": 88, "target_layer": 3,
                     "title": "饮品", "content": "以前喜欢咖啡，2026-11 起改喝茶", "importance": 6}]

        candidates = [{"id": 1, "decision": "core", "topic": "饮品", "normalized_content": "",
                       "importance": 5,
                       "source": frag(1, "最近改喝茶了", created_at="2026-11-18T15:40:00+00:00",
                                      event_date="2026-11-18", kind="fact")}]
        existing = [{"id": 88, "content": "喜欢喝咖啡", "layer": 3, "kind": "fact",
                     "event_date": None, "created_at": None}]
        with patch.object(rebuilder, "_post_json_array", fake_post):
            actions = await rebuilder._synthesize_once(candidates, "t", existing)
        fragment = sent[0]["fragments"][0]
        self.assertEqual(fragment["event_date"], "2026-11-18")
        self.assertEqual(fragment["kind"], "fact")
        self.assertTrue(fragment["created_at"].startswith("2026-11-"))
        self.assertNotIn("normalized_content", fragment)
        self.assertEqual(sent[0]["existing"][0]["note"], "已有记忆，id=88")
        self.assertEqual(actions[0]["target_content"], "喜欢喝咖啡")
        self.assertEqual(actions[0]["target_hash"], rebuilder._content_hash("喜欢喝咖啡"))

    async def test_cross_batch_round_sees_original_fragments_and_existing(self):
        candidates = [
            {"id": i, "decision": "core", "topic": "t", "normalized_content": "", "importance": 5,
             "source": frag(i, f"原文{i}")}
            for i in (1, 2, 3)
        ]
        calls = []

        async def fake_once(items, label, existing=None):
            calls.append((label, items, existing))
            if label.startswith("跨批次"):
                return [{"action": "MERGE", "source_ids": [items[0]["id"], items[1]["id"]],
                         "target_layer": 3, "title": "", "content": "合并", "importance": 5,
                         "reason": ""}]
            if len(items) == 2:
                return [{"action": "MERGE", "source_ids": [1, 2], "target_layer": 3,
                         "title": "", "content": "压过的版本", "importance": 5, "reason": ""}]
            return [{"action": "KEEP", "source_ids": [3], "target_layer": 1,
                     "title": "", "content": "原文3", "importance": 5, "reason": ""}]

        with patch.object(rebuilder, "REBUILD_GROUP_SIZE", 2), \
                patch.object(rebuilder, "_synthesize_once", fake_once):
            await rebuilder._synthesize_domain(candidates, "goals", [{"id": 88}])
        first_round = [call for call in calls if not call[0].startswith("跨批次")]
        self.assertTrue(all(existing is None for _, _, existing in first_round))
        label, pseudo, existing = calls[-1]
        self.assertEqual(existing, [{"id": 88}])
        self.assertEqual(pseudo[0]["_original_contents"], ["原文1", "原文2"])
        payload = rebuilder._candidate_payload(pseudo[0])
        self.assertEqual(payload["original_fragments"], ["原文1", "原文2"])


def plan_with_update(sources, target):
    return {
        "version": 2, "scope": "unprocessed_fragments",
        "source_hashes": {str(item["id"]): rebuilder._content_hash(item["content"]) for item in sources},
        "actions": [
            {"action": "UPDATE", "source_ids": [1], "target_id": target["id"], "target_layer": 3,
             "title": "饮品", "content": "以前喜欢咖啡，2026-11 起改喝茶", "importance": 6,
             "reason": "", "target_hash": rebuilder._content_hash(target["content"]),
             "target_content": target["content"]},
            {"action": "EVENT", "source_ids": [2], "target_layer": 2, "title": "看海",
             "content": "2026-11-15 去青岛看海", "importance": 6, "reason": ""},
        ],
    }


class ApplyUpdateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sources = [frag(1, "改喝茶了", kind="fact"),
                        frag(2, "去青岛看海", kind="event", event_date=date(2026, 11, 15))]
        self.target = {"id": 88, "content": "喜欢喝咖啡", "importance": 6, "layer": 3,
                       "title": "饮品", "event_date": None, "created_at": None,
                       "merged_from": [9], "kind": "fact", "valid_until": None}
        self.conn = FakeConnection(plan_with_update(self.sources, self.target),
                                   self.sources + [self.target])
        self.conn.plan["source_count"] = 2

    async def apply(self, mode):
        with patch.object(rebuilder, "ensure_rebuild_table", AsyncMock()), \
                patch.object(rebuilder, "get_pool", AsyncMock(return_value=FakePool(self.conn))), \
                patch.object(database, "MEMORY_UPDATE_MODE", mode), \
                patch.object(database, "MEMORY_VECTOR_ENABLED", False), \
                redirect_stdout(io.StringIO()):
            return await rebuilder.apply_memory_rebuild_plan(1, 1)

    async def test_apply_mode_archives_old_and_creates_new(self):
        result = await self.apply("apply")
        self.assertEqual((result["superseded"], result["proposals"]), (1, 0))
        old = self.conn.sources[88]
        self.assertFalse(old["is_active"])
        new = self.conn.sources[old["superseded_by"]]
        self.assertEqual(new["content"], "以前喜欢咖啡，2026-11 起改喝茶")
        self.assertEqual(new["kind"], "fact")
        self.assertFalse(self.conn.sources[1]["is_active"])
        event = next(row for row in self.conn.sources.values() if row["content"] == "2026-11-15 去青岛看海")
        self.assertEqual((event["kind"], event["event_date"]), ("event", date(2026, 11, 15)))

    async def test_observe_mode_only_writes_a_proposal(self):
        result = await self.apply("observe")
        self.assertEqual((result["superseded"], result["proposals"]), (0, 1))
        self.assertTrue(self.conn.sources[88]["is_active"])
        self.assertTrue(self.conn.sources[1]["is_active"])  # 碎片先留着，采纳时才归档
        old_id, new_content, old_content, payload = self.conn.proposals[0]
        self.assertEqual((old_id, old_content), (88, "喜欢喝咖啡"))
        self.assertEqual(json.loads(payload)["merged_from"], [1])
        # 碎片已被这份方案覆盖，下次不会再被当成新碎片
        self.assertEqual(self.conn.plan["status"], "applied")

    async def test_changed_target_rejects_the_plan(self):
        self.conn.sources[88]["content"] = "被手动改过"
        with self.assertRaises(rebuilder.RebuildApplyConflict):
            await self.apply("apply")
        self.assertTrue(self.conn.sources[1]["is_active"])


class ProposalAcceptConn:
    def __init__(self, payload):
        self.proposal = {"id": 5, "new_memory_id": None, "old_memory_id": 88,
                         "status": "pending", "payload": json.dumps(payload)}
        self.memories = {88: {"is_active": True}, 1: {"is_active": True}}
        self.inserted = None
        self.log = []

    def transaction(self):
        conn = self

        class Tx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *args):
                return False
        return Tx()

    async def fetchrow(self, sql, *args):
        return self.proposal

    async def fetchval(self, sql, *args):
        if "INSERT INTO memories" in sql:
            self.inserted = args
            self.memories[300] = {"is_active": True}
            return 300
        return self.memories[args[0]]["is_active"]

    async def execute(self, sql, *args):
        self.log.append((sql, args))
        if "superseded_by" in sql:
            self.memories[args[1]].update(is_active=False, superseded_by=args[0])
            return "UPDATE 1"
        if "UPDATE memories SET is_active = FALSE" in sql:
            for memory_id in args[0]:
                self.memories[memory_id]["is_active"] = False
            return f"UPDATE {len(args[0])}"
        return "UPDATE 1"


class ProposalAcceptTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepting_a_rebuild_proposal_creates_then_supersedes(self):
        conn = ProposalAcceptConn({"content": "以前喜欢咖啡，2026-11 起改喝茶", "importance": 6,
                                   "layer": 3, "title": "饮品", "merged_from": [1],
                                   "kind": "fact", "event_date": None, "valid_until": None})

        class Pool:
            def acquire(self):
                class Ctx:
                    async def __aenter__(self_inner):
                        return conn

                    async def __aexit__(self_inner, *args):
                        return False
                return Ctx()

        with patch.object(database, "get_pool", AsyncMock(return_value=Pool())), \
                patch.object(database, "MEMORY_VECTOR_ENABLED", False):
            result = await database.resolve_update_proposal(5, accept=True)
        self.assertEqual(result["created_id"], 300)
        self.assertEqual(conn.inserted[0], "以前喜欢咖啡，2026-11 起改喝茶")
        self.assertFalse(conn.memories[1]["is_active"])
        self.assertEqual(conn.memories[88]["superseded_by"], 300)


if __name__ == "__main__":
    unittest.main()
