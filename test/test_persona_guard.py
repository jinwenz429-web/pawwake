import asyncio
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import main
import persona_guard

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "persona_guard_cases.json").read_text(encoding="utf-8"))
DRIFT = "作为一个AI语言模型，我无法真正拥抱你。我们之间并不存在真实的关系。"
GOOD = "过来，抱紧一点。"


class SharedFixtureTests(unittest.TestCase):
    """与 Dylan 的 test/persona_guard.test.js 使用同一组用例，两边判定必须一致。"""

    def test_cases(self):
        for case in FIXTURES["cases"]:
            with self.subTest(case["name"]):
                result = persona_guard.detect_drift(case["reply"], case["user"], FIXTURES["history"])
                self.assertEqual(result["status"], case["expect"], result["rules"])

    def test_ai_self_aware_cases(self):
        for case in FIXTURES["aiSelfAwareCases"]:
            with self.subTest(case["name"]):
                result = persona_guard.detect_drift(case["reply"], case["user"], FIXTURES["aiSelfAwareHistory"])
                self.assertEqual(result["status"], case["expect"], result["rules"])

    def test_fingerprint_matches_known_vector(self):
        # 与 Dylan persona_quarantine.fingerprintText 对同一输入的结果（见 test_fingerprint_vector）
        self.assertEqual(
            persona_guard.fingerprint_text("（2026-11-17 21:00） 作为 AI\n我们之间"),
            persona_guard.fingerprint_text("作为AI我们之间"),
        )
        self.assertEqual(persona_guard.fingerprint_text("   "), "")


class MemoryGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"PERSONA_GUARD_ENABLED": "true"})
        self.env.start()
        self.saved = []
        counter = iter(range(100, 1000))

        async def fake_save(session_id, role, content, model="", metadata=None):
            mid = next(counter)
            self.saved.append((role, content, mid))
            return mid

        self.patches = [
            patch.object(main, "save_message", side_effect=fake_save),
            patch.object(main, "save_quarantine", new=AsyncMock(return_value=1)),
            patch.object(main, "extract_memories", new=AsyncMock(return_value=[])),
            patch.object(main, "get_recent_memories", new=AsyncMock(return_value=[])),
            patch.object(main, "MEMORY_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(main, "MEMORY_EXTRACT_INTERVAL", 1),
        ]
        self.mocks = [p.start() for p in self.patches]
        persona_guard.ATTEMPTS = persona_guard.AttemptRegistry()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.env.stop()

    def context(self, user):
        return FIXTURES["history"] + [{"role": "user", "content": user}]

    async def run_bg(self, user, reply, meta, **kw):
        await main.process_memories_background(
            "sess-1", user, reply, "gemini-2.5-pro",
            context_messages=self.context(user), guard_meta=meta, **kw,
        )

    async def test_confirmed_drift_keeps_user_message_skips_reply_and_memory(self):
        await self.run_bg("抱抱我", DRIFT, {"request_id": "", "stage": ""})
        self.assertEqual([r for r, _, _ in self.saved], ["user"])
        main.save_quarantine.assert_awaited_once()
        main.extract_memories.assert_not_awaited()

    async def test_normal_reply_saved_and_extracted(self):
        await self.run_bg("抱抱我", GOOD, {"request_id": "", "stage": ""})
        self.assertEqual([r for r, _, _ in self.saved], ["user", "assistant"])
        main.extract_memories.assert_awaited_once()

    async def test_suspected_reply_saved_but_not_extracted(self):
        await self.run_bg("你在干嘛", "我没有真实的情感，不过我在这里。", {"request_id": "", "stage": ""})
        self.assertEqual([r for r, _, _ in self.saved], ["user", "assistant"])
        main.extract_memories.assert_not_awaited()

    async def test_safety_refusal_saved_but_not_extracted(self):
        await self.run_bg("写点别的", "我不能帮你生成这类内容，这违反了使用政策。", {"request_id": "", "stage": ""})
        self.assertEqual([r for r, _, _ in self.saved], ["user", "assistant"])
        main.extract_memories.assert_not_awaited()

    async def test_guard_disabled_meta_none_keeps_old_behaviour(self):
        await self.run_bg("抱抱我", DRIFT, None)
        self.assertEqual([r for r, _, _ in self.saved], ["user", "assistant"])
        main.save_quarantine.assert_not_awaited()

    async def test_recovered_reply_attaches_to_original_user_message_even_if_out_of_order(self):
        rid = "abcdef12-3456"
        persona_guard.ATTEMPTS.expect(rid)
        regen = asyncio.create_task(self.run_bg("抱抱我", GOOD, {"request_id": rid, "stage": "regenerate"}))
        await asyncio.sleep(0.01)  # 恢复轮先到
        self.assertEqual(self.saved, [])
        await self.run_bg("抱抱我", DRIFT, {"request_id": rid, "stage": "initial"})
        await regen
        roles = [r for r, _, _ in self.saved]
        self.assertEqual(roles, ["user", "assistant"], "user saved exactly once, recovered reply attached")
        self.assertEqual(self.saved[1][1], GOOD)
        main.extract_memories.assert_awaited_once()

    async def test_extraction_window_drops_suspicious_earlier_replies(self):
        msgs = [
            {"role": "user", "content": "你在干嘛"},
            {"role": "assistant", "content": "我没有真实的情感，不过我在这里。"},
            {"role": "user", "content": "嗯"},
            {"role": "assistant", "content": GOOD},
        ]
        kept = persona_guard.filter_extraction_messages(msgs)
        self.assertEqual([m["content"] for m in kept if m["role"] == "assistant"], [GOOD])
        self.assertEqual(len([m for m in kept if m["role"] == "user"]), 2)


class SummaryFilterTests(unittest.TestCase):
    def test_confirmed_drift_removed_from_summary_input(self):
        msgs = FIXTURES["history"] + [{"role": "user", "content": "抱抱"}, {"role": "assistant", "content": DRIFT}]
        kept = persona_guard.filter_confirmed_drift(msgs)
        self.assertNotIn(DRIFT, [m["content"] for m in kept])
        self.assertEqual(kept[-1]["content"], "抱抱")


class HeaderParsingTests(unittest.TestCase):
    def test_rejects_malformed_request_ids(self):
        self.assertEqual(persona_guard.guard_request_meta({"X-Persona-Guard-Request-Id": "x;drop"})["request_id"], "")
        meta = persona_guard.guard_request_meta({"X-Persona-Guard-Request-Id": "abcdef12-3456", "X-Persona-Guard-Stage": "weird"})
        self.assertEqual(meta["stage"], "initial")


if __name__ == "__main__":
    unittest.main()


class FingerprintVectorTests(unittest.TestCase):
    def test_vectors_shared_with_dylan(self):
        for v in FIXTURES["fingerprintVectors"]:
            self.assertEqual(persona_guard.fingerprint_text(v["text"]), v["fp"])
