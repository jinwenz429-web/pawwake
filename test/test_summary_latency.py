import asyncio
import time
import unittest
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch

import main
from test import test_partition_session


class SummaryLatencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = {"summary_parts": [], "a_start_round": 0}
        self.history = []
        for n in range(8):
            self.history.extend([{"role": "user", "content": f"user {n}"},
                                 {"role": "assistant", "content": f"answer {n}"}])
        self.history.append({"role": "user", "content": "current"})
        self.calls = 0

    async def state_read(self, sid):
        return deepcopy(self.state)

    async def state_save(self, sid, parts, cursor):
        self.state = {"summary_parts": list(parts), "a_start_round": cursor}

    def configure(self, generate, rotations=2, budget=1):
        stack = ExitStack()
        for name, value in (("CACHE_PARTITION_X", 2), ("CACHE_PARTITION_TRIGGER", "rounds"),
                            ("CACHE_MAX_ROTATIONS", rotations),
                            ("CACHE_SUMMARY_BUDGET_SECONDS", budget),
                            ("CACHE_SUMMARY_MODEL", "summary"), ("MEMORY_ENABLED", False),
                            ("get_session_cache_state", self.state_read),
                            ("save_session_cache_state", self.state_save),
                            ("generate_summary", generate)):
            stack.enter_context(patch.object(main, name, value, create=True))
        return stack

    async def build(self, sid="latency-test", history=None):
        return await main.build_partitioned_messages(
            sid, deepcopy(history or self.history), "system", "current")

    async def test_round_mode_has_a_per_request_rotation_limit(self):
        async def summary(*args):
            self.calls += 1
            return "summary"
        with self.configure(summary, rotations=1):
            await self.build()
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.state["a_start_round"], 2)

    async def test_summary_budget_defers_without_archiving_unfinished_details(self):
        async def slow(*args):
            self.calls += 1
            await asyncio.sleep(0.3)
            return "summary"
        with self.configure(slow, budget=0.02):
            started = time.monotonic()
            result = await self.build()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.2)
        self.assertEqual(self.state["a_start_round"], 0)
        self.assertIn("answer 0", str(result))

    async def test_concurrent_builds_share_committed_rotation_progress(self):
        async def summary(*args):
            self.calls += 1
            await asyncio.sleep(0.02)
            return "summary"
        short = self.history[:8] + [self.history[-1]]
        with self.configure(summary):
            await asyncio.gather(self.build(history=short), self.build(history=short))
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.state["a_start_round"], 2)

    async def test_one_budget_covers_all_rotations_and_keeps_completed_progress(self):
        async def summary(*args):
            self.calls += 1
            await asyncio.sleep(0.001 if self.calls == 1 else 0.3)
            return "completed summary"
        with self.configure(summary, budget=0.05):
            started = time.monotonic()
            result = await self.build()
        self.assertLess(time.monotonic() - started, 0.25)
        self.assertEqual(self.state["a_start_round"], 2)
        self.assertEqual(self.state["summary_parts"], ["completed summary"])
        self.assertIn("answer 2", str(result))

    async def test_waiting_on_another_request_uses_the_same_budget(self):
        async def summary(*args):
            raise AssertionError("waiting request started duplicate summary work")
        lock = main._partition_build_lock("busy-session")
        await lock.acquire()
        try:
            with self.configure(summary, budget=0.02):
                started = time.monotonic()
                result = await self.build(sid="busy-session")
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertEqual(self.state["a_start_round"], 0)
            self.assertIn("answer 0", str(result))
        finally:
            lock.release()

    def test_suggestions_with_injected_events_are_still_auxiliary(self):
        messages = [
            {"role": "system", "content": 'Generate candidate next messages that the USER can send to the assistant. Return JSON with "suggestions".'},
            {"role": "user", "content": test_partition_session.AuxiliarySuggestionTests.suggestion_prompt},
            {"role": "assistant", "content": "historical wake event"},
        ]
        self.assertTrue(main._is_chat_suggestion_request(messages))
        messages.insert(1, {"role": "user", "content": "actual chat"})
        self.assertFalse(main._is_chat_suggestion_request(messages))

    def test_cleanup_preserves_only_summaries_before_the_removed_rounds(self):
        history = [dict(message, id=index + 1)
                   for index, message in enumerate(self.history[:-1])]
        ranges = [(7, 8, "suggestion prompt")]
        self.assertTrue(main._can_preserve_partition_cache(history, ranges, 3))
        self.assertTrue(main._can_preserve_partition_cache(history, ranges, 2))
        self.assertFalse(main._can_preserve_partition_cache(history, ranges, 4))
        self.assertFalse(main._can_preserve_partition_cache(history, [(99, 100, "missing")], 2))


if __name__ == "__main__":
    unittest.main()
