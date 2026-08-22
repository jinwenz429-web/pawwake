import asyncio
import copy
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

import database
import main


class _FakeRequest:
    def __init__(self, body, headers=None):
        self._body = body
        self.headers = headers or {}

    async def json(self):
        return copy.deepcopy(self._body)


class _FakeResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return _FakeResponse()


class _MemoryConnection:
    def __init__(self):
        self.states = {}

    async def fetchrow(self, query, session_id):
        row = self.states.get(session_id)
        if row is None:
            return None
        copied = dict(row)
        copied["seen_fragment_ids"] = list(row["seen_fragment_ids"])
        copied["seen_fragment_times"] = dict(row["seen_fragment_times"])
        return copied

    async def execute(self, query, *args):
        session_id = args[0]
        row = self.states.setdefault(
            session_id,
            {
                "summary": "",
                "a_start_round": 0,
                "seen_fragment_ids": [],
                "seen_fragment_times": {},
                "updated_at": datetime.now(timezone.utc),
            },
        )
        if "seen_fragment_times" in query:
            row["seen_fragment_times"].update(json.loads(args[1]))
        else:
            row["summary"] = args[1]
            row["a_start_round"] = args[2]
        row["updated_at"] = datetime.now(timezone.utc)
        return "OK"


class _AcquireConnection:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _MemoryPool:
    def __init__(self):
        self.connection = _MemoryConnection()

    def acquire(self):
        return _AcquireConnection(self.connection)


class SessionResolutionTests(unittest.TestCase):
    def resolve(self, header=None, body=None, partition="legacy", generated="generated"):
        return main.resolve_effective_session_id(
            header,
            body or {},
            partition,
            generated,
        )

    def test_a_distinct_conversation_headers_resolve_to_distinct_sessions(self):
        session_a, source_a = self.resolve("  conversation-A  ")
        session_b, source_b = self.resolve("conversation-B")

        self.assertEqual((session_a, source_a), ("conversation-A", "x_conversation_id"))
        self.assertEqual((session_b, source_b), ("conversation-B", "x_conversation_id"))
        self.assertNotEqual(session_a, session_b)

    def test_b_header_overrides_partition_session(self):
        self.assertEqual(
            self.resolve("header-session", partition="dashboard-active"),
            ("header-session", "x_conversation_id"),
        )

    def test_c_body_session_id_precedes_conversation_id_and_partition_session(self):
        self.assertEqual(
            self.resolve(
                " ",
                {"session_id": " body-session ", "conversation_id": "body-conversation"},
                "dashboard-active",
            ),
            ("body-session", "body_session_id"),
        )

    def test_d_partition_session_remains_legacy_fallback(self):
        self.assertEqual(
            self.resolve("", {"session_id": "", "conversation_id": " "}, "dashboard-active"),
            ("dashboard-active", "partition_session_id"),
        )

    def test_body_conversation_id_precedes_partition_session(self):
        self.assertEqual(
            self.resolve("", {"conversation_id": " body-conversation "}, "dashboard-active"),
            ("body-conversation", "body_conversation_id"),
        )

    def test_empty_explicit_sessions_use_generated_fallback(self):
        self.assertEqual(
            self.resolve(" ", {"session_id": None, "conversation_id": ""}, " ", "new-random"),
            ("new-random", "generated"),
        )

    def test_dashboard_active_session_semantics_are_unchanged(self):
        with patch.object(main, "PARTITION_SESSION_ID", "dashboard-active"):
            self.assertEqual(main.get_active_session_id(), "dashboard-active")

    def test_route_reads_header_and_logs_only_session_hash(self):
        history_session_ids = []
        recall_session_ids = []
        seen_session_ids = []

        async def fake_get_conversation_messages(session_id, limit=10000):
            history_session_ids.append(session_id)
            return []

        async def fake_get_session_cache_state(session_id, seen_ttl_hours=None):
            return {
                "summary_parts": [],
                "a_start_round": 0,
                "seen_fragment_ids": [],
                "updated_at": None,
            }

        async def fake_get_system_prompt():
            return ""

        async def fake_build_conversation_recall_text(user_message, session_id):
            recall_session_ids.append(session_id)
            return "", ["fragment-1"]

        async def fake_mark_fragments_seen(session_id, fragment_ids, ttl_hours):
            seen_session_ids.append(session_id)
            return len(fragment_ids)

        request = _FakeRequest(
            {"messages": [{"role": "assistant", "content": "context"}], "model": "test-model"},
            {"X-Conversation-Id": "  header-session-secret  "},
        )
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "CACHE_PARTITION_ENABLED", True))
            stack.enter_context(patch.object(main, "PARTITION_SESSION_ID", "dashboard-active"))
            stack.enter_context(patch.object(main, "MEMORY_ENABLED", False))
            stack.enter_context(patch.object(main, "MEMORY_EXTRACT_ENABLED", False))
            stack.enter_context(patch.object(main, "get_conversation_messages", fake_get_conversation_messages))
            stack.enter_context(patch.object(main, "get_session_cache_state", fake_get_session_cache_state))
            stack.enter_context(patch.object(main, "get_system_prompt", fake_get_system_prompt))
            stack.enter_context(patch.object(main, "build_conversation_recall_text", fake_build_conversation_recall_text))
            stack.enter_context(patch.object(main, "mark_fragments_seen", fake_mark_fragments_seen))
            stack.enter_context(patch.object(main.httpx, "AsyncClient", _FakeAsyncClient))
            stack.enter_context(redirect_stdout(output))
            response = asyncio.run(main._chat_completions_inner(request))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(history_session_ids, ["header-session-secret"])
        self.assertEqual(recall_session_ids, ["header-session-secret"])
        self.assertEqual(seen_session_ids, ["header-session-secret"])
        log_text = output.getvalue()
        self.assertIn(
            "partition_session source=x_conversation_id id_hash=6e63ddf510c6",
            log_text,
        )
        self.assertNotIn("header-session-secret", log_text)


class SessionStateIsolationTests(unittest.TestCase):
    def test_e_summary_cursor_and_seen_state_are_isolated_by_resolved_session(self):
        pool = _MemoryPool()

        async def scenario():
            session_a, _ = main.resolve_effective_session_id("conversation-A", {}, "legacy", "generated-A")
            session_b, _ = main.resolve_effective_session_id("conversation-B", {}, "legacy", "generated-B")

            await database.save_session_cache_state(session_a, ["summary-A"], 15)
            await database.mark_fragments_seen(session_a, ["fragment-A"], 6)
            await database.save_session_cache_state(session_b, ["summary-B"], 30)
            await database.mark_fragments_seen(session_b, ["fragment-B"], 6)

            state_a = await database.get_session_cache_state(session_a, 6)
            state_b = await database.get_session_cache_state(session_b, 6)
            return state_a, state_b

        with patch.object(database, "get_pool", return_value=pool):
            state_a, state_b = asyncio.run(scenario())

        self.assertEqual(state_a["summary_parts"], ["summary-A"])
        self.assertEqual(state_a["a_start_round"], 15)
        self.assertEqual(state_a["seen_fragment_ids"], ["fragment-A"])
        self.assertEqual(state_b["summary_parts"], ["summary-B"])
        self.assertEqual(state_b["a_start_round"], 30)
        self.assertEqual(state_b["seen_fragment_ids"], ["fragment-B"])

    def test_conversation_recall_reuses_session_scoped_seen_state(self):
        state_session_ids = []
        search_arguments = []

        async def fake_get_session_cache_state(session_id, seen_ttl_hours=None):
            state_session_ids.append(session_id)
            return {"seen_fragment_ids": ["seen-for-A"]}

        async def fake_search_chat_fragments(query, **kwargs):
            search_arguments.append((query, kwargs))
            return [], 0

        with ExitStack() as stack:
            stack.enter_context(patch.object(main._db_module, "CONVERSATION_RECALL_ENABLED", True))
            stack.enter_context(patch.object(main, "MAX_CONVERSATIONS_INJECT", 3))
            stack.enter_context(patch.object(main, "get_session_cache_state", fake_get_session_cache_state))
            stack.enter_context(patch.object(main, "search_chat_fragments", fake_search_chat_fragments))
            recall_text, fragment_ids = asyncio.run(
                main.build_conversation_recall_text("query", "conversation-A")
            )

        self.assertEqual((recall_text, fragment_ids), ("", []))
        self.assertEqual(state_session_ids, ["conversation-A"])
        self.assertEqual(search_arguments[0][1]["exclude_session_ids"], ["conversation-A"])
        self.assertEqual(search_arguments[0][1]["exclude_fragment_ids"], ["seen-for-A"])


class PartitionGrowthCompatibilityTests(unittest.TestCase):
    def test_f_summary_failure_advances_with_bounded_marker_and_strips_reasoning(self):
        saved_states = []

        async def fake_get_session_cache_state(session_id):
            return {"summary_parts": [], "a_start_round": 0}

        async def fake_generate_summary(messages, session_id):
            return ""

        async def fake_save_session_cache_state(session_id, summary_parts, a_start_round):
            saved_states.append((session_id, list(summary_parts), a_start_round))

        history = []
        for index in range(4):
            history.extend([
                {"role": "user", "content": f"user-{index}", "reasoning": f"user-secret-{index}"},
                {"role": "assistant", "content": f"assistant-{index}", "reasoning_content": f"assistant-secret-{index}"},
            ])
        history.append({"role": "user", "content": "current"})

        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "CACHE_PARTITION_X", 2))
            stack.enter_context(patch.object(main, "CACHE_PARTITION_TRIGGER", "rounds"))
            stack.enter_context(patch.object(main, "CACHE_SUMMARY_MODEL", "summary-model"))
            stack.enter_context(patch.object(main, "MEMORY_ENABLED", False))
            stack.enter_context(patch.object(main, "get_session_cache_state", fake_get_session_cache_state))
            stack.enter_context(patch.object(main, "generate_summary", fake_generate_summary))
            stack.enter_context(patch.object(main, "save_session_cache_state", fake_save_session_cache_state))
            output = io.StringIO()
            stack.enter_context(redirect_stdout(output))
            result = asyncio.run(main.build_partitioned_messages(
                "header-session-secret",
                copy.deepcopy(history),
                "system prompt",
                "current",
            ))

        self.assertEqual(len(saved_states), 1)
        self.assertEqual(saved_states[0][2], 2)
        self.assertEqual(len(saved_states[0][1]), 1)
        marker = saved_states[0][1][0]
        self.assertLess(len(marker), 200)
        self.assertNotIn("user-0", marker)
        self.assertNotIn("assistant-0", marker)
        self.assertNotIn("header-session-secret", output.getvalue())
        self.assertIn("id_hash=6e63ddf510c6", output.getvalue())
        for message in result:
            self.assertNotIn("reasoning_content", message)
            self.assertNotIn("reasoning", message)

    def test_f_basic_cache_strips_reasoning_fields(self):
        history = [
            {"role": "user", "content": "user", "reasoning": "user-secret"},
            {"role": "assistant", "content": "assistant", "reasoning_content": "assistant-secret"},
        ]

        result = asyncio.run(main._build_basic_cached(
            copy.deepcopy(history),
            "system prompt",
            "",
            None,
        ))

        for message in result:
            self.assertNotIn("reasoning_content", message)
            self.assertNotIn("reasoning", message)


if __name__ == "__main__":
    unittest.main()
