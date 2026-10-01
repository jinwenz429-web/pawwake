import asyncio
import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import main


class _FakeRequest:
    def __init__(self, body):
        self._body = body
        self.headers = {"X-Conversation-Id": "tool-test-session"}

    async def json(self):
        return copy.deepcopy(self._body)


class _FakeResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "done"}}]}


class _RecordingAsyncClient:
    posted_bodies = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        self.posted_bodies.append(copy.deepcopy(kwargs["json"]))
        return _FakeResponse()


def _tool_call(call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": f"tool_{call_id}", "arguments": "{}"},
    }


def _db_history(tool_calls=None):
    if tool_calls is None:
        tool_calls = [_tool_call("call_a"), _tool_call("call_b"), _tool_call("call_c")]
    return [
        {
            "role": "user",
            "content": "run tools",
            "metadata": None,
            "created_at": None,
        },
        {
            "role": "assistant",
            "content": "",
            "metadata": json.dumps(
                {"tool_calls": tool_calls}
            ),
            "created_at": None,
        },
    ]


async def _identity_partition_messages(session_id, all_messages, *args, **kwargs):
    return all_messages


class ToolResultReconciliationTests(unittest.TestCase):
    def setUp(self):
        _RecordingAsyncClient.posted_bodies = []

    def _run_request(self, tool_messages, tool_calls=None, real_partition=False, prior_rounds=0,
                     force_time_rotation=False):
        async def fake_get_conversation_messages(session_id, limit=10000):
            history = []
            for index in range(prior_rounds):
                history.extend([
                    {"role": "user", "content": f"earlier question {index}", "metadata": None, "created_at": None},
                    {"role": "assistant", "content": f"earlier answer {index}", "metadata": None, "created_at": None},
                ])
            return history + _db_history(tool_calls)

        async def fake_get_session_cache_state(session_id):
            return {"summary_parts": [], "a_start_round": 0}

        def fake_db_row_to_message(row):
            message = {"role": row["role"], "content": row["content"]}
            if row["metadata"]:
                message.update(json.loads(row["metadata"]))
            return message

        request = _FakeRequest(
            {
                "model": "gemini-test",
                "stream": False,
                "messages": tool_messages,
            }
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "CACHE_PARTITION_ENABLED", True))
            stack.enter_context(patch.object(main, "MEMORY_ENABLED", False))
            stack.enter_context(patch.object(main, "MEMORY_EXTRACT_ENABLED", False))
            stack.enter_context(patch.object(main, "FORCE_STREAM", False))
            stack.enter_context(patch.object(main, "REASONING_EFFORT", ""))
            stack.enter_context(patch.object(main, "get_system_prompt", unittest.mock.AsyncMock(return_value="")))
            stack.enter_context(patch.object(main, "get_conversation_messages", fake_get_conversation_messages))
            stack.enter_context(patch.object(main, "db_row_to_message", fake_db_row_to_message))
            stack.enter_context(patch.object(main, "build_conversation_recall_text", unittest.mock.AsyncMock(return_value=("", []))))
            if real_partition:
                stack.enter_context(patch.object(main, "CACHE_PARTITION_X", 15))
                stack.enter_context(patch.object(main, "get_session_cache_state", fake_get_session_cache_state))
                if force_time_rotation:
                    stack.enter_context(patch.object(main, "CACHE_PARTITION_TRIGGER", "time"))
                    stack.enter_context(patch.object(main, "CACHE_MAX_ROTATIONS", 1))
                    stack.enter_context(patch.object(main, "_should_rotate", return_value=True))
                    stack.enter_context(patch.object(main, "generate_summary", unittest.mock.AsyncMock(return_value="summary")))
                    stack.enter_context(patch.object(main, "save_session_cache_state", unittest.mock.AsyncMock()))
            else:
                stack.enter_context(patch.object(main, "build_partitioned_messages", _identity_partition_messages))
            stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=False))
            stack.enter_context(patch.object(main.httpx, "AsyncClient", _RecordingAsyncClient))
            return asyncio.run(main._chat_completions_inner(request))

    def test_incomplete_current_tool_turn_is_rejected_before_upstream(self):
        response = self._run_request(
            [
                {"role": "tool", "tool_call_id": "old_call", "content": "old"},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
                {"role": "tool", "tool_call_id": "call_b", "content": "result-b"},
            ]
        )

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(payload["error"]["type"], "incomplete_tool_results")
        self.assertEqual(payload["error"]["expected_count"], 3)
        self.assertEqual(payload["error"]["received_count"], 2)
        self.assertEqual(payload["error"]["missing_tool_call_ids"], ["call_c"])
        self.assertEqual(payload["error"]["duplicate_tool_call_ids"], [])
        self.assertEqual(_RecordingAsyncClient.posted_bodies, [])

    def test_error_result_with_matching_call_id_completes_turn(self):
        error_content = json.dumps({"error": "permission denied"})
        response = self._run_request(
            [
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
                {"role": "tool", "tool_call_id": "call_b", "content": "result-b"},
                {"role": "tool", "tool_call_id": "call_c", "content": error_content},
            ]
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(_RecordingAsyncClient.posted_bodies), 1)
        forwarded_tools = [
            message
            for message in _RecordingAsyncClient.posted_bodies[0]["messages"]
            if message.get("role") == "tool"
        ]
        self.assertEqual([message["tool_call_id"] for message in forwarded_tools], ["call_a", "call_b", "call_c"])
        self.assertEqual(forwarded_tools[2]["content"], error_content)

    def test_duplicate_current_tool_result_is_rejected_before_upstream(self):
        response = self._run_request(
            [
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a-1"},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a-2"},
                {"role": "tool", "tool_call_id": "call_b", "content": "result-b"},
                {"role": "tool", "tool_call_id": "call_c", "content": "result-c"},
            ]
        )

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(payload["error"]["type"], "incomplete_tool_results")
        self.assertEqual(payload["error"]["missing_tool_call_ids"], [])
        self.assertEqual(payload["error"]["duplicate_tool_call_ids"], ["call_a"])
        self.assertEqual(_RecordingAsyncClient.posted_bodies, [])

    def test_client_trimmed_failed_call_gets_unknown_result(self):
        response = self._run_request(
            [
                {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_a"), _tool_call("call_b")]},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
                {"role": "tool", "tool_call_id": "call_b", "content": "result-b"},
            ]
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual([m["tool_call_id"] for m in forwarded if m["role"] == "tool"],
                         ["call_a", "call_b", "call_c"])
        recovered = json.loads(forwarded[-1]["content"])
        self.assertEqual(recovered["error"], "client_omitted_tool_result")
        self.assertEqual(recovered["status"], "unknown")
        self.assertIs(recovered["retryable"], False)
        self.assertEqual(forwarded[-1]["name"], "tool_call_c")

    def test_two_calls_one_result_recovers_with_real_partition_builder(self):
        battery_call = _tool_call("call_b")
        battery_call["function"]["name"] = "get_battery"
        response = self._run_request(
            [
                {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_a")]},
                {"role": "tool", "tool_call_id": "call_a", "content": "screen time result"},
            ],
            tool_calls=[_tool_call("call_a"), battery_call],
            real_partition=True,
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual([m["tool_call_id"] for m in forwarded if m["role"] == "tool"],
                         ["call_a", "call_b"])
        battery_result = next(m for m in forwarded if m.get("tool_call_id") == "call_b")
        self.assertEqual(battery_result["name"], "get_battery")
        self.assertEqual(json.loads(battery_result["content"])["status"], "unknown")

    def test_recovered_tool_turn_survives_partition_a_boundary(self):
        response = self._run_request(
            [
                {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_a")]},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
            ],
            tool_calls=[_tool_call("call_a"), _tool_call("call_b")],
            real_partition=True,
            prior_rounds=14,
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual(
            [call["id"] for message in forwarded for call in message.get("tool_calls", [])],
            ["call_a", "call_b"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in forwarded if message.get("role") == "tool"],
            ["call_a", "call_b"],
        )
        missing_result = next(message for message in forwarded if message.get("tool_call_id") == "call_b")
        self.assertEqual(json.loads(missing_result["content"])["status"], "unknown")

    def test_recovered_tool_turn_precedes_new_user_at_partition_a_boundary(self):
        response = self._run_request(
            [
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
                {"role": "user", "content": "continue after the error"},
            ],
            tool_calls=[_tool_call("call_a"), _tool_call("call_b")],
            real_partition=True,
            prior_rounds=14,
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual(
            [(message["role"], message.get("tool_call_id")) for message in forwarded[-4:]],
            [("assistant", None), ("tool", "call_a"), ("tool", "call_b"), ("user", None)],
        )
        self.assertEqual(
            [call["id"] for call in forwarded[-4]["tool_calls"]],
            ["call_a", "call_b"],
        )

    def test_recovered_tool_turn_is_not_rotated_into_summary(self):
        response = self._run_request(
            [
                {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_a")]},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
            ],
            tool_calls=[_tool_call("call_a"), _tool_call("call_b")],
            real_partition=True,
            prior_rounds=14,
            force_time_rotation=True,
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual(
            [message["tool_call_id"] for message in forwarded if message.get("role") == "tool"],
            ["call_a", "call_b"],
        )

    def test_matching_error_result_survives_partition_a_boundary(self):
        error_content = json.dumps({"error": "get_battery timed out"})
        response = self._run_request(
            [
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
                {"role": "tool", "tool_call_id": "call_b", "content": error_content},
            ],
            tool_calls=[_tool_call("call_a"), _tool_call("call_b")],
            real_partition=True,
            prior_rounds=14,
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual(
            [message["tool_call_id"] for message in forwarded if message.get("role") == "tool"],
            ["call_a", "call_b"],
        )
        error_result = next(message for message in forwarded if message.get("tool_call_id") == "call_b")
        self.assertEqual(error_result["content"], error_content)

    def test_new_user_closes_pending_turn_before_user_message(self):
        response = self._run_request(
            [
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
                {"role": "user", "content": "Can we continue?"},
            ]
        )

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual([m["role"] for m in forwarded],
                         ["user", "assistant", "tool", "tool", "tool", "user"])
        self.assertEqual([m["tool_call_id"] for m in forwarded if m["role"] == "tool"],
                         ["call_a", "call_b", "call_c"])
        self.assertEqual(forwarded[-1]["content"], "Can we continue?")

    def test_new_user_with_no_tool_results_closes_pending_turn(self):
        response = self._run_request([{"role": "user", "content": "Please continue"}])

        self.assertEqual(response.status_code, 200)
        forwarded = _RecordingAsyncClient.posted_bodies[0]["messages"]
        self.assertEqual([m["tool_call_id"] for m in forwarded if m["role"] == "tool"],
                         ["call_a", "call_b", "call_c"])
        self.assertEqual(forwarded[-1]["content"], "Please continue")

    def test_duplicate_result_stays_rejected_after_client_trims_call(self):
        response = self._run_request(
            [
                {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_a"), _tool_call("call_b")]},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a-1"},
                {"role": "tool", "tool_call_id": "call_a", "content": "result-a-2"},
                {"role": "tool", "tool_call_id": "call_b", "content": "result-b"},
            ]
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(_RecordingAsyncClient.posted_bodies, [])

    def test_followup_user_is_saved_after_recovered_tool_results(self):
        saved = []

        async def fake_save_message(session_id, role, content, model, metadata=None):
            saved.append((role, content))

        context = [
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_a")]},
            {"role": "tool", "tool_call_id": "call_a", "content": "result-a"},
            {"role": "user", "content": "Please continue"},
        ]
        with (
            patch.object(main, "MEMORY_ENABLED", False),
            patch.object(main, "save_message", fake_save_message),
        ):
            asyncio.run(main.process_memories_background(
                "tool-test-session",
                "Please continue",
                "done",
                "gemini-test",
                context_messages=context,
                tool_messages=[{"role": "tool", "tool_call_id": "call_a", "content": "result-a"}],
            ))

        self.assertEqual(saved, [
            ("tool", "result-a"),
            ("user", "Please continue"),
            ("assistant", "done"),
        ])


if __name__ == "__main__":
    unittest.main()
