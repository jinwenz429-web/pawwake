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


def _db_history():
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
                {"tool_calls": [_tool_call("call_a"), _tool_call("call_b"), _tool_call("call_c")]}
            ),
            "created_at": None,
        },
    ]


async def _identity_partition_messages(session_id, all_messages, *args, **kwargs):
    return all_messages


class ToolResultReconciliationTests(unittest.TestCase):
    def setUp(self):
        _RecordingAsyncClient.posted_bodies = []

    def _run_request(self, tool_messages):
        async def fake_get_conversation_messages(session_id, limit=10000):
            return _db_history()

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


if __name__ == "__main__":
    unittest.main()
