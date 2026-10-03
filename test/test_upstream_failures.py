import asyncio
import copy
import io
import json
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx
import main


SUBMISSION_ERROR = (
    "The prompt could not be submitted. The prompt contains sensitive words "
    "that violate Google's Generative AI Prohibited Use policy. "
    "Try rephrasing the prompt. If you think this was an error, send feedback."
)
REAL_ASYNC_CLIENT = httpx.AsyncClient


def completion(content, finish_reason="stop", **message_fields):
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": content, **message_fields,
        }, "finish_reason": finish_reason}],
    }


def event(delta=None, finish_reason=None, **fields):
    payload = {"choices": [{"index": 0, "delta": delta or {},
                            "finish_reason": finish_reason}], **fields}
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class Request:
    headers = {"X-Conversation-Id": "test-conversation"}

    async def json(self):
        return {"model": "test-model", "stream": False,
                "messages": [{"role": "user", "content": "hello"}]}


class UpstreamFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saved = []
        self.seen = []
        self.upstream_requests = []

    def boundaries(self, response):
        async def save_message(session_id, role, content, model, metadata=None):
            self.saved.append({"role": role, "content": content, "metadata": metadata})
            return len(self.saved)

        async def mark_seen(session_id, fragment_ids, ttl_hours):
            self.seen.extend(fragment_ids)

        def upstream(request):
            self.upstream_requests.append(json.loads(request.content))
            return response

        stack = ExitStack()
        for name, value in {
            "CACHE_PARTITION_ENABLED": False,
            "MEMORY_ENABLED": False,
            "MEMORY_EXTRACT_ENABLED": False,
            "FORCE_STREAM": False,
            "REASONING_EFFORT": "",
        }.items():
            stack.enter_context(patch.object(main, name, value))
        stack.enter_context(patch.object(main, "get_system_prompt", AsyncMock(return_value="")))
        stack.enter_context(patch.object(main, "conversation_persistence_enabled", return_value=True))
        stack.enter_context(patch.object(main, "save_message", save_message))
        stack.enter_context(patch.object(main, "mark_fragments_seen", mark_seen))
        stack.enter_context(patch.object(main.httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(upstream), **kwargs)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    async def nonstream(self, payload, status=200):
        with self.boundaries(httpx.Response(status, json=copy.deepcopy(payload))):
            response = await main._chat_completions_inner(Request())
            await asyncio.sleep(0)
        self.assertEqual(len(self.upstream_requests), 1, "failed requests must not be retried")
        self.assertEqual(self.upstream_requests[0]["model"], "test-model")
        return response.status_code, json.loads(response.body)

    async def stream(self, chunks, status=200, content_type="text/event-stream"):
        response = httpx.Response(status, headers={"content-type": content_type},
                                  stream=ByteStream(chunks))
        with self.boundaries(response):
            output = b"".join([chunk async for chunk in main.stream_and_capture(
                {}, {"model": "test-model", "stream": True,
                     "messages": [{"role": "user", "content": "hello"}]},
                "test-conversation", "hello", "test-model",
                pending_fragment_ids=["fragment-1"],
            )])
            await asyncio.sleep(0)
        self.assertEqual(len(self.upstream_requests), 1, "failed requests must not be retried")
        return output

    def assert_stream_error(self, output, code):
        payloads = [json.loads(line[5:].strip()) for line in output.decode().splitlines()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]"]
        errors = [payload["error"] for payload in payloads if "error" in payload]
        self.assertTrue(errors, "client must receive an explicit SSE error")
        self.assertEqual(errors[-1]["code"], code)
        self.assertEqual(self.saved, [], "failed responses must not enter conversation storage")
        self.assertEqual(self.seen, [], "failed requests must not consume recalled fragments")

    async def test_submission_error_in_success_envelope_is_reported_as_failure(self):
        status, payload = await self.nonstream(completion(SUBMISSION_ERROR))
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "content_filter")
        self.assertEqual(self.saved, [])

    async def test_error_envelope_with_http_200_does_not_save_empty_reply(self):
        status, payload = await self.nonstream({"error": {
            "message": "blocked by provider", "type": "content_filter", "code": "content_filter",
        }})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["message"], "blocked by provider")
        self.assertEqual(self.saved, [])

    async def test_empty_completion_is_reported_as_failure(self):
        status, payload = await self.nonstream(completion(""))
        self.assertEqual(status, 502)
        self.assertEqual(payload["error"]["code"], "upstream_empty_response")
        self.assertEqual(self.saved, [])

    async def test_filtered_partial_output_is_not_saved(self):
        status, payload = await self.nonstream(completion("partial", "content_filter"))
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "content_filter")
        self.assertEqual(self.saved, [])

    async def test_native_prompt_block_feedback_is_not_saved(self):
        status, payload = await self.nonstream({"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "content_filter")
        self.assertEqual(self.saved, [])

    async def test_normal_assistant_refusal_remains_a_response(self):
        text = "I cannot continue that description, but I can help with a non-explicit scene."
        status, payload = await self.nonstream(completion(text))
        self.assertEqual(status, 200)
        self.assertEqual(payload["choices"][0]["message"]["content"], text)
        self.assertEqual(self.saved[-1]["content"], text)

    async def test_quoted_error_explanation_is_not_misclassified(self):
        text = "This provider error means submission failed: " + SUBMISSION_ERROR
        status, _ = await self.nonstream(completion(text))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved[-1]["content"], text)

    async def test_tool_only_completion_is_valid(self):
        call = {"id": "call-1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city":"Taipei"}',
        }}
        status, _ = await self.nonstream(completion(None, "tool_calls", tool_calls=[call]))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(self.saved[-1]["metadata"])["tool_calls"], [call])

    async def test_fragmented_submission_error_stream_does_not_enter_memory(self):
        chunks = [event({"content": SUBMISSION_ERROR[:28]}),
                  event({"content": SUBMISSION_ERROR[28:100]}),
                  event({"content": SUBMISSION_ERROR[100:]}),
                  event(finish_reason="stop"), b"data: [DONE]\n\n"]
        self.assert_stream_error(await self.stream(chunks), "content_filter")

    async def test_sse_error_after_partial_output_does_not_enter_memory(self):
        chunks = [event({"content": "partial"}),
                  b'data: {"error":{"message":"provider failed","code":"upstream_error"}}\n\n',
                  b"data: [DONE]\n\n"]
        self.assert_stream_error(await self.stream(chunks), "upstream_error")

    async def test_non_200_json_in_stream_request_is_exposed_as_sse_error(self):
        raw = b'{"error":{"message":"quota exhausted","code":429}}'
        output = await self.stream([raw], 429, "application/json")
        self.assert_stream_error(output, 429)

    async def test_http_200_json_error_in_stream_request_is_not_silent(self):
        raw = b'{"error":{"message":"blocked","code":"content_filter"}}'
        self.assert_stream_error(await self.stream([raw], content_type="application/json"),
                                 "content_filter")

    async def test_empty_sse_completion_is_an_error(self):
        self.assert_stream_error(await self.stream([b"data: [DONE]\n\n"]),
                                 "upstream_empty_response")

    async def test_unfinished_stream_is_not_saved_as_complete_reply(self):
        self.assert_stream_error(await self.stream([event({"content": "partial"})]),
                                 "upstream_incomplete_response")

    async def test_utf8_split_at_every_byte_preserves_saved_response_and_wire_data(self):
        raw = event({"content": "你好，今天心情怎样？"}) + event(finish_reason="stop") + b"data: [DONE]\n\n"
        output = await self.stream([raw[i:i+1] for i in range(len(raw))])
        self.assertEqual(output, raw)
        self.assertEqual(self.saved[-1]["content"], "你好，今天心情怎样？")
        self.assertEqual(self.seen, ["fragment-1"])

    async def test_tool_call_stream_keeps_function_and_arguments(self):
        first = {"index": 0, "id": "call-1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city":',
        }}
        second = {"index": 0, "function": {"arguments": '"Taipei"}'}}
        raw = event({"tool_calls": [first]}) + event({"tool_calls": [second]}) + event(
            finish_reason="tool_calls") + b"data: [DONE]\n\n"
        self.assertEqual(await self.stream([raw]), raw)
        calls = json.loads(self.saved[-1]["metadata"])["tool_calls"]
        self.assertEqual(calls[0]["function"], {"name": "get_weather", "arguments": '{"city":"Taipei"}'})


if __name__ == "__main__":
    unittest.main()
