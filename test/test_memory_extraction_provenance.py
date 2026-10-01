import json
import unittest
from unittest.mock import patch

import memory_extractor


class _Response:
    status_code = 200

    def json(self):
        return {
            "choices": [{
                "message": {"content": json.dumps([
                    {"content": "A lasting promise", "importance": 8,
                     "source_ids": [17]},
                ])},
                "finish_reason": "stop",
            }],
        }


class _Client:
    sent = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        self.sent = kwargs["json"]
        return _Response()


class MemoryExtractionProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_requests_and_preserves_message_sources(self):
        client = _Client()
        with (
            patch.object(memory_extractor, "get_memory_api_key", return_value="test"),
            patch.object(memory_extractor.httpx, "AsyncClient", return_value=client),
        ):
            memories = await memory_extractor.extract_memories([
                {"role": "assistant", "content": "I will remember this",
                 "source_id": 17},
            ])
        self.assertEqual(memories[0]["source_ids"], [17])
        self.assertIn("[消息 17] AI:", client.sent["messages"][1]["content"])
        self.assertIn("安全审查", client.sent["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
