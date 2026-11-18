import json
import unittest
from unittest.mock import patch

import memory_extractor


class _Response:
    status_code = 200
    payload = None

    def json(self):
        return {
            "choices": [{
                "message": {"content": json.dumps(self.payload or [
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
        self.assertRegex(client.sent["messages"][1]["content"],
                         r"\[消息 17｜\d{4}-\d{2}-\d{2} 周. \d{2}:\d{2}\] AI：")
        self.assertIn("只记对方的事", client.sent["messages"][0]["content"])

    async def test_messages_carry_send_time_and_prompt_states_now(self):
        from datetime import datetime, timezone
        client = _Client()
        now = datetime(2026, 11, 18, 15, 40, tzinfo=timezone.utc)  # 东八区 23:40 周三
        with (
            patch.object(memory_extractor, "get_memory_api_key", return_value="test"),
            patch.object(memory_extractor.httpx, "AsyncClient", return_value=client),
            patch.dict("os.environ", {"TIMEZONE_HOURS": "8"}),
        ):
            await memory_extractor.extract_memories([
                {"role": "user", "content": "昨天和妈妈吵架了", "source_id": 12,
                 "created_at": datetime(2026, 11, 17, 14, 5, tzinfo=timezone.utc)},
                {"role": "tool", "content": "不该出现", "source_id": 13},
                {"role": "assistant", "content": "抱抱", "source_id": None},
            ], existing_memories=[{"id": 123, "content": "喜欢喝咖啡",
                                   "created_at": datetime(2026, 10, 2, 3, 0, tzinfo=timezone.utc)}],
                now=now)
        system = client.sent["messages"][0]["content"]
        body = client.sent["messages"][1]["content"]
        self.assertIn("现在是 2026-11-18 周三 23:40", system)
        self.assertIn("[#123｜2026-10-02] 喜欢喝咖啡", system)
        self.assertIn("[消息 12｜2026-11-17 周二 22:05] 用户：昨天和妈妈吵架了", body)
        self.assertIn("[无编号｜2026-11-18 周三 23:40] AI：抱抱", body)
        self.assertNotIn("不该出现", body)

    async def test_output_fields_are_parsed(self):
        client = _Client()
        _Response.payload = [
            {"content": "计划 2026-12-20 考研初试", "importance": 9, "source_ids": [1],
             "kind": "plan", "event_date": "2026-12-20", "action": "new"},
            {"content": "以前喜欢咖啡，2026-11 起改喝茶", "importance": 5, "source_ids": [1],
             "kind": "fact", "action": "update", "target_id": "123"},
            {"content": "坏字段", "importance": "x", "source_ids": [1], "kind": "gossip",
             "event_date": "下周", "action": "update"},
        ]
        try:
            with (
                patch.object(memory_extractor, "get_memory_api_key", return_value="test"),
                patch.object(memory_extractor.httpx, "AsyncClient", return_value=client),
            ):
                memories = await memory_extractor.extract_memories([
                    {"role": "user", "content": "x", "source_id": 1}])
        finally:
            _Response.payload = None
        from datetime import date
        self.assertEqual((memories[0]["kind"], memories[0]["event_date"], memories[0]["action"]),
                         ("plan", date(2026, 12, 20), "new"))
        self.assertEqual((memories[1]["action"], memories[1]["target_id"]), ("update", 123))
        self.assertEqual((memories[2]["kind"], memories[2]["event_date"], memories[2]["action"],
                          memories[2]["importance"]), (None, None, "new", 5))


if __name__ == "__main__":
    unittest.main()
