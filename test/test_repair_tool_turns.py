import json
import unittest

import main


def _call(call_id, name="search_memory"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _result(call_id, content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _ids(messages):
    return [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]


class RepairToolTurnsTest(unittest.TestCase):
    def test_complete_turn_is_unchanged(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [_call("a"), _call("b")]},
            _result("a"),
            _result("b"),
            {"role": "assistant", "content": "done"},
        ]
        self.assertEqual(main.repair_tool_turns(history), history)

    def test_missing_result_is_filled_in_call_order(self):
        history = [
            {"role": "assistant", "content": "", "tool_calls": [_call("a"), _call("b"), _call("c")]},
            _result("c"),
            _result("a"),
            {"role": "assistant", "content": "done"},
        ]
        repaired = main.repair_tool_turns(history)
        self.assertEqual(_ids(repaired), ["a", "b", "c"])
        filler = repaired[2]
        self.assertEqual(json.loads(filler["content"])["error"], "tool_result_missing")
        self.assertEqual(repaired[-1]["content"], "done")

    def test_duplicates_and_strays_are_dropped(self):
        history = [
            {"role": "assistant", "content": "", "tool_calls": [_call("a")]},
            _result("a", "old"),
            _result("a", "new"),
            _result("zzz"),
            {"role": "user", "content": "next"},
            _result("orphan"),
        ]
        repaired = main.repair_tool_turns(history)
        self.assertEqual(_ids(repaired), ["a"])
        self.assertEqual(repaired[1]["content"], "new")
        self.assertEqual(repaired[-1], {"role": "user", "content": "next"})

    def test_calls_without_ids_match_by_position(self):
        history = [
            {"role": "assistant", "content": "", "tool_calls": [_call(None), _call(None)]},
            _result(None),
            _result(None),
            _result(None),
        ]
        repaired = main.repair_tool_turns(history)
        self.assertEqual(len([m for m in repaired if m["role"] == "tool"]), 2)


if __name__ == "__main__":
    unittest.main()
