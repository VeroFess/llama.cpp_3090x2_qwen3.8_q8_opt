#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("protocol_fixtures.py")
SPEC = importlib.util.spec_from_file_location("qwen38_protocol_fixtures", MODULE_PATH)
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


class ProtocolFixtureTests(unittest.TestCase):
    def test_sse_split_handles_lf_and_crlf(self):
        frames, pending = FIXTURES.split_sse_frames(b"data: {\"a\":1}\n\ndata: [DONE]\r\n\r\npartial")
        self.assertEqual(len(frames), 2)
        self.assertEqual(pending, b"partial")
        self.assertEqual(FIXTURES.parse_sse_frame(frames[0])["data"], {"a": 1})
        self.assertEqual(FIXTURES.parse_sse_frame(frames[1])["data"], "[DONE]")

    def test_openai_history_covers_tool_protocol(self):
        messages, kwargs = FIXTURES.openai_history(False, True)
        self.assertTrue(kwargs["parallel_tool_calls"])
        self.assertTrue(kwargs["preserve_thinking"])
        self.assertEqual([message["role"] for message in messages], [
            "user", "assistant", "tool", "tool", "assistant", "tool", "user",
        ])
        self.assertEqual(len(messages[1]["tool_calls"]), 2)
        self.assertIsInstance(messages[1]["tool_calls"][0]["function"]["arguments"], dict)

    def test_openai_string_arguments_are_valid_json(self):
        messages, _ = FIXTURES.openai_history(True, False)
        for assistant in (messages[1], messages[4]):
            for call in assistant["tool_calls"]:
                self.assertIsInstance(json.loads(call["function"]["arguments"]), dict)

    def test_run_fixture_records_failure(self):
        result = FIXTURES.run_fixture("failure", ["expected"], lambda: FIXTURES.require(False, "failed"))
        self.assertFalse(result["success"])
        self.assertIn("failed", result["error"])


if __name__ == "__main__":
    unittest.main()
