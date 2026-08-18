#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest
from types import SimpleNamespace


MODULE_PATH = pathlib.Path(__file__).with_name("bench.py")
SPEC = importlib.util.spec_from_file_location("qwen38_bench", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(BENCH.percentile([], 0.95), 0.0)
        self.assertEqual(BENCH.percentile([3, 1, 2], 0.50), 2)
        self.assertEqual(BENCH.percentile([3, 1, 2], 0.95), 3)

    def test_catalog_has_required_workloads(self):
        catalog = BENCH.scenario_catalog(False, 256)
        self.assertEqual(set(catalog), set(BENCH.ALL_SCENARIOS))
        self.assertEqual(catalog["single_32k"]["requests"][0]["prompt_tokens_requested"], 32768)
        self.assertEqual(catalog["single_128k"]["requests"][0]["prompt_tokens_requested"], 131072)
        self.assertEqual(catalog["single_262k"]["requests"][0]["prompt_tokens_requested"], 260000)

    def test_shared_prefix_is_exact(self):
        definition = BENCH.shared_prefix_scenario(1024, 32)
        contents = [item["messages"][0]["content"] for item in definition["requests"]]
        prefix = contents[0].split("\nTask branch", 1)[0]
        self.assertTrue(all(content.startswith(prefix) for content in contents))
        self.assertEqual(len(definition["warmups"]), 1)

    def test_coding_agent_fixture_has_tool_history(self):
        definition = BENCH.coding_agent_scenario(256, 32)
        self.assertEqual(len(definition["requests"]), 8)
        for item in definition["requests"]:
            self.assertTrue(item["tools"])
            roles = [message["role"] for message in item["messages"]]
            self.assertIn("assistant", roles)
            self.assertIn("tool", roles)

    def test_request_body_records_axes_independent_request(self):
        spec = BENCH.request_spec(3, 128, output_tokens=7)
        body = BENCH.request_body("model", spec, 32)
        self.assertEqual(body["max_tokens"], 7)
        self.assertEqual(body["temperature"], 0)
        self.assertTrue(body["stream_options"]["include_usage"])

    def test_prompt_calibration_converges(self):
        original = BENCH.tokenize_count
        try:
            BENCH.tokenize_count = lambda base_url, content: len(content) // 3
            content, calibration = BENCH.calibrate_prompt("http://127.0.0.1:8080", 1000, 1)
            self.assertLessEqual(abs(calibration["measured_content_tokens"] - 1000), 4)
            self.assertLessEqual(abs(len(content) - 3000), 3)
            self.assertGreaterEqual(len(calibration["attempts"]), 2)
        finally:
            BENCH.tokenize_count = original

    def test_calibrate_request_reserves_chat_overhead(self):
        spec = BENCH.request_spec(0, 1024, calibrate=True)
        original = BENCH.calibrate_prompt
        try:
            BENCH.calibrate_prompt = lambda base_url, target, request_id: ("body", {
                "target_content_tokens": target,
                "measured_content_tokens": target,
                "attempts": [],
            })
            BENCH.calibrate_request("http://127.0.0.1:8080", spec)
            self.assertEqual(spec["messages"][0]["content"], "body")
            self.assertEqual(spec["calibration"]["target_content_tokens"], 960)
        finally:
            BENCH.calibrate_prompt = original

    def test_runtime_snapshot_shape(self):
        original_command = BENCH.command_output
        original_json = BENCH.safe_get_json
        original_text = BENCH.safe_get_text
        try:
            BENCH.command_output = lambda command, cwd=None: {"ok": True, "stdout": "gpu"}
            BENCH.safe_get_json = lambda base_url, path: {"ok": True, "value": path}
            BENCH.safe_get_text = lambda base_url, path: {"ok": True, "value": path}
            snapshot = BENCH.runtime_snapshot("http://127.0.0.1:8080")
            self.assertEqual(snapshot["gpu_state"]["stdout"], "gpu")
            self.assertEqual(snapshot["server"]["metrics"]["value"], "/metrics")
        finally:
            BENCH.command_output = original_command
            BENCH.safe_get_json = original_json
            BENCH.safe_get_text = original_text

    def test_manifest_validation_matches_runtime(self):
        args = SimpleNamespace(kv_mode="q8_0", split_boundary="36/28", cuda_graph="on", pipeline="on")
        manifest = {
            "source_commit": "abc",
            "model_sha256": "def",
            "server_command": "server",
            "host": {
                "gpu_topology": {"ok": True},
                "gpu_p2p_read": {"ok": True},
                "gpu_p2p_write": {"ok": True},
            },
            "server": {"profile": {"ok": True, "value": {
                "kv_cache_type_k": "q8_0",
                "kv_cache_type_v": "q8_0",
                "layer_boundary": {"gpu0": 36, "gpu1": 28},
            }}},
            "after": {"server": {"graphs": {"ok": True, "value": {
                "hits": 10,
                "captures": 2,
                "pipeline_transfer": {
                    "mode": "pinned_host_staged",
                    "depth": 2,
                    "direct_peer_transfers": 0,
                    "d2h_bytes": 1024,
                    "h2d_bytes": 1024,
                },
            }}}},
        }
        validation = BENCH.validate_manifest(args, manifest)
        self.assertEqual(validation["status"], "pass")
        self.assertTrue(all(item["status"] == "pass" for item in validation["checks"]))

    def test_manifest_validation_detects_mismatch(self):
        args = SimpleNamespace(kv_mode="q8_0", split_boundary="36/28", cuda_graph="on", pipeline="off")
        manifest = {
            "source_commit": "abc",
            "model_sha256": "def",
            "server_command": "server",
            "host": {
                "gpu_topology": {"ok": True},
                "gpu_p2p_read": {"ok": True},
                "gpu_p2p_write": {"ok": True},
            },
            "server": {"profile": {"ok": True, "value": {
                "kv_cache_type_k": "q8_0",
                "kv_cache_type_v": "q8_0",
                "layer_boundary": {"gpu0": 36, "gpu1": 28},
            }}},
            "after": {"server": {"graphs": {"ok": True, "value": {
                "hits": 1,
                "captures": 0,
                "pipeline_transfer": {"mode": "pinned_host_staged", "depth": 2},
            }}}},
        }
        validation = BENCH.validate_manifest(args, manifest)
        self.assertEqual(validation["status"], "fail")
        self.assertIn("pipeline_depth", [item["name"] for item in validation["checks"] if item["status"] == "fail"])


if __name__ == "__main__":
    unittest.main()
