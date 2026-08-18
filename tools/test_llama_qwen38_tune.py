#!/usr/bin/env python3

import importlib.util
from importlib.machinery import SourceFileLoader
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("llama-qwen38-tune")
LOADER = SourceFileLoader("llama_qwen38_tune", str(MODULE_PATH))
SPEC = importlib.util.spec_from_loader("llama_qwen38_tune", LOADER)
TUNER = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(TUNER)


class TunerTests(unittest.TestCase):
    def test_candidate_splits_include_mtp_layers(self):
        candidates = TUNER.candidate_grid(False)
        boundaries = {candidate["layer_boundary"] for candidate in candidates}
        self.assertEqual(boundaries, {28, 32, 36})
        for candidate in candidates:
            left, right = [int(value) for value in candidate["split"].split("/")]
            self.assertEqual(left + right, TUNER.TOTAL_PLACEMENT_LAYERS)
        self.assertEqual(len(TUNER.candidate_grid(True)), 8)
        self.assertEqual(len(candidates), 54)

    def test_select_uses_worst_prompt_score(self):
        results = [{
            "layer_boundary": boundary,
            "max_num_batched_tokens": 2048,
            "ubatch_size": 64,
            "prompt_tokens": prompt,
            "success": True,
            "score": score,
        } for boundary, values in ((32, [(4096, 20.0), (32768, 19.0)]), (36, [(4096, 30.0), (32768, 18.0)]))
          for prompt, score in values]
        selected = TUNER.select_candidate(results, [4096, 32768])
        self.assertEqual(selected["layer_boundary"], 32)
        self.assertEqual(selected["robust_microbenchmark_score"], 19.0)

    def test_failed_prompt_disqualifies_candidate(self):
        results = [{
            "layer_boundary": 36,
            "max_num_batched_tokens": 2048,
            "ubatch_size": 64,
            "prompt_tokens": 4096,
            "success": True,
            "score": 10.0,
        }, {
            "layer_boundary": 36,
            "max_num_batched_tokens": 2048,
            "ubatch_size": 64,
            "prompt_tokens": 32768,
            "success": False,
        }]
        with self.assertRaises(RuntimeError):
            TUNER.select_candidate(results, [4096, 32768])

    def test_canonical_key_hash_is_order_independent(self):
        first = {"b": 2, "a": 1}
        second = {"a": 1, "b": 2}
        self.assertEqual(TUNER.canonical_sha256(first), TUNER.canonical_sha256(second))


if __name__ == "__main__":
    unittest.main()
