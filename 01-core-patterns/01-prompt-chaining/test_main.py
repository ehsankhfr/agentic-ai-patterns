"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.TestCase):
    def test_article_chain_passes_outputs_forward(self):
        with patch.object(m, "llm_call", side_effect=["outline", "draft", "edited"]) as call:
            self.assertEqual(m.run_article_chain("topic"), "edited")
            self.assertIn("outline", call.call_args_list[1].args[0])
            self.assertIn("draft", call.call_args_list[2].args[0])

    def test_translation_retries_failed_gate(self):
        with patch.object(m, "llm_call", side_effect=["bad", "FAIL", "good", "PASS", "polished"]) as call:
            self.assertEqual(m.run_translation_chain("hello", "French"), "polished")
            self.assertIn("good", call.call_args.args[0])
            self.assertEqual(call.call_count, 5)

    def test_translation_budget_uses_last_draft(self):
        with patch.object(m, "llm_call", side_effect=["a", "FAIL", "b", "FAIL", "final"]) as call:
            self.assertEqual(m.run_translation_chain("hello", "French", max_retries=1), "final")
            self.assertIn("b", call.call_args.args[0])

    def test_gate_normalizes_result(self):
        with patch.object(m, "llm_call", side_effect=[" pass ", "FAIL"]):
            self.assertTrue(m.gate_verify_translation("a", "b", "French"))
            self.assertFalse(m.gate_verify_translation("a", "b", "French"))


if __name__ == "__main__":
    unittest.main()
