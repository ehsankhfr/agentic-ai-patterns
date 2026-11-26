"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.TestCase):
    def test_route_normalization_and_fallback(self):
        with patch.object(m, "llm_call", side_effect=[" BILLING ", "unknown"]):
            self.assertEqual(m.route_support_query("invoice"), "billing")
            self.assertEqual(m.route_support_query("x"), "general")

    def test_support_dispatches_to_specialist(self):
        with patch.object(m, "llm_call", side_effect=["refund", "answer"]) as call:
            self.assertEqual(m.support_router("refund please"), {"category": "refund", "response": "answer"})
            self.assertIn("refund specialist", call.call_args.args[1])

    def test_structured_classification(self):
        with patch.object(m, "llm_call", side_effect=['{"content_type":"code","confidence":0.9}', "code answer"]) as call:
            result = m.structured_route("write code")
            self.assertEqual(result["content_type"], "code")
            self.assertEqual(result["classification"]["confidence"], 0.9)
            self.assertIn("software engineer", call.call_args.args[1])

    def test_structured_fallbacks(self):
        for raw in ("broken JSON", '{"content_type":"unknown"}'):
            with self.subTest(raw=raw), patch.object(m, "llm_call", side_effect=[raw, "answer"]):
                self.assertEqual(m.structured_route("x")["content_type"], "other")


if __name__ == "__main__":
    unittest.main()
