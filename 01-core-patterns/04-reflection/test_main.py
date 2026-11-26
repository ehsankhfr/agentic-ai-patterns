"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.TestCase):
    def test_self_reflection_passes_critique_to_revision(self):
        with patch.object(m, "llm_call", side_effect=["draft", "fix this", "revised"]) as call:
            self.assertEqual(m.self_reflect_and_revise("task"), "revised")
            self.assertIn("draft", call.call_args_list[1].args[0])
            self.assertIn("fix this", call.call_args.args[0])

    def test_zero_iterations_returns_initial_draft(self):
        with patch.object(m, "llm_call", return_value="draft") as call:
            self.assertEqual(m.self_reflect_and_revise("task", iterations=0), "draft")
            self.assertEqual(call.call_count, 1)

    def test_critic_approval_stops_loop(self):
        with patch.object(m, "llm_call", side_effect=["draft", "APPROVED"]) as call:
            self.assertEqual(m.two_agent_reflection_loop("task"), "draft")
            self.assertEqual(call.call_count, 2)

    def test_iteration_limit_returns_last_revision(self):
        with patch.object(m, "llm_call", side_effect=["draft", "fix", "revised"]) as call:
            self.assertEqual(m.two_agent_reflection_loop("task", max_iterations=1), "revised")
            self.assertEqual(call.call_count, 3)


if __name__ == "__main__":
    unittest.main()
