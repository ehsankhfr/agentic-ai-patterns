"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.TestCase):
    def test_extract_json_repairs_trailing_comma(self):
        self.assertEqual(m.extract_json('prefix [{"step":1,},] suffix'), [{"step": 1}])
        with self.assertRaises(ValueError):
            m.extract_json("no plan")

    def test_plan_tracks_completed_and_pending(self):
        first = m.Step(1, "first", "act", result="result", completed=True)
        second = m.Step(2, "second", "act")
        plan = m.Plan("goal", [first, second])
        self.assertEqual(plan.pending(), [second])
        self.assertIn("result", plan.completed_summary())
        self.assertNotIn("second", plan.completed_summary())

    def test_static_plan_uses_prior_results(self):
        raw = '[{"step":1,"description":"first","action":"a"},{"step":2,"description":"second","action":"b"}]'
        with patch.object(m, "llm_call", side_effect=[raw, "first result", "second result", "final"]) as call:
            self.assertEqual(m.static_plan_and_execute("goal"), "final")
            self.assertIn("first result", call.call_args_list[2].args[0])
            self.assertIn("second result", call.call_args.args[0])

    def test_replan_falls_back_to_pending(self):
        first = m.Step(1, "first", "act", completed=True)
        second = m.Step(2, "second", "act")
        with patch.object(m, "llm_call", return_value="invalid"):
            self.assertEqual(m.replan(m.Plan("goal", [first, second]), first), [second])

    def test_replan_can_replace_remaining_steps(self):
        first = m.Step(1, "first", "act", completed=True)
        second = m.Step(2, "old", "act")
        with patch.object(m, "llm_call", return_value='[{"step":2,"description":"new","action":"change"}]'):
            result = m.replan(m.Plan("goal", [first, second]), first)
        self.assertEqual(result[0].description, "new")


if __name__ == "__main__":
    unittest.main()
