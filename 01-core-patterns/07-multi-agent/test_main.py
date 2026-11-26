"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.TestCase):
    def test_workspace_contains_only_contributed_artifacts(self):
        workspace = m.SharedWorkspace("task", [m.Agent("A", "research", "fact"), m.Agent("B", "critic")])
        self.assertIn("fact", workspace.context())
        self.assertNotIn("B - critic", workspace.context())
        workspace.review = "review"
        self.assertIn("review", workspace.context())

    def test_sequential_pipeline_passes_previous_handoff(self):
        with patch.object(m, "ask_role", side_effect=["research", "critique", "draft"]) as role, patch.object(m, "synthesize", return_value="final") as synth:
            self.assertEqual(m.sequential_collaboration("task"), "final")
            self.assertEqual(role.call_args_list[1].args[2], "research")
            self.assertEqual(role.call_args_list[2].args[2], "critique")
            self.assertIn("draft", synth.call_args.args[1])

    def test_supervisor_collects_worker_evidence(self):
        with patch.object(m, "ask_role", side_effect=["A", "B", "C"]), patch.object(m, "synthesize", return_value="final") as synth:
            self.assertEqual(m.supervisor_worker_collaboration("task"), "final")
            self.assertIn("Worker assignment", synth.call_args.args[1])
            self.assertIn("C", synth.call_args.args[1])

    def test_debate_rounds_and_negative_guard(self):
        with self.assertRaises(ValueError):
            m.debate_collaboration("task", rounds=-1)
        with patch.object(m, "ask_role", side_effect=["a", "b", "c", "d"]) as role, patch.object(m, "synthesize", return_value="decision") as synth:
            self.assertEqual(m.debate_collaboration("task", rounds=1), "decision")
            self.assertEqual(role.call_count, 4)
            self.assertIn("Negative:\nd", synth.call_args.args[1])

    def test_blackboard_agent_adds_contribution(self):
        workspace = m.SharedWorkspace("task")
        agent = m.Agent("A", "research")
        with patch.object(m, "llm_call", return_value="fact"):
            self.assertEqual(m.run_agent(agent, workspace), "fact")
        self.assertEqual(workspace.contributions, [agent])
        self.assertIn("fact", workspace.context())


if __name__ == "__main__":
    unittest.main()
