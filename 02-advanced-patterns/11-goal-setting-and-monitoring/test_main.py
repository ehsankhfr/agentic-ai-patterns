"""Deterministic unit tests; no live model required."""
import unittest
from unittest.mock import Mock, patch

import main as m


class PatternTests(unittest.TestCase):
    def test_progress_is_clamped_and_handles_zero_target(self):
        for target, current, expected in [(10, 15, 1), (10, -2, 0), (0, 5, 0), (10, 5, 0.5)]:
            with self.subTest(target=target, current=current):
                self.assertEqual(m.Milestone("step", target, current).progress, expected)

    def test_goal_tree_aggregation_and_paths(self):
        direct = m.Milestone("a", 10, 10)
        child = m.Milestone("b", 10, 0)
        goal = m.Goal("root", [direct], [m.Goal("child", [child])])
        self.assertEqual(goal.overall_progress(), 0.5)
        self.assertEqual(goal.iter_milestones(), [("root", direct), ("root / child", child)])
        self.assertEqual(m.Goal("empty").overall_progress(), 0)

    def test_on_track_does_not_call_model(self):
        goal = m.Goal("root", [m.Milestone("a", 10, 10)])
        with patch.object(m, "traced_llm_call") as call:
            report = m.GoalMonitor().review(goal)
        self.assertTrue(report.on_track)
        call.assert_not_called()

    def test_stall_detector_and_reset(self):
        step = m.Milestone("blocked", 10, 4)
        never_started = m.Milestone("waiting", 10, 0)
        goal = m.Goal("root", [step, never_started])
        detector = m.StallDetector(max_idle_reviews=2)
        self.assertEqual(detector.observe(goal, 0.7), [])
        self.assertEqual(detector.observe(goal, 0.7), [])
        self.assertEqual(detector.observe(goal, 0.7), ["blocked"])
        detector.reset()
        self.assertEqual(detector.observe(goal, 0.7), [])
        step.current = 8
        self.assertEqual(detector.observe(goal, 0.7), [])

    def test_measurements_update_only_named_milestones(self):
        first = m.Milestone("a", 10, 1)
        second = m.Milestone("b", 10, 2)
        goal = m.Goal("root", [first, second])
        m.apply_measurements(goal, {"a": 7})
        self.assertEqual((first.current, second.current), (7, 2))

    def test_escalation_passes_brief_and_resume_flag_to_reviewer(self):
        decision = m.HumanDecision("take_over", "manual follow-up")
        reviewer = Mock(return_value=decision)
        with patch.object(m, "traced_llm_call", return_value="handoff brief") as call:
            self.assertIs(m.escalate("stalled", "known facts", reviewer, False), decision)
        reviewer.assert_called_once_with("handoff brief", False)
        self.assertIn("known facts", call.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
