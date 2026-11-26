"""Deterministic unit tests; no live model required."""
import unittest
from unittest.mock import Mock, patch

import main as m


class PatternTests(unittest.TestCase):
    def test_feedback_is_reused_in_adaptation(self):
        learner = m.FeedbackLearner()
        attempt = m.Attempt("task", "output", "failure", feedback="check facts")
        with patch.object(m, "llm_call", side_effect=["Verify sources", "improved plan"]) as call:
            self.assertEqual(learner.learn(attempt), "Verify sources")
            self.assertEqual(learner.adapt_plan("next task"), "improved plan")
        self.assertIn("Verify sources", call.call_args.args[0])

    def test_example_reuse_ignores_failed_attempts(self):
        failed = m.Attempt("write Python code", "", "failure", approach="bad")
        good = m.Attempt("write Python code", "", "success", approach="test first")
        learner = m.ExampleBasedLearner([failed, good])
        self.assertIs(learner.recommend("write Python code"), good)
        self.assertIsNone(learner.recommend("unrelated"))
        with patch.object(m, "llm_call", return_value="plan") as call:
            learner.adapt_plan("write Python code")
        self.assertIn("test first", call.call_args.args[0])

    def test_strategy_exploration_then_best_mean(self):
        selector = m.StrategySelector()
        self.assertEqual(selector.select(["a", "b"]), "a")
        selector.record("a", 0.5)
        self.assertEqual(selector.select(["a", "b"]), "b")
        selector.record("b", 0.9)
        selector.record("b", 0.7)
        self.assertEqual(selector.select(["a", "b"]), "b")

    def test_strategy_rejects_invalid_scores_and_empty_choices(self):
        selector = m.StrategySelector()
        with self.assertRaises(ValueError):
            selector.select([])
        for score in (-0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                selector.record("a", score)

    def test_similarity_handles_empty_and_identical_inputs(self):
        self.assertEqual(m._similarity("", "task"), 0)
        self.assertAlmostEqual(m._similarity("Write code", "write code"), 1)


if __name__ == "__main__":
    unittest.main()
