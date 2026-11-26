"""Offline regression tests for proactive and reflective recovery."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main as pattern
from variations import Checkpoints, Stage, monitored_workflow, read_order, reflective_recovery


VALID = "SELECT order_id, carrier, eta FROM orders WHERE order_id='4821'"
INVALID = "SELECT order_id, carrier, delivery_date FROM orders WHERE order_id='4821'"


class ReflectionTests(unittest.TestCase):
    def test_correct_query_needs_no_reflection(self):
        ask = Mock()
        result = reflective_recovery(ask, VALID)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.value, {"order_id": "4821", "carrier": "UPS", "eta": "Thursday"})
        ask.assert_not_called()

    def test_error_is_reflected_on_and_revised_query_validated(self):
        ask = Mock(side_effect=["The column is eta, not delivery_date.", VALID])
        result = reflective_recovery(ask, INVALID)
        self.assertEqual((result.status, result.attempts), ("recovered", 2))
        self.assertIn("delivery_date", ask.call_args_list[0].args[0])
        self.assertIn("The column is eta", ask.call_args_list[1].args[0])
        self.assertIn("recover:validated_result", result.events)

    def test_budget_bounds_unsuccessful_recovery(self):
        ask = Mock(side_effect=["Fix the field", "SELECT unknown FROM orders"])
        result = reflective_recovery(ask, INVALID, max_attempts=2)
        self.assertEqual((result.status, result.attempts), ("controlled_failure", 2))
        self.assertIsNone(result.value)
        self.assertEqual(ask.call_count, 2)

    def test_model_outage_stops_recovery(self):
        result = reflective_recovery(Mock(side_effect=RuntimeError("offline")), INVALID)
        self.assertEqual(result.status, "controlled_failure")
        self.assertIn("model_error:RuntimeError", result.events)

    def test_unchanged_query_is_not_retried(self):
        result = reflective_recovery(Mock(side_effect=["reflection", INVALID]), INVALID)
        self.assertEqual((result.status, result.attempts), ("controlled_failure", 1))

    def test_writes_and_wrong_order_are_rejected(self):
        import sqlite3
        with self.assertRaises(sqlite3.Error):
            read_order("DELETE FROM orders")
        with self.assertRaises(ValueError):
            read_order("SELECT order_id, carrier, eta FROM orders WHERE order_id='7310'")
        with self.assertRaises(ValueError):
            read_order("SELECT order_id, carrier, eta FROM orders")

    def test_initial_model_failure_is_controlled(self):
        result = reflective_recovery(Mock(side_effect=RuntimeError("offline")))
        self.assertEqual(result.status, "controlled_failure")
        self.assertEqual(result.attempts, 0)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "checkpoint.sqlite3"

    def test_preflight_blocks_work_before_any_stage_runs(self):
        first = Mock()
        second = Mock()
        result = monitored_workflow(Checkpoints(self.path), "run", [
            Stage("one", lambda: True, first),
            Stage("two", lambda: False, second),
        ])
        self.assertEqual(result.status, "paused")
        self.assertIn("preflight_blocked:two", result.events)
        first.assert_not_called()
        second.assert_not_called()

    def test_resume_after_execution_failure_does_not_repeat_completed_stage(self):
        lookup = Mock(return_value={"carrier": "UPS"})
        draft = Mock(side_effect=[RuntimeError("offline"), "UPS update"])
        stages = [Stage("lookup", lambda: True, lookup), Stage("draft", lambda: True, draft)]
        first = monitored_workflow(Checkpoints(self.path), "run", stages)
        self.assertEqual(first.status, "paused")
        self.assertEqual(first.value, {"lookup": {"carrier": "UPS"}})
        second = monitored_workflow(Checkpoints(self.path), "run", stages)
        self.assertEqual(second.status, "completed")
        self.assertEqual(second.value["draft"], "UPS update")
        lookup.assert_called_once()
        self.assertEqual(second.events.count("checkpoint:lookup"), 1)
        self.assertIn("controlled_failure:draft:RuntimeError", second.events)

    def test_runtime_health_is_rechecked_after_preflight(self):
        health = Mock(side_effect=[True, False])
        run = Mock()
        result = monitored_workflow(Checkpoints(self.path), "run", [Stage("one", health, run)])
        self.assertEqual(result.status, "paused")
        run.assert_not_called()

    def test_checkpoint_plan_mismatch_is_rejected(self):
        store = Checkpoints(self.path)
        monitored_workflow(store, "run", [Stage("one", lambda: True, lambda _: "ok")])
        with self.assertRaises(ValueError):
            monitored_workflow(store, "run", [Stage("different", lambda: True, lambda _: "ok")])

    def test_completed_run_is_not_reexecuted(self):
        run = Mock(return_value="done")
        stages = [Stage("one", lambda: True, run)]
        monitored_workflow(Checkpoints(self.path), "run", stages)
        result = monitored_workflow(Checkpoints(self.path), "run", stages)
        self.assertEqual(result.status, "completed")
        run.assert_called_once()


class ReactiveRegressionTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(pattern, "INCIDENTS", pattern.IncidentLog())
        p.start()
        self.addCleanup(p.stop)

    def test_unexpected_forward_exception_still_rolls_back(self):
        undo = Mock()
        steps = [
            pattern.PlanStep("first", Mock(), undo),
            pattern.PlanStep("second", Mock(side_effect=RuntimeError("unexpected")), Mock()),
        ]
        result = pattern.run_with_rollback(steps)
        self.assertEqual(result.failed_step, "second")
        self.assertEqual(result.rolled_back, ["first"])
        undo.assert_called_once()

    def test_model_failure_in_agent_loop_is_controlled(self):
        with patch.object(pattern, "chat", side_effect=pattern.TransientError("offline")):
            result = pattern.run_agent("task", [], "system")
        self.assertEqual(result.stopped_by, "model_error")
        self.assertEqual(result.steps, 0)

    def test_final_status_model_failure_is_controlled(self):
        with patch.object(pattern, "chat", side_effect=RuntimeError("offline")):
            result = pattern.run_agent("task", [], "system", max_steps=1, first_turn="invalid")
        self.assertEqual(result.stopped_by, "step_budget")
        self.assertIn("manual review", result.answer)

    def test_handoff_survives_model_outage(self):
        outcome = pattern.PlanOutcome(failed_step="email", error="offline")
        with patch.object(pattern, "llm_call", side_effect=RuntimeError("offline")):
            note = pattern.write_handoff_note("refund", outcome, pattern.Environment())
        self.assertIn("Nothing is changed", note)
        self.assertIn("email", note)

    def test_diagnostics_survive_model_outage(self):
        pattern.INCIDENTS.detected("order API", "503 unavailable")
        output = io.StringIO()
        with patch.object(pattern, "llm_call", side_effect=RuntimeError("offline")), contextlib.redirect_stdout(output):
            pattern.diagnose()
        self.assertIn("root cause unconfirmed", output.getvalue())
        self.assertIn("503 unavailable", output.getvalue())

    def test_incident_log_persists_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incidents.jsonl"
            pattern.IncidentLog(ticket="T-1", log_path=str(path)).detected("api", "timeout")
            pattern.IncidentLog(ticket="T-2", log_path=str(path)).notify("ops", "manual review")
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["ticket"] for r in records], ["T-1", "T-2"])
        self.assertEqual([r["stage"] for r in records], ["detect", "notify"])


if __name__ == "__main__":
    unittest.main()
