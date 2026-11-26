"""Deterministic tests: no model server or real delivery required."""

import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from variations import ApprovalStore, clarify_and_draft


class ClarificationTests(unittest.TestCase):
    def test_missing_field_asks_human_and_passes_answer_to_model(self):
        calls, questions = [], []

        def model(prompt, system):
            calls.append(prompt)
            return "What is the tracking number?" if len(calls) == 1 else "Draft"

        def human(field, question):
            questions.append(field)
            return "TRACK-123"

        status, draft, facts = clarify_and_draft({"carrier": "DHL"}, model, human)
        self.assertEqual((status, draft), ("drafted", "Draft"))
        self.assertEqual(questions, ["tracking_number"])
        self.assertEqual(facts["tracking_number"], "TRACK-123")
        self.assertIn("TRACK-123", calls[-1])

    def test_complete_facts_skip_human(self):
        def unexpected(*args):
            self.fail("No clarification necessary")

        result = clarify_and_draft({"carrier": "DHL", "tracking_number": "123"},
                                   lambda *args: "Draft", unexpected)
        self.assertEqual(result[0], "drafted")

    def test_blank_human_answer_escalates(self):
        result = clarify_and_draft({}, lambda *args: "Question", lambda *args: " ")
        self.assertEqual(result[:2], ("escalated", ""))

    def test_missing_reviewer_escalates(self):
        def unavailable(*args):
            raise EOFError()

        result = clarify_and_draft({}, lambda *args: "Question", unavailable)
        self.assertEqual(result[0], "escalated")

    def test_model_error_escalates(self):
        def broken(*args):
            raise RuntimeError()

        result = clarify_and_draft({}, broken, lambda *args: "Answer")
        self.assertEqual(result[0], "escalated")

    def test_empty_final_draft_escalates(self):
        result = clarify_and_draft({"carrier": "DHL", "tracking_number": "123"},
                                   lambda *args: " ", lambda *args: "Answer")
        self.assertEqual(result[0], "escalated")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "approvals.sqlite3"
        self.store = ApprovalStore(self.path)
        self.request_id = self.store.propose("T-1", "Exact draft")

    def test_pending_survives_restart_and_cannot_execute(self):
        store = ApprovalStore(self.path)
        self.assertEqual(store.get(self.request_id), ("T-1", "Exact draft", "pending"))
        self.assertFalse(store.execute(self.request_id))
        self.assertIsNone(store.delivered(self.request_id))

    def test_approve_then_resume_exactly_once(self):
        self.assertTrue(ApprovalStore(self.path).decide(self.request_id, "approve"))
        store = ApprovalStore(self.path)
        self.assertTrue(store.execute(self.request_id))
        self.assertEqual(store.delivered(self.request_id), ("T-1", "Exact draft"))
        self.assertEqual(store.get(self.request_id)[2], "executed")
        self.assertFalse(store.execute(self.request_id))

    def test_rejected_request_cannot_be_approved_or_executed(self):
        self.assertTrue(self.store.decide(self.request_id, "reject"))
        self.assertFalse(self.store.decide(self.request_id, "approve"))
        self.assertFalse(self.store.execute(self.request_id))
        self.assertIsNone(self.store.delivered(self.request_id))

    def test_invalid_decision_and_unknown_id(self):
        self.assertFalse(self.store.decide(self.request_id, "maybe"))
        self.assertFalse(self.store.execute("unknown"))
        self.assertEqual(self.store.get(self.request_id)[2], "pending")

    def test_empty_proposal_rejected(self):
        with self.assertRaises(ValueError):
            self.store.propose("T-1", " ")

    def test_concurrent_resume_only_one_succeeds(self):
        self.store.decide(self.request_id, "approve")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.store.execute(self.request_id), range(2)))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(self.store.delivered(self.request_id), ("T-1", "Exact draft"))


if __name__ == "__main__":
    unittest.main()
