"""Test the review gate without Ollama or external side effects."""
import unittest

from main import Decision, Ticket, run_agent


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.ticket = Ticket("T-1", "Known facts", True)
        self.sent = []
        self.drafts = []
        self.reviews = []

    def draft(self, ticket, previous, feedback):
        self.drafts.append((previous, feedback))
        return "Revised draft" if feedback else "Initial draft"

    def send(self, ticket, draft):
        self.sent.append(draft)

    def run_with(self, *decisions, ticket=None, max_revisions=2):
        choices = iter(decisions)

        def reviewer(ticket, draft):
            self.reviews.append(draft)
            return next(choices, Decision("unavailable"))

        return run_agent(ticket or self.ticket, self.draft, reviewer, self.send,
                         max_revisions=max_revisions)

    def test_approve_sends_exact_reviewed_draft(self):
        result = self.run_with(Decision("approve"))
        self.assertEqual(result.status, "sent")
        self.assertEqual(self.sent, self.reviews)
        self.assertEqual(self.sent, ["Initial draft"])

    def test_reject_blocks_send(self):
        self.assertEqual(self.run_with(Decision("reject")).status, "rejected")
        self.assertEqual(self.sent, [])

    def test_revision_requires_fresh_approval(self):
        result = self.run_with(Decision("revise", "Clarify"), Decision("approve"))
        self.assertEqual(result.status, "sent")
        self.assertEqual(self.reviews, ["Initial draft", "Revised draft"])
        self.assertEqual(self.drafts[1], ("Initial draft", "Clarify"))
        self.assertEqual(self.sent, ["Revised draft"])

    def test_revision_then_rejection_blocks_send(self):
        result = self.run_with(Decision("revise", "Clarify"), Decision("reject"))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(self.sent, [])

    def test_revision_budget_blocks_send(self):
        result = self.run_with(Decision("revise", "Clarify"), max_revisions=0)
        self.assertEqual(result.status, "escalated")
        self.assertEqual(self.sent, [])

    def test_missing_feedback_blocks_send(self):
        self.assertEqual(self.run_with(Decision("revise")).status, "escalated")
        self.assertEqual(self.sent, [])

    def test_takeover_blocks_send(self):
        self.assertEqual(self.run_with(Decision("takeover")).status, "human_takeover")
        self.assertEqual(self.sent, [])

    def test_missing_or_invalid_decision_blocks_send(self):
        for decision in (Decision("unavailable"), Decision("maybe"), None):
            with self.subTest(decision=decision):
                self.assertEqual(self.run_with(decision).status, "escalated")
                self.assertEqual(self.sent, [])

    def test_low_risk_policy_bypasses_review(self):
        result = self.run_with(ticket=Ticket("T-2", "Known facts", False))
        self.assertEqual(result.status, "sent")
        self.assertEqual(self.reviews, [])
        self.assertEqual(self.sent, ["Initial draft"])

    def test_reviewer_error_blocks_send(self):
        def broken(*args):
            raise TimeoutError()

        result = run_agent(self.ticket, self.draft, broken, self.send)
        self.assertEqual(result.status, "escalated")
        self.assertEqual(self.sent, [])

    def test_model_error_blocks_send(self):
        def broken(*args):
            raise RuntimeError()

        result = run_agent(self.ticket, broken, lambda *args: Decision("approve"), self.send)
        self.assertEqual(result.status, "escalated")
        self.assertEqual(self.sent, [])

    def test_empty_draft_blocks_send(self):
        result = run_agent(self.ticket, lambda *args: " ", lambda *args: Decision("approve"), self.send)
        self.assertEqual(result.status, "escalated")
        self.assertEqual(self.sent, [])

    def test_delivery_error_is_not_retried(self):
        attempts = []

        def broken(ticket, draft):
            attempts.append(draft)
            raise TimeoutError()

        result = run_agent(self.ticket, self.draft, lambda *args: Decision("approve"), broken)
        self.assertEqual(result.status, "delivery_unknown")
        self.assertEqual(attempts, ["Initial draft"])


if __name__ == "__main__":
    unittest.main()
