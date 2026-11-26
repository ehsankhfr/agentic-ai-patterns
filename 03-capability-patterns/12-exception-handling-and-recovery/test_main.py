"""Deterministic tests for exception detection, handling and recovery.

All model calls are mocked or scripted; no Ollama or live service is needed.
"""
import itertools
import json
import unittest
from unittest.mock import Mock, patch

import main as pattern


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        patches = [
            patch.object(pattern, "INCIDENTS", pattern.IncidentLog()),
            patch.object(pattern.time, "sleep"),
            patch.object(pattern, "llm_call", side_effect=AssertionError("Unexpected LLM call")),
            patch.object(pattern, "llm_chat", side_effect=AssertionError("Unexpected LLM call")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_http_error_taxonomy(self):
        cases = {
            400: pattern.InvalidRequestError, 422: pattern.InvalidRequestError,
            404: pattern.NotFoundError, 408: pattern.TransientError,
            429: pattern.TransientError, 500: pattern.TransientError,
            503: pattern.TransientError, 401: pattern.PermanentError,
            403: pattern.PermanentError,
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                self.assertIs(type(pattern.classify_error(pattern.HttpError(status, "test"))), expected)

    def test_unknown_error_is_permanent(self):
        original = RuntimeError("unexpected")
        error = pattern.classify_error(original)
        self.assertIs(type(error), pattern.PermanentError)
        self.assertIs(error.__cause__, original)

    def test_existing_agent_error_preserved(self):
        error = pattern.InvalidRequestError("change arguments")
        self.assertIs(pattern.classify_error(error), error)

    def test_deadline_returns_completed_value(self):
        self.assertEqual(pattern.call_with_deadline(lambda: "ok", 2), "ok")

    def test_deadline_exceeded_is_transient(self):
        with patch.object(pattern, "ThreadPoolExecutor") as executor:
            pool = executor.return_value
            pool.submit.return_value.result.side_effect = pattern.DeadlineExceeded()
            with self.assertRaises(pattern.TransientError):
                pattern.call_with_deadline(lambda: None, 0.2)
            pool.submit.return_value.result.assert_called_once_with(timeout=0.2)
            pool.shutdown.assert_called_once_with(wait=False)

    def test_validate_order_rejects_missing_fields_and_wrong_type(self):
        for payload in ([], None, {"order_id": "4821", "status": "shipped"}):
            with self.subTest(payload=payload), self.assertRaises(pattern.TransientError):
                pattern.validate_order(payload)
        self.assertEqual(pattern.validate_order(pattern.ORDERS["4821"]), pattern.ORDERS["4821"])

    def test_grounding_accepts_alternative_phrasing(self):
        pattern.check_grounding("DHL delivery on Monday; may be OUT OF DATE.",
                                {"carrier": ("DHL",), "warning": ("cached", "out of date")})

    def test_grounding_rejects_missing_fact(self):
        with self.assertRaisesRegex(pattern.InvalidRequestError, "delivery"):
            pattern.check_grounding("Shipped by UPS", {"delivery": ("Thursday",)})

    def test_retry_transient_then_success_with_exponential_backoff(self):
        operation = Mock(side_effect=[pattern.TransientError("one"),
                                      pattern.TransientError("two"), "ok"])
        with patch.object(pattern.random, "uniform", return_value=1):
            self.assertEqual(pattern.retry_with_backoff(operation, "service", base_delay=0.1), "ok")
        self.assertEqual([c.args for c in operation.call_args_list], [(1,), (2,), (3,)])
        self.assertEqual([c.args for c in pattern.time.sleep.call_args_list], [(0.1,), (0.2,)])

    def test_retry_budget_exhausted(self):
        operation = Mock(side_effect=pattern.TransientError("down"))
        with self.assertRaises(pattern.TransientError):
            pattern.retry_with_backoff(operation, "service", max_attempts=2)
        self.assertEqual(operation.call_count, 2)
        self.assertEqual(pattern.time.sleep.call_count, 1)

    def test_permanent_and_invalid_requests_not_retried(self):
        for error in (pattern.PermanentError("denied"), pattern.NotFoundError("missing"),
                      pattern.InvalidRequestError("bad arguments")):
            with self.subTest(error=error):
                operation = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    pattern.retry_with_backoff(operation, "service")
                operation.assert_called_once_with(1)
        pattern.time.sleep.assert_not_called()

    def test_model_fallback_and_total_outage(self):
        first = pattern.ModelEndpoint("primary", "test", itertools.repeat(pattern.TransientError("down")))
        backup = pattern.ModelEndpoint("backup", "test", iter(["backup response"]))
        self.assertEqual(pattern.ask_models([first, backup], "question"), "backup response")
        self.assertIsNone(pattern.ask_models([first], "question"))

    def test_primary_recovers_from_malformed_payload(self):
        state = {"order_id": "4821"}
        api = pattern.OrderAPI(iter(["malformed", "ok"]))
        pattern.primary_handler(state, api)
        self.assertEqual(state["source"], "live order API")
        self.assertEqual(state["data"]["carrier"], "UPS")

    def test_cache_fallback_marks_data_stale_and_notifies(self):
        state = {"order_id": "7310", "primary_error": pattern.TransientError("down")}
        pattern.fallback_handler(state)
        self.assertTrue(state["stale"])
        self.assertEqual(state["data"], pattern.CACHE["7310"])
        self.assertTrue(any("ops on-call" in entry for entry in pattern.INCIDENTS.entries))
        self.assertIn("out of date", pattern.template_reply(state))

    def test_cache_miss_and_not_found(self):
        for error, expected in [(pattern.TransientError("down"), "no cached copy"),
                                (pattern.NotFoundError("missing"), "does not exist")]:
            with self.subTest(error=error):
                state = {"order_id": "0000", "data": None, "primary_error": error}
                pattern.fallback_handler(state)
                self.assertIsNone(state["data"])
                self.assertIn(expected, state["problem"])
                self.assertIn(expected, pattern.template_reply(state))

    def test_response_agent_self_corrects(self):
        state = {"question": "Where is my order?", "data": pattern.ORDERS["4821"], "stale": False}
        with patch.object(pattern, "ask_models", side_effect=["Invoice is ready.", "UPS delivery Thursday."]) as ask:
            self.assertEqual(pattern.response_agent(state, []), "UPS delivery Thursday.")
            self.assertIn("reviewer rejected", ask.call_args.args[1])
            self.assertEqual(ask.call_count, 2)

    def test_response_agent_uses_template_after_bad_drafts_or_outage(self):
        state = {"question": "Where is my order?", "data": pattern.CACHE["7310"], "stale": True}
        for replies in (["Wrong answer", "Still wrong"], [None]):
            with self.subTest(replies=replies), patch.object(pattern, "ask_models", side_effect=replies):
                self.assertEqual(pattern.response_agent(state, []), pattern.template_reply(state))

    def test_action_parsing_accepts_final_and_tool_call(self):
        final = {"action": "final", "answer": "done"}
        tool = {"action": "call_tool", "tool": "lookup", "arguments": {"id": "1"}}
        self.assertEqual(pattern.parse_action(json.dumps(final)), final)
        self.assertEqual(pattern.parse_action(json.dumps(tool)), tool)
        self.assertEqual(pattern.parse_action('prefix {"action":"final","answer":"done",} suffix'), final)

    def test_action_parsing_rejects_invalid_shapes(self):
        for raw in ("not JSON", '{}', '{"action":"final","answer":3}',
                    '{"action":"call_tool","tool":"lookup","arguments":[]}'):
            with self.subTest(raw=raw), self.assertRaises(pattern.InvalidRequestError):
                pattern.parse_action(raw)

    def test_unknown_tool_does_not_run(self):
        run = Mock()
        result = pattern.execute_tool({"lookup": pattern.Tool("lookup", "lookup()", run)},
                                      {"tool": "missing"}, {}, 3)
        self.assertIn("unknown tool", result)
        run.assert_not_called()

    def test_permanent_error_opens_circuit_immediately(self):
        run = Mock(side_effect=pattern.PermanentError("denied"))
        registry = {"lookup": pattern.Tool("lookup", "lookup()", run)}
        failures = {}
        for _ in range(2):
            self.assertIn("circuit_open", pattern.execute_tool(registry, {"tool": "lookup"}, failures, 3))
        run.assert_called_once()
        self.assertEqual(failures["lookup"], 3)

    def test_repeated_invalid_requests_open_circuit(self):
        run = Mock(side_effect=pattern.InvalidRequestError("bad input"))
        registry = {"lookup": pattern.Tool("lookup", "lookup()", run)}
        failures = {}
        for _ in range(2):
            self.assertIn("invalid_request", pattern.execute_tool(registry, {"tool": "lookup"}, failures, 3))
        self.assertIn("circuit_open", pattern.execute_tool(registry, {"tool": "lookup"}, failures, 3))
        self.assertIn("circuit_open", pattern.execute_tool(registry, {"tool": "lookup"}, failures, 3))
        self.assertEqual(run.call_count, 3)

    def test_success_resets_consecutive_failure_count(self):
        run = Mock(side_effect=[pattern.InvalidRequestError("bad input"), "ok"])
        registry = {"lookup": pattern.Tool("lookup", "lookup()", run)}
        failures = {}
        pattern.execute_tool(registry, {"tool": "lookup"}, failures, 3)
        self.assertEqual(failures["lookup"], 1)
        self.assertIn("RESULT: ok", pattern.execute_tool(registry, {"tool": "lookup"}, failures, 3))
        self.assertEqual(failures["lookup"], 0)

    def test_agent_corrects_bad_action_and_finishes(self):
        with patch.object(pattern, "chat", return_value='{"action":"final","answer":"done"}') as chat:
            result = pattern.run_agent("task", [], "system", first_turn="invalid")
            self.assertEqual((result.answer, result.steps, result.stopped_by), ("done", 2, "answer"))
            self.assertTrue(any("invalid_request" in m["content"] for m in chat.call_args.args[0]))

    def test_step_budget_stops_tool_execution(self):
        run = Mock(return_value="ok")
        tool = pattern.Tool("lookup", "lookup()", run)
        request = '{"action":"call_tool","tool":"lookup","arguments":{}}'
        with patch.object(pattern, "chat", return_value=request):
            result = pattern.run_agent("task", [tool], "system", first_turn=request, max_steps=1)
        self.assertEqual(result.stopped_by, "step_budget")
        run.assert_called_once()

    def test_successful_plan_does_not_undo(self):
        action, undo = Mock(), Mock()
        result = pattern.run_with_rollback([pattern.PlanStep("one", action, undo)])
        self.assertTrue(result.succeeded)
        self.assertEqual(result.completed, ["one"])
        action.assert_called_once()
        undo.assert_not_called()

    def test_rollback_reverses_completed_steps_only(self):
        events = []
        def fail():
            raise pattern.PermanentError("failed")
        steps = [
            pattern.PlanStep("one", lambda: events.append("do1"), lambda: events.append("undo1")),
            pattern.PlanStep("two", lambda: events.append("do2"), lambda: events.append("undo2")),
            pattern.PlanStep("three", fail, lambda: events.append("undo3")),
        ]
        outcome = pattern.run_with_rollback(steps)
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.failed_step, "three")
        self.assertEqual(outcome.rolled_back, ["two", "one"])
        self.assertEqual(events, ["do1", "do2", "undo2", "undo1"])

    def test_rollback_continues_after_undo_failure(self):
        first_undo = Mock()
        steps = [
            pattern.PlanStep("one", Mock(), first_undo),
            pattern.PlanStep("two", Mock(), Mock(side_effect=RuntimeError("cannot undo"))),
            pattern.PlanStep("three", Mock(side_effect=pattern.PermanentError("down")), Mock()),
        ]
        result = pattern.run_with_rollback(steps)
        self.assertEqual(result.rolled_back, ["one"])
        self.assertEqual(result.undo_failures, ["two: cannot undo"])
        first_undo.assert_called_once()

    def test_refund_invalid_input_does_not_change_resources(self):
        desk = pattern.RefundDesk(email_up=True)
        for amount in ("39,90 EUR", -1, 100, float("nan")):
            with self.subTest(amount=amount), self.assertRaises(pattern.InvalidRequestError):
                desk.issue_refund({"order_id": "4821", "amount": amount, "reason": "damaged"})
            self.assertEqual(desk.env.resources, [])

    def test_refund_rolls_back_when_confirmation_fails(self):
        desk = pattern.RefundDesk(email_up=False)
        with self.assertRaises(pattern.PermanentError):
            desk.issue_refund({"order_id": "4821", "amount": 39.90, "reason": "damaged"})
        self.assertEqual(desk.env.resources, [])
        self.assertEqual(desk.outcome.rolled_back, ["credit_payment", "reserve_refund"])

    def test_successful_refund_retains_resources(self):
        desk = pattern.RefundDesk(email_up=True)
        self.assertIn("39.90 EUR", desk.issue_refund({"order_id": "4821", "amount": 39.90, "reason": "damaged"}))
        self.assertTrue(desk.outcome.succeeded)
        self.assertEqual(desk.env.resources, ["reservation:4821", "credit:39.90 EUR"])

    def test_handoff_prompt_states_current_system_state(self):
        outcome = pattern.PlanOutcome(failed_step="confirmation", error="down", rolled_back=["credit"])
        with patch.object(pattern, "llm_call", return_value="handoff") as llm:
            self.assertEqual(pattern.write_handoff_note("refund", outcome, pattern.Environment()), "handoff")
            self.assertIn("Nothing is changed", llm.call_args.args[0])
            self.assertIn("credit", llm.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
