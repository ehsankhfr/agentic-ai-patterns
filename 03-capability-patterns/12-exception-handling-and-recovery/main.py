"""
Exception Handling and Recovery Pattern

An agent that works in the real world cannot assume everything will go right.
Tools time out, services return errors or garbage, the model API throttles, the
model itself produces a bad tool call or an answer that ignores its data, and a
multi-step action can fail halfway through. This pattern gives an agent a way to
notice each of these, respond in proportion, and return to a stable state
instead of crashing or, worse, inventing an answer to hide the failure. It has
three stages that feed one another:

  1. Error detection   – Spot the problem. The agent checks HTTP status codes
                         (404 and 500 mean different things), enforces a
                         response deadline, validates the shape of every tool
                         payload, and reviews the model's answer against the
                         data it was given.

  2. Error handling    – Respond right away, cheapest option first: log the
                         incident, retry transient errors with backoff (and a
                         longer deadline each time), fall back to a backup model
                         or a cached copy, degrade gracefully and say so, and
                         notify a human when quality has dropped.

  3. Recovery          – Return to a stable state: send tool errors back to the
                         model so it corrects its own call, roll back a failed
                         multi-step action with compensating "undo" steps,
                         escalate to a human with a handoff note, and diagnose
                         the root causes from the incident log afterwards.

Scenario: a customer-support agent works through its queue on a bad day. The
order API is slow and sometimes returns broken payloads, then goes down, the LLM
provider has an outage, the model replies with an answer that ignores the data,
and a refund fails at its last step, after money has already moved. Every event is written to
one incident log, tagged [detect], [handle], [recover] or [notify], so you can
read each ticket as a trip around the detect → handle → recover loop.

The order status tickets are a layered pipeline over shared state (live API →
cached fallback → response agent). The refund ticket is a model-driven tool loop
with a step budget and a circuit breaker. Failures in the demo are simulated;
the LLM calls are real.
"""

import itertools
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as DeadlineExceeded
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
)

load_dotenv(find_dotenv())

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

SUPPORT_SYSTEM = "You are a concise support assistant. Use only the supplied data. Never invent facts."


def llm_chat(messages: list[dict], model: str = MODEL) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def llm_call(prompt: str, system: str = "", model: str = MODEL) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return llm_chat(messages, model)


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object from a string, tolerating fences and trailing commas."""
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{text}")
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Local models often emit trailing commas; strip them and retry.
        repaired = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON object from response:\n{text}") from exc


# ---------------------------------------------------------------------------
# The incident log
# One record of everything the agent noticed and did. Writing it down is itself
# part of error handling, and the diagnosis at the end of the shift reads it back.
# ---------------------------------------------------------------------------

@dataclass
class IncidentLog:
    ticket: str = "-"
    entries: list[str] = field(default_factory=list)
    problems: list[tuple[str, str, str]] = field(default_factory=list)  # (component, ticket, what was detected)

    def _add(self, stage: str, text: str) -> None:
        line = f"[{stage:<7}] {self.ticket} {text}"
        self.entries.append(line)
        print(f"  {line}")

    def detected(self, component: str, text: str) -> None:
        self.problems.append((component, self.ticket, text))
        self._add("detect", f"{component}: {text}")

    def handled(self, text: str) -> None:
        self._add("handle", text)

    def recovered(self, text: str) -> None:
        self._add("recover", text)

    def notify(self, audience: str, text: str) -> None:
        self._add("notify", f"{audience}: {text}")


INCIDENTS = IncidentLog()


# ---------------------------------------------------------------------------
# Stage 1: Error detection
# Every failure is mapped onto a small taxonomy. The class, not the original
# exception type, decides the response.
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base class for failures the agent knows how to reason about."""

    kind = "error"
    hint = ""  # advice appended to the error when it is shown to the model
    opens_circuit = False  # True: one failure disables the tool for the rest of the task


class TransientError(AgentError):
    """Temporary failure (timeout, rate limit, 5xx). Repeating the call may succeed."""

    kind = "transient"
    hint = "It was already retried automatically."


class PermanentError(AgentError):
    """Failure that repeating the same call will not fix (bad credentials, service down)."""

    kind = "permanent"
    hint = "Retrying will not help; do not repeat this call."
    opens_circuit = True


class NotFoundError(PermanentError):
    """The thing asked for does not exist. That is an answer, not an outage."""

    kind = "not_found"
    opens_circuit = False  # another order id may still work


class InvalidRequestError(AgentError):
    """The request or the model's output is wrong. It must be changed, not resent."""

    kind = "invalid_request"
    hint = "Fix the arguments and try again."


class HttpError(Exception):
    """HTTP failure raised by a simulated backend. `status_code` mirrors the
    attribute on the openai SDK's APIStatusError."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"{status_code} {message}")
        self.status_code = status_code


def classify_error(exc: Exception) -> AgentError:
    """Map an SDK or backend exception onto the taxonomy. Unknown errors fail closed."""
    if isinstance(exc, AgentError):
        return exc
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        error: AgentError = TransientError(f"{type(exc).__name__}: {exc}")
    elif isinstance(exc, (APIStatusError, HttpError)):
        text = str(exc) if isinstance(exc, HttpError) else f"{type(exc).__name__}: {exc}"
        status = exc.status_code
        if status == 404:
            error = NotFoundError(text)
        elif status in (400, 422):
            # Typically a malformed request, or a prompt that exceeds the context
            # window. The fix is to change the request (e.g. summarise older turns).
            error = InvalidRequestError(text)
        elif status in (408, 429) or status >= 500:
            # A 429 for an exhausted quota is really permanent; inspect the error
            # code before treating every rate limit as retryable in production.
            error = TransientError(text)
        else:
            error = PermanentError(text)  # 401, 403, ...
    else:
        # Repeating a failure we do not understand is a guess, so do not retry it.
        error = PermanentError(f"unexpected {type(exc).__name__}: {exc}")
    error.__cause__ = exc
    return error


def call_with_deadline(operation: Callable[[], Any], seconds: float) -> Any:
    """Detect an unusually slow response: give up if it takes longer than `seconds`."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(operation).result(timeout=seconds)
    except DeadlineExceeded:
        raise TransientError(f"no response within {seconds:.2f}s") from None
    finally:
        pool.shutdown(wait=False)


def validate_order(record: object) -> dict:
    """Detect a malformed tool output before the agent trusts it."""
    if not isinstance(record, dict):
        raise TransientError(f"malformed payload: expected an object, got {type(record).__name__}")
    missing = [key for key in ("order_id", "status", "carrier", "eta") if not record.get(key)]
    if missing:
        raise TransientError(f"malformed payload: missing {', '.join(missing)}")
    return record


def check_grounding(answer: str, required: dict[str, tuple[str, ...]]) -> None:
    """Detect an incoherent or ungrounded model answer. `required` maps a label to
    the alternative phrasings of a fact the answer must state. A cheap rule-based
    reviewer; a second LLM can play the same role for fuzzier checks."""
    text = answer.lower()
    missing = [label for label, options in required.items() if not any(o.lower() in text for o in options)]
    if missing:
        raise InvalidRequestError(f"does not mention {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Stage 2: Error handling
# Retry with backoff, model fallback, notification. Every step is logged.
# ---------------------------------------------------------------------------

def retry_with_backoff(
    operation: Callable[[int], Any], label: str, max_attempts: int = 3, base_delay: float = 0.1
) -> Any:
    """Retry transient errors with growing, jittered delays. `operation` receives
    the attempt number so it can adjust its parameters (e.g. a longer deadline).
    Permanent and invalid-request errors are raised immediately: repeating them
    cannot help."""
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(attempt)
        except Exception as exc:
            error = classify_error(exc)
            INCIDENTS.detected(label, f"{error.kind} - {error}")
            if not isinstance(error, TransientError):
                INCIDENTS.handled(f"{label}: not retrying ({error.kind})")
                raise error
            if attempt == max_attempts:
                INCIDENTS.handled(f"{label}: giving up after {attempt} attempts")
                raise error
            # Jitter keeps many agents that failed together from all retrying at once.
            delay = base_delay * 2 ** (attempt - 1) * random.uniform(0.5, 1.0)
            INCIDENTS.handled(f"{label}: retry {attempt}/{max_attempts - 1} in {delay:.2f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")  # the loop always returns or raises


@dataclass
class StubResponse:
    """Just enough of an HTTP response for the SDK to build its exceptions. This
    keeps the demo independent of which HTTP library the installed openai
    version uses (httpx or httpx2)."""

    status_code: int
    request: object = None
    headers: dict = field(default_factory=dict)


def api_error(error_class: type[APIStatusError], status: int, message: str) -> APIStatusError:
    return error_class(message, response=StubResponse(status), body=None)


@dataclass
class ModelEndpoint:
    """A model the agent can call. `faults` are used one per call, in order: an
    exception is raised, a string is returned as the model's (bad) reply. Once they
    run out, calls reach the real model. itertools.repeat(error) is an outage."""

    name: str
    model: str
    faults: Iterator[Exception | str] = field(default_factory=lambda: iter(()))

    def ask(self, prompt: str, system: str = "") -> str:
        fault = next(self.faults, None)
        if isinstance(fault, Exception):
            raise fault
        if fault is not None:
            return fault
        return llm_call(prompt, system, self.model)


def ask_models(models: list[ModelEndpoint], prompt: str) -> str | None:
    """Walk the model chain (primary → backup). None means every model is down."""
    for position, endpoint in enumerate(models):
        try:
            return retry_with_backoff(
                lambda attempt, e=endpoint: e.ask(prompt, SUPPORT_SYSTEM),
                label=endpoint.name,
                max_attempts=2,
            )
        except AgentError:
            if position + 1 < len(models):
                INCIDENTS.handled(f"fallback: {endpoint.name} -> {models[position + 1].name}")
    return None


# ---------------------------------------------------------------------------
# Order status tickets: a layered pipeline over shared state
# primary_handler (live API) → fallback_handler (cache) → response_agent (LLM).
# Each agent reads and writes `state`, so later agents know what earlier ones
# managed to do.
# ---------------------------------------------------------------------------

ORDERS = {
    "4821": {"order_id": "4821", "status": "shipped", "carrier": "UPS", "eta": "Thursday"},
    "7310": {"order_id": "7310", "status": "shipped", "carrier": "DHL", "eta": "Monday"},
}
CACHE = {"7310": ORDERS["7310"]}  # snapshot taken three hours ago


@dataclass
class OrderAPI:
    """Simulated live order service. `script` says what each call does, in order
    (`slow`, `malformed` or `500`); after that calls behave normally."""

    script: Iterator[str]

    def lookup(self, order_id: str) -> dict:
        behaviour = next(self.script, "ok")
        if behaviour == "500":
            raise HttpError(500, "internal server error")
        if behaviour == "malformed":
            return {"order_id": order_id, "status": "shipped"}  # carrier and eta missing
        if behaviour == "slow":
            time.sleep(0.5)
        if order_id not in ORDERS:
            raise HttpError(404, f"order {order_id} not found")
        return ORDERS[order_id]


def primary_handler(state: dict, api: OrderAPI) -> None:
    """Agent 1: ask the live order API."""

    def lookup(attempt: int) -> dict:
        deadline = 0.2 * 2 ** (attempt - 1)  # adjusted parameter: each retry waits longer
        if attempt > 1:
            INCIDENTS.handled(f"live order API: attempt {attempt} with a {deadline:.1f}s deadline")
        return validate_order(call_with_deadline(partial(api.lookup, state["order_id"]), deadline))

    try:
        state["data"] = retry_with_backoff(lookup, "live order API")
        state["source"] = "live order API"
    except AgentError as exc:
        state["primary_error"] = exc


def fallback_handler(state: dict) -> None:
    """Agent 2: runs only if agent 1 failed. Tries the cache and degrades gracefully."""
    error = state.get("primary_error")
    if error is None:
        return
    if isinstance(error, NotFoundError):
        INCIDENTS.handled("no fallback: the order does not exist, so no other source can help")
        state["problem"] = f"order {state['order_id']} does not exist in our system"
        return
    INCIDENTS.handled("fallback: live order API -> cached snapshot")
    snapshot = CACHE.get(state["order_id"])
    if snapshot is None:
        INCIDENTS.detected("cached snapshot", "cache miss")
        state["problem"] = "the order system is unavailable and there is no cached copy"
        return
    state.update(data=snapshot, source="cached snapshot", stale=True)
    INCIDENTS.handled("graceful degradation: answering from the cached snapshot and flagging it as possibly out of date")
    INCIDENTS.notify("ops on-call", "live order API is failing; the agent is serving cached data")


def template_reply(state: dict) -> str:
    """Last-resort reply built directly from the data, with no model involved."""
    data = state["data"]
    if data is None:
        return f"Sorry, I cannot give you an answer right now: {state['problem']}. Please contact support."
    text = f"Order {data['order_id']} is {data['status']} with {data['carrier']}, estimated delivery {data['eta']}."
    return text + (" This comes from a cached copy and may be out of date." if state["stale"] else "")


STALE_WARNING = ("cached", "out of date", "outdated", "out-of-date", "not up to date", "may have changed", "not be current")


def response_agent(state: dict, models: list[ModelEndpoint]) -> str:
    """Agent 3: write the customer reply, honestly for whatever data we ended up with."""
    data, question = state["data"], state["question"]
    if data is None:
        prompt = (
            f"Customer question: {question}\nProblem: {state['problem']}\n"
            "Apologise briefly. Do not guess or invent order details. Suggest one next step "
            "the customer can take, without phone numbers, email addresses or placeholders. "
            "Keep it to two sentences."
        )
        required: dict[str, tuple[str, ...]] = {}
    elif state["stale"]:
        prompt = (
            f"Customer question: {question}\n"
            f"Order data from a cached copy that may be out of date: order {data['order_id']} is "
            f"{data['status']}, carrier {data['carrier']}, estimated delivery {data['eta']}.\n"
            "Write two sentences. Sentence 1 gives the status and delivery day. Sentence 2 says the "
            "figures come from a cached copy and may be out of date."
        )
        required = {f"delivery day {data['eta']}": (data["eta"],), "a stale-data warning": STALE_WARNING}
    else:
        prompt = (
            f"Customer question: {question}\n"
            f"Order data (live): order {data['order_id']} is {data['status']}, carrier "
            f"{data['carrier']}, estimated delivery {data['eta']}.\n"
            "Answer in two sentences. State the carrier and the delivery day."
        )
        required = {f"carrier {data['carrier']}": (data["carrier"],), f"delivery day {data['eta']}": (data["eta"],)}

    feedback = ""
    for attempt in (1, 2):
        draft = ask_models(models, prompt + feedback)
        if draft is None:
            INCIDENTS.detected("model chain", "every model is unavailable")
            break
        try:
            check_grounding(draft, required)
            return draft
        except InvalidRequestError as exc:
            INCIDENTS.detected("response review", f"the reply {exc}")
            feedback = f"\n\nA reviewer rejected your previous reply because it {exc}. Rewrite it using the data. Reply with the customer message only."
            if attempt == 1:
                INCIDENTS.recovered("self-correction: asking the model to rewrite its reply")
    INCIDENTS.handled("graceful degradation: using a template reply built directly from the data")
    return template_reply(state)


@dataclass
class StatusTicket:
    ticket_id: str
    question: str
    order_id: str
    api: OrderAPI
    models: list[ModelEndpoint]


def status_tickets() -> list[StatusTicket]:
    def models(primary_faults: Iterator[Exception | str] = iter(())) -> list[ModelEndpoint]:
        return [ModelEndpoint("primary model", MODEL, primary_faults), ModelEndpoint("backup model", MODEL)]

    # In production the backup is a different provider or a smaller model. Here both
    # resolve to the same local model so the demo needs only one pull.
    ignores_data = "Thanks for contacting us! Your invoice has been paid in full and your account is in good standing."
    outage = api_error(InternalServerError, 503, "503 model overloaded")
    return [
        # Slow response, then a broken payload, then success. The model's first draft ignores the data.
        StatusTicket("T-101", "Where is order 4821?", "4821", OrderAPI(iter(["slow", "malformed", "slow"])), models(iter([ignores_data]))),
        # The order API is down for good. A cached copy exists.
        StatusTicket("T-102", "Where is order 7310?", "7310", OrderAPI(itertools.repeat("500")), models()),
        # The order does not exist, and the primary model provider is down.
        StatusTicket("T-103", "Where is order 0001?", "0001", OrderAPI(iter(())), models(itertools.repeat(outage))),
    ]


def handle_status_ticket(ticket: StatusTicket) -> None:
    INCIDENTS.ticket = ticket.ticket_id
    print(f'\n--- {ticket.ticket_id}: "{ticket.question}" ---')
    state = {
        "question": ticket.question,
        "order_id": ticket.order_id,
        "data": None,
        "source": None,
        "stale": False,
        "problem": None,
    }
    primary_handler(state, ticket.api)
    fallback_handler(state)
    reply = response_agent(state, ticket.models)
    print(f"Reply (data: {state['source'] or 'none'}): {reply}")


# ---------------------------------------------------------------------------
# Stage 3: Recovery
# Self-correction in a tool loop, rollback of multi-step actions, escalation to a
# human, and diagnosis from the incident log.
# ---------------------------------------------------------------------------

# The tool loop. Each turn the model replies with one JSON action: call a tool or
# finish. Tool failures come back as observations so the model can react to them.
# The harness, not the model, enforces the limits.

ACTION_PROTOCOL = (
    "Reply with exactly ONE JSON object per turn and nothing else.\n"
    'To call a tool:  {"action": "call_tool", "tool": "<tool name>", "arguments": {...}}\n'
    'To finish:       {"action": "final", "answer": "<message for the user>"}\n'
    "Every tool call is answered with RESULT: or ERROR: in the next message. If a "
    "tool returns an error, read it, fix the arguments and try again. Once the "
    "request is fulfilled, finish; never repeat a call that already succeeded."
)


@dataclass
class Tool:
    name: str
    usage: str  # one line shown to the model: signature plus argument rules
    run: Callable[[dict], str]


@dataclass
class AgentRun:
    answer: str
    steps: int
    stopped_by: str  # "answer" or "step_budget"
    open_circuits: list[str]  # tools cut off after repeated failures


def chat(messages: list[dict]) -> str:
    """One model turn, retried on transient API failures."""
    return retry_with_backoff(lambda attempt: llm_chat(messages), label="agent model")


def parse_action(raw: str) -> dict:
    """Validate one model turn. The error text is written for the model to act on."""
    try:
        action = extract_json_object(raw)
    except ValueError:
        raise InvalidRequestError("your reply was not a JSON object. Reply with exactly one JSON object.") from None
    if action.get("action") == "final" and isinstance(action.get("answer"), str):
        return action
    if (
        action.get("action") == "call_tool"
        and isinstance(action.get("tool"), str)
        and isinstance(action.get("arguments", {}), dict)
    ):
        return action
    raise InvalidRequestError(
        'the "action" field must be "call_tool" (with "tool" and "arguments") or "final" (with "answer").'
    )


def circuit_open_message(name: str, last_error: object) -> str:
    return (
        f"ERROR (circuit_open): '{name}' is disabled for this task (last error: {last_error}). "
        "Do not call it again. Finish now: tell the user plainly what you could not do and "
        "what they can do next."
    )


def execute_tool(registry: dict[str, Tool], action: dict, failures: dict[str, int], failure_limit: int) -> str:
    """Run one tool call and return the observation the model will see."""
    name = action["tool"]
    tool = registry.get(name)
    if tool is None:
        INCIDENTS.detected("agent model", f"called unknown tool {name!r}")
        return f"ERROR (invalid_request): unknown tool {name!r}. Available tools: {', '.join(registry)}"
    if failures.get(name, 0) >= failure_limit:
        # Circuit is open: do not touch the failing tool again.
        return circuit_open_message(name, "not executed")
    try:
        # Transient tool errors are retried here, cheaply, so the model only sees
        # a failure that survived the retries.
        result = retry_with_backoff(lambda attempt: tool.run(action.get("arguments", {})), f"tool {name}", max_attempts=2)
    except AgentError as exc:
        # A permanent failure opens the circuit at once; other failures open it
        # after `failure_limit` in a row.
        failures[name] = failure_limit if exc.opens_circuit else failures.get(name, 0) + 1
        if failures[name] >= failure_limit:
            INCIDENTS.handled(f"circuit breaker opened for {name}: the agent will not call it again")
            return circuit_open_message(name, exc)
        if isinstance(exc, InvalidRequestError):
            INCIDENTS.recovered(f"self-correction: sending the {name} error back to the model")
        return f"ERROR ({exc.kind}): {exc}. {exc.hint}"
    if failures.get(name):
        INCIDENTS.recovered(f"self-correction worked: {name} succeeded after {failures[name]} failed attempt(s)")
    failures[name] = 0  # a success closes the circuit again
    # Models tend to keep acting after a side-effecting call succeeds (a duplicate
    # refund, say), so the observation says what to do next.
    return f"RESULT: {result} If this fulfils the request, finish now."


def run_agent(
    task: str,
    tools: list[Tool],
    system: str,
    first_turn: str | None = None,
    max_steps: int = 6,
    failure_limit: int = 3,
) -> AgentRun:
    """Run the loop. `first_turn` replaces the model's first reply (used to script
    a known mistake); every later turn is the real model. The step budget bounds
    the whole run; the circuit breaker disables a tool that fails permanently or
    keeps failing, so the agent never retries an action that cannot succeed."""
    registry = {tool.name: tool for tool in tools}
    catalogue = "\n".join(f"- {tool.usage}" for tool in tools)
    messages = [
        {"role": "system", "content": f"{system}\n\nTools:\n{catalogue}\n\n{ACTION_PROTOCOL}"},
        {"role": "user", "content": task},
    ]
    failures: dict[str, int] = {}

    def open_circuits() -> list[str]:
        return [name for name, count in failures.items() if count >= failure_limit]

    for step in range(1, max_steps + 1):
        raw = first_turn if (step == 1 and first_turn) else chat(messages)
        print(f"  step {step}: {raw}")
        messages.append({"role": "assistant", "content": raw})
        try:
            action = parse_action(raw)
        except InvalidRequestError as exc:
            INCIDENTS.detected("agent model", "output is not a valid action")
            observation = f"ERROR ({exc.kind}): {exc}"
        else:
            if action["action"] == "final":
                return AgentRun(action["answer"], step, "answer", open_circuits())
            observation = execute_tool(registry, action, failures, failure_limit)
        print(f"          {observation}")
        messages.append({"role": "user", "content": observation})

    # The budget ran out before the model finished. Stop it from acting and ask
    # for an honest status report instead of letting it run on.
    INCIDENTS.handled(f"step budget of {max_steps} used up")
    messages.append(
        {
            "role": "user",
            "content": (
                "Your step budget is used up. Do not call tools. In two plain-text "
                "sentences, say what was done, what was not, and what the user should do next."
            ),
        }
    )
    return AgentRun(chat(messages), max_steps, "step_budget", open_circuits())


# Rollback: an agent's multi-step action can change real systems. Each step is
# registered with a compensating undo; on failure, completed steps are undone in
# reverse order so the system is left consistent.

@dataclass
class Environment:
    resources: list[str] = field(default_factory=list)


@dataclass
class PlanStep:
    name: str
    action: Callable[[], None]
    undo: Callable[[], None]


@dataclass
class PlanOutcome:
    completed: list[str] = field(default_factory=list)
    failed_step: str | None = None
    error: str | None = None
    rolled_back: list[str] = field(default_factory=list)
    undo_failures: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.failed_step is None


def run_with_rollback(steps: list[PlanStep]) -> PlanOutcome:
    outcome = PlanOutcome()
    done: list[PlanStep] = []
    for step in steps:
        try:
            step.action()
        except AgentError as exc:
            outcome.failed_step, outcome.error = step.name, str(exc)
            INCIDENTS.detected(f"step {step.name}", str(exc))
            break
        done.append(step)
        outcome.completed.append(step.name)

    if not outcome.succeeded:
        for step in reversed(done):
            try:
                step.undo()
                outcome.rolled_back.append(step.name)
                INCIDENTS.recovered(f"rolled back {step.name}")
            except Exception as exc:  # keep undoing the rest and report what is left over
                outcome.undo_failures.append(f"{step.name}: {exc}")
                INCIDENTS.detected(f"rollback {step.name}", f"could not undo: {exc}")
    return outcome


def write_handoff_note(task: str, outcome: PlanOutcome, env: Environment) -> str:
    system = (
        "You write incident handoff notes for human operators. Be factual and "
        "brief. Only mention facts that are listed in the input."
    )
    # State the end result explicitly: models misread a list of "completed"
    # steps as the current state even when those steps were later undone.
    if env.resources:
        state = f"Resources still provisioned: {', '.join(env.resources)}."
    else:
        state = "Nothing is changed. The system is back to its original state."
    prompt = (
        f"Task: {task}\n"
        f"Failed step: {outcome.failed_step}\n"
        f"Error: {outcome.error}\n"
        f"Steps that succeeded and were then undone: {', '.join(outcome.rolled_back) or 'none'}\n"
        f"Steps that could not be undone: {', '.join(outcome.undo_failures) or 'none'}\n"
        f"CURRENT SYSTEM STATE: {state}\n\n"
        "Write a handoff note of at most four lines: what failed and why, the "
        "current system state exactly as given above, and one recommended next "
        "action. Output only the note."
    )
    return llm_call(prompt, system)


# The refund ticket: the model drives a tool; the tool is a multi-step action.

ORDER_TOTALS = {"4821": 39.90}


def validate_refund(args: dict) -> None:
    problems = []
    order_id = str(args.get("order_id"))
    if order_id not in ORDER_TOTALS:
        raise NotFoundError(f"order {order_id} does not exist")
    try:
        amount = float(args["amount"])
        if not 0 < amount <= ORDER_TOTALS[order_id]:
            problems.append(f"amount must be between 0 and the order total {ORDER_TOTALS[order_id]} (got {amount})")
    except (KeyError, TypeError, ValueError):
        problems.append(f"amount must be a plain number such as 12.5 (got {args.get('amount')!r})")
    if not str(args.get("reason", "")).strip():
        problems.append("reason must be a non-empty string")
    if problems:
        raise InvalidRequestError("; ".join(problems))


@dataclass
class RefundDesk:
    """Simulated payment and email systems. Refunding is three steps with side effects."""

    email_up: bool
    env: Environment = field(default_factory=Environment)
    outcome: PlanOutcome | None = None

    def issue_refund(self, args: dict) -> str:
        validate_refund(args)
        order_id, amount = args["order_id"], float(args["amount"])
        reservation, credit = f"reservation:{order_id}", f"credit:{amount:.2f} EUR"

        def add(resource: str) -> Callable[[], None]:
            return lambda: self.env.resources.append(resource)

        def remove(resource: str) -> Callable[[], None]:
            return lambda: self.env.resources.remove(resource)

        def send_confirmation() -> None:
            if not self.email_up:
                raise PermanentError("email service rejected the message: mailbox unavailable")

        self.outcome = run_with_rollback(
            [
                PlanStep("reserve_refund", add(reservation), remove(reservation)),
                PlanStep("credit_payment", add(credit), remove(credit)),
                PlanStep("send_confirmation", send_confirmation, lambda: None),
            ]
        )
        if not self.outcome.succeeded:
            raise PermanentError(f"refund stopped at {self.outcome.failed_step} and was rolled back")
        return f"refund of {amount:.2f} EUR issued for order {order_id}."


def run_refund_ticket() -> None:
    INCIDENTS.ticket = "T-104"
    request = "Please refund order 4821 (39.90 EUR). It arrived damaged."
    print(f'\n--- T-104: "{request}" ---')

    desk = RefundDesk(email_up=False)
    refund_tool = Tool(
        "issue_refund",
        "issue_refund(order_id: string, amount: number in EUR, reason: non-empty string) - "
        "refunds an order to the customer's original payment method",
        desk.issue_refund,
    )
    # The model's first turn is scripted with an invalid amount so the demo always
    # exercises self-correction. Every later turn is the real model.
    flawed_turn = json.dumps(
        {
            "action": "call_tool",
            "tool": "issue_refund",
            "arguments": {"order_id": "4821", "amount": "39,90 EUR", "reason": "arrived damaged"},
        }
    )
    run = run_agent(request, [refund_tool], "You are a refunds assistant for the support team.", first_turn=flawed_turn)

    escalated = False
    if desk.outcome is not None and not desk.outcome.succeeded:
        # The refund could not be completed even after recovery, so a human takes over.
        note = write_handoff_note(f"Refund order 4821 (39.90 EUR): {request}", desk.outcome, desk.env)
        INCIDENTS.recovered("escalated to a human with a handoff note")
        INCIDENTS.notify("refunds team", "refund could not be completed automatically, manual follow-up needed")
        print(f"Handoff note for the refunds team:\n{note}")
        escalated = True
    elif run.open_circuits or run.stopped_by != "answer":
        # The loop got stuck before the refund ran; nothing was changed.
        INCIDENTS.recovered("escalated to a human: the agent could not finish the request")
        INCIDENTS.notify("refunds team", f"agent stopped early (stopped by {run.stopped_by}, disabled: {run.open_circuits or 'none'})")
        escalated = True

    if escalated:
        # The model must not promise anything the harness has not done, so the
        # customer gets a fixed message once the case is escalated.
        state = "No money has moved." if not desk.env.resources else "Part of the refund may need review."
        print(f"Reply: We could not complete your refund automatically. {state} A human agent will contact you.")
    else:
        print(f"Reply: {run.answer}")


# ---------------------------------------------------------------------------
# Diagnosis
# After the shift, the LLM reads the incident log and looks for root causes, so
# the same failures are less likely to recur.
# ---------------------------------------------------------------------------

def diagnose() -> None:
    """Ask the LLM for a likely root cause and a follow-up for each component that
    had trouble. One small prompt per component works far better with a small model
    than one prompt over the whole log."""
    by_component: dict[str, Counter] = {}
    for component, ticket, text in INCIDENTS.problems:
        by_component.setdefault(component, Counter())[f"{ticket}: {text}"] += 1
    system = (
        "You are a site reliability assistant. Be factual and brief. Use only the facts you are given; "
        "do not mention networks, DNS or anything else that is not listed."
    )
    for component, problems in by_component.items():
        observed = "\n".join(f"- {text}" + (f" (x{n})" if n > 1 else "") for text, n in problems.items())
        prompt = (
            f"Component: {component}\nProblems detected during a support agent's shift:\n{observed}\n\n"
            "Reply with exactly one line in this format and nothing else:\n"
            "Cause (hypothesis): <at most 15 words>. Follow-up: <at most 15 words>."
        )
        print(f"- {component}: {llm_call(prompt, system)}")


def run_demo() -> None:
    print("A support agent works through its queue. Today the order API, the LLM provider")
    print("and the refund system all misbehave. Watch each ticket go detect -> handle -> recover.")
    for ticket in status_tickets():
        handle_status_ticket(ticket)
    run_refund_ticket()

    INCIDENTS.ticket = "-"
    print(f"\n=== End of shift: {len(INCIDENTS.entries)} events logged. Diagnosis from the log ===")
    diagnose()


if __name__ == "__main__":
    run_demo()
