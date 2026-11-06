"""
Exception Handling and Recovery Pattern

An agent fails in more places than an ordinary program does. The model API
throttles or times out. The model replies with something that is not valid JSON,
or that a tool rejects. A tool goes down. The model keeps repeating a call that
cannot succeed. A multi-step plan fails halfway, after side effects have already
happened. Without a plan for each, one bad call derails the whole task. This
pattern shows how to classify a failure, respond in proportion to it, and
recover without losing the user's goal. Five complementary mechanisms are
demonstrated, ordered from the cheapest response to the most disruptive:

  1. Retry with backoff           – Model API calls fail with rate limits,
                                    timeouts and 5xx errors. SDK exceptions are
                                    classified; transient ones are retried with
                                    growing, jittered delays and a hard attempt
                                    cap. Bad credentials and over-long prompts
                                    are never retried, because repeating them
                                    cannot help.

  2. Fallback chains and          – When the primary model is down the agent
     graceful degradation           walks an ordered chain (primary model →
                                    backup model → static notice). When a tool
                                    is down it falls back to a cache, and the
                                    LLM is told how fresh its data is, so it
                                    answers honestly instead of inventing an
                                    answer to hide the failure.

  3. Self-correction              – In the agent loop, a tool error is returned
                                    to the model as an observation. The model
                                    reads it and calls the tool again with fixed
                                    arguments. Malformed model output (not JSON,
                                    unknown tool) is handled the same way.

  4. Loop guardrails              – A circuit breaker disables a tool after
                                    repeated consecutive failures, and a step
                                    budget bounds the whole run. Together they
                                    stop an agent that retries a call that can
                                    never succeed.

  5. Rollback and escalation      – When an agent's multi-step plan fails
                                    part-way, compensating "undo" actions run
                                    for the completed steps in reverse order,
                                    leaving the system consistent. The LLM then
                                    writes a handoff note so a human can pick up
                                    from a known state.

All five depend on telling errors apart. Transient errors may succeed if
repeated, permanent errors will not, and invalid-request errors mean the request
(or the model's output) is wrong and must be changed rather than resent.
Provider and tool failures in the demos are simulated; the LLM calls are real.
"""

import itertools
import json
import random
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from functools import partial

from dotenv import find_dotenv, load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)

load_dotenv(find_dotenv())

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

SUPPORT_SYSTEM = "You are a concise support assistant. Use only the supplied data."


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
# Error taxonomy
# Every failure is mapped onto one of three classes; the class, not the original
# exception type, decides the response.
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base class for failures the agent knows how to reason about."""

    kind = "error"


class TransientError(AgentError):
    """Temporary failure (timeout, rate limit, 5xx). Repeating the call may succeed."""

    kind = "transient"


class PermanentError(AgentError):
    """Failure that repeating the same call will not fix (bad credentials, service down)."""

    kind = "permanent"


class InvalidRequestError(AgentError):
    """The request or the model's output is wrong. It must be changed, not resent."""

    kind = "invalid_request"


def classify_error(exc: Exception) -> AgentError:
    """Map an SDK or tool exception onto the taxonomy. Unknown errors fail closed."""
    if isinstance(exc, AgentError):
        return exc
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        # A 429 for an exhausted quota is really permanent; inspect the error
        # code before treating every rate limit as retryable in production.
        error: AgentError = TransientError(f"{type(exc).__name__}: {exc}")
    elif isinstance(exc, (BadRequestError, UnprocessableEntityError)):
        # Typically a malformed request or a prompt that exceeds the context
        # window. The fix is to change the request (e.g. summarise older turns).
        error = InvalidRequestError(f"{type(exc).__name__}: {exc}")
    elif isinstance(exc, APIStatusError):
        error = PermanentError(f"{type(exc).__name__}: {exc}")  # 401, 403, 404, ...
    else:
        # Repeating a failure we do not understand is a guess, so do not retry it.
        error = PermanentError(f"unexpected {type(exc).__name__}: {exc}")
    error.__cause__ = exc
    return error


# ---------------------------------------------------------------------------
# Simulated provider trouble
# Faults are raised before the call reaches the real model, as the SDK would
# raise them, so the recovery code handles genuine openai exception types.
# ---------------------------------------------------------------------------

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


def timeout_error() -> APITimeoutError:
    return APITimeoutError(request=None)


@dataclass
class ModelEndpoint:
    """A model the agent can call. `faults` are raised one per call; once they run
    out, calls reach the real model. Use itertools.repeat(error) for an outage."""

    model: str
    faults: Iterator[Exception] = field(default_factory=lambda: iter(()))
    calls: int = 0

    def ask(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        fault = next(self.faults, None)
        if fault is not None:
            raise fault
        return llm_call(prompt, system, self.model)


# ---------------------------------------------------------------------------
# Strategy 1: Retry with exponential backoff
# Only transient errors are retried; anything else propagates immediately.
# ---------------------------------------------------------------------------

def retry_with_backoff(
    operation: Callable[[], str], max_attempts: int = 4, base_delay: float = 0.25
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            error = classify_error(exc)
            if not isinstance(error, TransientError) or attempt == max_attempts:
                raise error
            # Jitter keeps many agents that were throttled together from all
            # retrying at the same instant.
            delay = base_delay * 2 ** (attempt - 1) * random.uniform(0.5, 1.0)
            print(f"  attempt {attempt} failed ({error}); retrying in {delay:.2f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")  # the loop always returns or raises


def run_retry_demo() -> None:
    print("=== Strategy 1: Retry with backoff (model API calls) ===")
    print("Transient API errors are retried:")
    endpoint = ModelEndpoint(
        MODEL,
        faults=iter(
            [
                api_error(RateLimitError, 429, "429 rate limit exceeded"),
                timeout_error(),
            ]
        ),
    )
    prompt = (
        "Customer question: 'Where is my order 4821?'\n"
        "Order system returned: Order 4821: shipped via UPS, estimated delivery Thursday.\n"
        "Reply in two sentences."
    )
    answer = retry_with_backoff(lambda: endpoint.ask(prompt, SUPPORT_SYSTEM))
    print(f"  succeeded on call {endpoint.calls}")
    print(answer)

    print("\nPermanent and invalid-request errors are not retried:")
    for error in (
        api_error(AuthenticationError, 401, "401 invalid API key"),
        api_error(BadRequestError, 400, "400 context_length_exceeded: prompt is too long"),
    ):
        endpoint = ModelEndpoint(MODEL, faults=iter([error]))
        try:
            retry_with_backoff(partial(endpoint.ask, prompt, SUPPORT_SYSTEM))
        except AgentError as exc:
            print(f"  {type(exc).__name__} after {endpoint.calls} call: {exc}")


# ---------------------------------------------------------------------------
# Strategy 2: Fallback chains and graceful degradation
# Try sources in order. For models the chain is primary → backup → static
# notice. For tools the chain is live data → cache, and the LLM is told how good
# the data it got actually is.
# ---------------------------------------------------------------------------

@dataclass
class Source:
    name: str
    fetch: Callable[[], str]
    stale: bool = False


@dataclass
class FetchResult:
    data: str | None
    source: str | None
    stale: bool
    failures: list[str]


def fetch_with_fallback(sources: list[Source]) -> FetchResult:
    failures: list[str] = []
    for source in sources:
        try:
            return FetchResult(source.fetch(), source.name, source.stale, failures)
        except (TransientError, PermanentError) as exc:
            failures.append(f"{source.name}: {exc}")
            print(f"  {source.name} failed ({exc})")
    return FetchResult(None, None, False, failures)


def model_source(name: str, endpoint: ModelEndpoint, prompt: str) -> Source:
    # Each model gets a short retry budget of its own before the chain moves on.
    return Source(
        name,
        lambda: retry_with_backoff(
            lambda: endpoint.ask(prompt, SUPPORT_SYSTEM), max_attempts=2, base_delay=0.1
        ),
    )


def answer_with_degradation(question: str, result: FetchResult) -> str:
    if result.data is None:
        status = "No data could be retrieved. Failures: " + "; ".join(result.failures)
        instruction = (
            "Apologise briefly, say you cannot confirm the answer right now, do "
            "not guess, and suggest one next step the customer can take."
        )
    elif result.stale:
        status = f"Data from '{result.source}' (may be out of date): {result.data}"
        instruction = (
            "Write two sentences. Sentence 1 answers the question and includes "
            "the exact figure from the data. Sentence 2 says the figure comes "
            "from a cached copy and may be out of date."
        )
    else:
        status = f"Data from '{result.source}': {result.data}"
        instruction = "Answer using the data."
    return llm_call(
        f"Customer question: {question}\n{status}\n{instruction} Keep it to two sentences.",
        system="You are a support assistant. Never state facts that are not in the supplied data.",
    )


def run_fallback_demo() -> None:
    print("\n=== Strategy 2: Fallback chains and graceful degradation ===")

    # In production the backup is a different provider or a smaller model. Here
    # both resolve to the same local model so the demo needs only one pull.
    static_notice = Source(
        "static notice",
        lambda: "Our assistant is temporarily unavailable. Please try again in a few minutes.",
    )
    prompt = (
        "Customer question: 'How do I reset my password?'\n"
        "Help article: Open Settings > Security and choose 'Reset password'.\n"
        "Reply in two sentences."
    )

    print("Primary model down, backup model available:")
    outage = api_error(InternalServerError, 503, "503 model overloaded")
    primary = ModelEndpoint(MODEL, faults=itertools.repeat(outage))
    backup = ModelEndpoint(MODEL)
    result = fetch_with_fallback(
        [model_source("primary model", primary, prompt), model_source("backup model", backup, prompt), static_notice]
    )
    print(f"  answered by: {result.source}")
    print(result.data)

    print("\nEvery model down:")
    primary = ModelEndpoint(MODEL, faults=itertools.repeat(outage))
    backup = ModelEndpoint(MODEL, faults=itertools.repeat(outage))
    result = fetch_with_fallback(
        [model_source("primary model", primary, prompt), model_source("backup model", backup, prompt), static_notice]
    )
    print(f"  answered by: {result.source}")
    print(result.data)

    def live_api() -> str:
        raise PermanentError("503 service unavailable")

    def cache() -> str:
        return "Standard plan: $42/month (cached 3 hours ago)"

    def empty_cache() -> str:
        raise PermanentError("cache miss")

    question = "How much is the Standard plan?"

    print("\nPricing tool: live API down, cache available:")
    result = fetch_with_fallback([Source("live API", live_api), Source("cache", cache, stale=True)])
    print(answer_with_degradation(question, result))

    print("\nPricing tool: every source down:")
    result = fetch_with_fallback(
        [Source("live API", live_api), Source("cache", empty_cache, stale=True)]
    )
    print(answer_with_degradation(question, result))


# ---------------------------------------------------------------------------
# The agent loop shared by Strategies 3 and 4
# Each turn the model replies with one JSON action: call a tool or finish. Tool
# failures come back as observations, so the model can react to them. The
# harness, not the model, enforces the limits.
# ---------------------------------------------------------------------------

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
    """One model turn, retried on transient API failures (Strategy 1)."""
    return retry_with_backoff(lambda: llm_chat(messages))


def parse_action(raw: str) -> dict:
    """Validate one model turn. The error text is written for the model to act on."""
    try:
        action = extract_json_object(raw)
    except ValueError:
        raise InvalidRequestError("your reply was not a JSON object. Reply with exactly one JSON object.")
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


def circuit_open_message(name: str, failures: int, last_error: object) -> str:
    return (
        f"ERROR (circuit_open): '{name}' failed {failures} times in a row (last error: "
        f"{last_error}) and is disabled for this task. Do not call it again. Finish now: "
        "tell the user plainly what you could not do and what they can do next."
    )


def execute_tool(
    registry: dict[str, Tool], action: dict, failures: dict[str, int], failure_limit: int
) -> str:
    """Run one tool call and return the observation the model will see."""
    name = action["tool"]
    tool = registry.get(name)
    if tool is None:
        return f"ERROR (invalid_request): unknown tool {name!r}. Available tools: {', '.join(registry)}"
    if failures.get(name, 0) >= failure_limit:
        # Circuit is open: do not touch the failing tool again.
        return circuit_open_message(name, failures[name], "not executed")
    try:
        # Transient tool errors are retried here, cheaply, so the model only
        # sees a failure that survived the retries.
        result = retry_with_backoff(
            lambda: tool.run(action.get("arguments", {})), max_attempts=2, base_delay=0.1
        )
    except AgentError as exc:
        failures[name] = failures.get(name, 0) + 1
        if failures[name] >= failure_limit:
            return circuit_open_message(name, failures[name], exc)
        return f"ERROR ({exc.kind}): {exc}"
    failures[name] = 0  # a success closes the circuit again
    # Models tend to keep acting after a side-effecting call succeeds (a second,
    # duplicate ticket, say), so the observation says what to do next.
    return f"RESULT: {result} If this fulfils the request, finish now."


def run_agent(
    task: str,
    tools: list[Tool],
    system: str,
    first_turn: str | None = None,
    max_steps: int = 6,
    failure_limit: int = 3,
) -> AgentRun:
    """Run the loop. `first_turn` replaces the model's first reply (used to
    script a known mistake); every later turn is the real model."""
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
            observation = f"ERROR ({exc.kind}): {exc}"
        else:
            if action["action"] == "final":
                return AgentRun(action["answer"], step, "answer", open_circuits())
            observation = execute_tool(registry, action, failures, failure_limit)
        print(f"          {observation}")
        messages.append({"role": "user", "content": observation})

    # The budget ran out before the model finished. Stop it from acting and ask
    # for an honest status report instead of letting it run on.
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


# ---------------------------------------------------------------------------
# Strategy 3: Self-correction
# The tool's error message is the model's diagnosis input. A good error says what
# is wrong, what is expected, and any context the model could not know.
# ---------------------------------------------------------------------------

def create_ticket(args: dict) -> str:
    """Simulated tool that validates its arguments strictly."""
    problems = []
    if not str(args.get("title", "")).strip():
        problems.append("title must be a non-empty string")
    if args.get("priority") not in ("low", "medium", "high"):
        problems.append(f"priority must be one of low, medium, high (got {args.get('priority')!r})")
    try:
        due = date.fromisoformat(str(args.get("due_date")))
    except ValueError:
        problems.append(
            f"due_date must be an ISO date YYYY-MM-DD (got {args.get('due_date')!r}); today is {date.today()}"
        )
    else:
        if due < date.today():
            problems.append(f"due_date {due} is in the past; today is {date.today()}")
    if problems:
        raise InvalidRequestError("; ".join(problems))
    return f"ticket TCK-1042 created (title={args['title']!r}, priority={args['priority']}, due {args['due_date']})."


TICKET_TOOL = Tool(
    "create_ticket",
    'create_ticket(title: non-empty string, priority: "low" | "medium" | "high", '
    "due_date: ISO date YYYY-MM-DD, not in the past) - opens a support ticket",
    create_ticket,
)
TICKETING_AGENT = "You are a ticketing assistant for the support team."


def run_self_correction_demo() -> None:
    request = "Open a high-priority ticket to fix the checkout page bug, due tomorrow."
    # The model's first turn is scripted with two invalid arguments so that the
    # demo always exercises recovery. Every later turn is the real model.
    flawed_turn = json.dumps(
        {
            "action": "call_tool",
            "tool": "create_ticket",
            "arguments": {"title": "Fix checkout page bug", "priority": "urgent", "due_date": "tomorrow"},
        }
    )

    print("\n=== Strategy 3: Self-correction (tool errors are observations) ===")
    print(f"Request: {request}")
    run = run_agent(request, [TICKET_TOOL], TICKETING_AGENT, first_turn=flawed_turn)
    print(f"Agent answer: {run.answer}")


# ---------------------------------------------------------------------------
# Strategy 4: Loop guardrails
# A model can keep calling a tool that will never succeed, burning tokens and
# time on near-identical attempts. The circuit breaker disables a tool after
# `failure_limit` consecutive failures; the step budget caps the run no matter
# what the model does. The agent then reports honestly and a human takes over.
# ---------------------------------------------------------------------------

def run_guardrail_demo() -> None:
    request = "Look up order 4821 and tell me its delivery status."

    def order_api(_args: dict) -> str:
        raise TransientError("503 order service unavailable")

    order_tool = Tool(
        "lookup_order",
        "lookup_order(order_id: string) - returns the shipping status of a customer order",
        order_api,
    )

    print("\n=== Strategy 4: Loop guardrails (circuit breaker and step budget) ===")
    print(f"Request: {request}")
    print("The order service is down for the whole run.")
    run = run_agent(request, [order_tool], "You are a support assistant. Never guess.", max_steps=6, failure_limit=2)
    print(f"Agent answer: {run.answer}")
    print(f"Stopped by: {run.stopped_by} after {run.steps} steps; disabled tools: {run.open_circuits or 'none'}")
    if run.open_circuits:
        print("Escalate: a human must investigate " + ", ".join(run.open_circuits) + ".")


# ---------------------------------------------------------------------------
# Strategy 5: Rollback and escalation
# An agent's plan can change real systems. Each action is registered with a
# compensating undo; on failure, undo what completed in reverse order, then hand
# off to a human.
# ---------------------------------------------------------------------------

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
            print(f"  step '{step.name}' failed: {exc}")
            break
        done.append(step)
        outcome.completed.append(step.name)
        print(f"  step '{step.name}' done")

    if not outcome.succeeded:
        for step in reversed(done):
            try:
                step.undo()
                outcome.rolled_back.append(step.name)
                print(f"  rolled back '{step.name}'")
            except Exception as exc:  # keep undoing the rest and report what is left over
                outcome.undo_failures.append(f"{step.name}: {exc}")
                print(f"  could not roll back '{step.name}': {exc}")
    return outcome


def write_escalation(task: str, outcome: PlanOutcome, env: Environment) -> str:
    system = (
        "You write incident handoff notes for human operators. Be factual and "
        "brief. Only mention facts that are listed in the input."
    )
    # State the end result explicitly: models misread a list of "completed"
    # steps as the current state even when those steps were later undone.
    if env.resources:
        state = f"Resources still provisioned: {', '.join(env.resources)}."
    else:
        state = "Nothing is provisioned. The system is back to its original state."
    prompt = (
        f"Task: {task}\n"
        f"Failed step: {outcome.failed_step}\n"
        f"Error: {outcome.error}\n"
        f"Steps that succeeded and were then undone: {', '.join(outcome.rolled_back) or 'none'}\n"
        f"Steps that could not be undone: {', '.join(outcome.undo_failures) or 'none'}\n"
        f"CURRENT SYSTEM STATE: {state}\n\n"
        "Write a handoff note of at most four lines: what failed and why, the "
        "current system state exactly as given above, and one recommended next "
        "action."
    )
    return llm_call(prompt, system)


def run_rollback_demo() -> None:
    env = Environment()

    def add(resource: str) -> Callable[[], None]:
        return lambda: env.resources.append(resource)

    def remove(resource: str) -> Callable[[], None]:
        return lambda: env.resources.remove(resource)

    def grant_vpn() -> None:
        raise PermanentError("identity provider rejected request: group 'vpn-users' does not exist")

    # The agent's plan for the task: one tool call per step, each with its undo.
    plan = [
        PlanStep("create_account", add("account:jlee"), remove("account:jlee")),
        PlanStep("assign_laptop", add("laptop:MBP-221"), remove("laptop:MBP-221")),
        PlanStep("grant_vpn_access", grant_vpn, lambda: None),
    ]
    task = "Onboard new employee J. Lee"

    print("\n=== Strategy 5: Rollback and escalation ===")
    print(f"Task: {task}")
    outcome = run_with_rollback(plan)
    print(f"Resources after rollback: {env.resources or 'none'}")
    if not outcome.succeeded:
        print("\nEscalation note for a human operator:")
        print(write_escalation(task, outcome, env))


def run_demo() -> None:
    run_retry_demo()
    run_fallback_demo()
    run_self_correction_demo()
    run_guardrail_demo()
    run_rollback_demo()


if __name__ == "__main__":
    run_demo()
