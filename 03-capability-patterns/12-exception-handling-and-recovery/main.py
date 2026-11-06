"""
Exception Handling and Recovery Pattern

Agents depend on tools, services and models that fail: calls time out, APIs
go down, and generated requests turn out to be malformed. Without a plan for
failure, one bad call derails the whole task. This pattern shows how to
classify a failure, respond in proportion to it, and recover without losing
the user's goal. Four complementary mechanisms are demonstrated, ordered from
the cheapest response to the most disruptive:

  1. Retry with backoff           – Transient errors (timeouts, rate limits)
                                    are retried with exponentially growing
                                    delays and a hard attempt cap. Permanent
                                    errors are never retried, because
                                    repeating them cannot help.

  2. Fallback and graceful        – When a source is down, the agent walks an
     degradation                    ordered chain of alternatives (live API →
                                    cache). The LLM then answers honestly for
                                    whatever it got back: fresh data, stale
                                    data, or nothing at all. It never invents
                                    an answer to hide the failure.

  3. Self-correction              – When a tool rejects a request as invalid,
                                    the validation error is fed back to the
                                    LLM, which diagnoses it and returns
                                    corrected arguments. Repair attempts are
                                    capped so the loop cannot run forever.

  4. Rollback and escalation      – In a multi-step workflow, a failure that
                                    cannot be recovered triggers compensating
                                    "undo" actions for the completed steps, in
                                    reverse order, leaving the system
                                    consistent. The LLM then writes a handoff
                                    note so a human can pick up from a known
                                    state.

All four depend on telling errors apart. Transient errors may succeed if
repeated, permanent errors will not, and invalid-request errors mean the
request itself is wrong and must be changed rather than resent. Failures in
the demos are simulated and deterministic; the LLM calls are real.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"


def llm_call(prompt: str, system: str = "", model: str = MODEL) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


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


class AgentError(Exception):
    """Base class for failures the agent knows how to reason about."""


class TransientError(AgentError):
    """Temporary failure (timeout, rate limit). Repeating the call may succeed."""


class PermanentError(AgentError):
    """Failure that repeating the same call will not fix (service down, bad credentials)."""


class InvalidRequestError(AgentError):
    """The request itself is wrong. It must be changed, not resent."""


# ---------------------------------------------------------------------------
# Strategy 1: Retry with exponential backoff
# Only transient errors are retried; anything else propagates immediately.
# ---------------------------------------------------------------------------

@dataclass
class FlakyService:
    """Simulated dependency that raises the given errors in order, then succeeds."""

    errors: list[AgentError]
    response: str
    calls: int = 0

    def call(self) -> str:
        self.calls += 1
        if self.calls <= len(self.errors):
            raise self.errors[self.calls - 1]
        return self.response


def retry_with_backoff(
    operation: Callable[[], str], max_attempts: int = 4, base_delay: float = 0.25
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except TransientError as exc:
            if attempt == max_attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            print(f"  attempt {attempt} failed ({exc}); retrying in {delay:.2f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")  # the loop always returns or raises


def run_retry_demo() -> None:
    print("=== Strategy 1: Retry with backoff ===")
    print("Transient errors are retried:")
    service = FlakyService(
        errors=[TransientError("timeout"), TransientError("429 rate limited")],
        response="Order 4821: shipped via UPS, estimated delivery Thursday.",
    )
    order = retry_with_backoff(service.call)
    print(f"  succeeded on call {service.calls}")
    print(
        llm_call(
            "Customer question: 'Where is my order 4821?'\n"
            f"Order system returned: {order}\n"
            "Reply in two sentences.",
            system="You are a concise support assistant. Use only the supplied order data.",
        )
    )

    print("\nPermanent errors are not retried:")
    broken = FlakyService(
        errors=[PermanentError("401 invalid API credentials")], response="unused"
    )
    try:
        retry_with_backoff(broken.call)
    except PermanentError as exc:
        print(f"  failed immediately after {broken.calls} call: {exc}")


# ---------------------------------------------------------------------------
# Strategy 2: Fallback chain and graceful degradation
# Try sources in order; tell the LLM how good the data it got actually is.
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
    def live_api() -> str:
        raise PermanentError("503 service unavailable")

    def cache() -> str:
        return "Standard plan: $42/month (cached 3 hours ago)"

    def empty_cache() -> str:
        raise PermanentError("cache miss")

    question = "How much is the Standard plan?"

    print("\n=== Strategy 2: Fallback and graceful degradation ===")
    print("Live API down, cache available:")
    result = fetch_with_fallback([Source("live API", live_api), Source("cache", cache, stale=True)])
    print(answer_with_degradation(question, result))

    print("\nEvery source down:")
    result = fetch_with_fallback(
        [Source("live API", live_api), Source("cache", empty_cache, stale=True)]
    )
    print(answer_with_degradation(question, result))


# ---------------------------------------------------------------------------
# Strategy 3: Self-correction
# The tool's error message is the diagnosis input; the LLM repairs the call.
# ---------------------------------------------------------------------------

TICKET_SCHEMA = (
    "title: non-empty string\n"
    'priority: one of "low", "medium", "high"\n'
    "due_date: ISO date string YYYY-MM-DD, not in the past"
)


def create_ticket(args: dict) -> str:
    """Simulated tool that validates its arguments strictly."""
    problems = []
    if not str(args.get("title", "")).strip():
        problems.append("title must be a non-empty string")
    if args.get("priority") not in ("low", "medium", "high"):
        problems.append(f"priority must be one of low, medium, high (got {args.get('priority')!r})")
    try:
        due = date.fromisoformat(str(args.get("due_date")))
        if due < date.today():
            problems.append(f"due_date {due} is in the past (today is {date.today()})")
    except ValueError:
        problems.append(f"due_date must be an ISO date YYYY-MM-DD (got {args.get('due_date')!r})")
    if problems:
        raise InvalidRequestError("; ".join(problems))
    return f"Ticket created: {args['title']!r}, priority={args['priority']}, due {args['due_date']}"


def repair_request(request: str, args: dict, error: str) -> dict:
    system = (
        "You repair malformed tool calls. Fix only what the error message "
        "describes. Respond ONLY with a JSON object of the corrected arguments."
    )
    prompt = (
        f"User request: {request}\n"
        f"Today's date: {date.today().isoformat()}\n"
        f"Tool argument schema:\n{TICKET_SCHEMA}\n\n"
        f"Rejected arguments:\n{json.dumps(args)}\n"
        f"Error from the tool: {error}\n\n"
        "Return the corrected arguments as a JSON object:"
    )
    try:
        return extract_json_object(llm_call(prompt, system))
    except ValueError:
        return args  # unparseable repair: keep the old arguments, the attempt still counts


def call_with_self_correction(request: str, args: dict, max_attempts: int = 3) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return create_ticket(args)
        except InvalidRequestError as exc:
            print(f"  attempt {attempt} rejected: {exc}")
            if attempt == max_attempts:
                raise
            args = repair_request(request, args, str(exc))
            print(f"  LLM proposed: {json.dumps(args)}")
    raise AssertionError("unreachable")  # the loop always returns or raises


def run_self_correction_demo() -> None:
    request = "Open a high-priority ticket to fix the checkout page bug, due next Friday."
    # Simulated first attempt from an upstream step: two arguments are invalid.
    flawed_args = {"title": "Fix checkout page bug", "priority": "urgent", "due_date": "next Friday"}

    print("\n=== Strategy 3: Self-correction ===")
    print(f"Request: {request}")
    print(f"First attempt: {json.dumps(flawed_args)}")
    try:
        print(call_with_self_correction(request, flawed_args))
    except InvalidRequestError:
        print("  could not self-correct within the attempt limit; escalate instead.")


# ---------------------------------------------------------------------------
# Strategy 4: Rollback and escalation
# Undo completed steps in reverse order, then hand off to a human.
# ---------------------------------------------------------------------------

@dataclass
class Environment:
    resources: list[str] = field(default_factory=list)


@dataclass
class WorkflowStep:
    name: str
    action: Callable[[], None]
    undo: Callable[[], None]


@dataclass
class WorkflowOutcome:
    completed: list[str] = field(default_factory=list)
    failed_step: str | None = None
    error: str | None = None
    rolled_back: list[str] = field(default_factory=list)
    undo_failures: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.failed_step is None


def run_with_rollback(steps: list[WorkflowStep]) -> WorkflowOutcome:
    outcome = WorkflowOutcome()
    done: list[WorkflowStep] = []
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


def write_escalation(task: str, outcome: WorkflowOutcome, env: Environment) -> str:
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

    steps = [
        WorkflowStep("create account", add("account:jlee"), remove("account:jlee")),
        WorkflowStep("assign laptop", add("laptop:MBP-221"), remove("laptop:MBP-221")),
        WorkflowStep("grant VPN access", grant_vpn, lambda: None),
    ]
    task = "Onboard new employee J. Lee"

    print("\n=== Strategy 4: Rollback and escalation ===")
    print(f"Task: {task}")
    outcome = run_with_rollback(steps)
    print(f"Resources after rollback: {env.resources or 'none'}")
    if not outcome.succeeded:
        print("\nEscalation note for a human operator:")
        print(write_escalation(task, outcome, env))


def run_demo() -> None:
    run_retry_demo()
    run_fallback_demo()
    run_self_correction_demo()
    run_rollback_demo()


if __name__ == "__main__":
    run_demo()
