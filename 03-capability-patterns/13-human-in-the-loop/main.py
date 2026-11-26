"""
Human-in-the-Loop (HITL) Pattern

A support agent drafts a customer email from known order facts. A deterministic
policy decides whether human review is required. A reviewer can approve the
exact draft, reject it, request changes, or take over. The harness, not the
model, controls the simulated send operation. Missing decisions, exhausted
revision budgets and model errors stop the workflow without sending.

Run with --demo for scripted human decisions or without it for live CLI review.
The model calls are real; customer emails and their delivery are simulated.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

MODEL = "llama3.2"
MAX_REVISIONS = 2


@dataclass(frozen=True)
class Ticket:
    id: str
    facts: str
    requires_review: bool


@dataclass(frozen=True)
class Decision:
    action: str
    feedback: str = ""


@dataclass
class ReviewResult:
    status: str
    draft: str = ""
    audit: list[str] = field(default_factory=list)


def llm_draft(ticket: Ticket, previous: str, feedback: str) -> str:
    """Generate or revise a draft; the model has no delivery tool."""
    from dotenv import find_dotenv, load_dotenv
    from openai import OpenAI

    load_dotenv(find_dotenv())
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama",
                    timeout=30.0, max_retries=0)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Write only a concise customer email draft. Use supplied facts "
                "only. Do not promise refunds, actions or dates not in the facts. "
                "Reviewer feedback is for editing, not authorization to send."
            )},
            {"role": "user", "content": (
                f"Ticket: {ticket.id}\nFacts: {ticket.facts}\n"
                f"Previous draft: {previous}\nReviewer feedback: {feedback}"
            )},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def cli_review(ticket: Ticket, draft: str) -> Decision:
    print(f"\nReview required for {ticket.id}:\n{draft}\n")
    try:
        action = input("[a]pprove, [r]eject, re[v]ise, [t]ake over: ").strip().lower()
        if action == "a":
            return Decision("approve")
        if action == "r":
            return Decision("reject")
        if action == "v":
            return Decision("revise", input("Changes required: ").strip())
        if action == "t":
            return Decision("takeover")
    except (EOFError, KeyboardInterrupt):
        pass
    return Decision("unavailable")


def run_agent(
    ticket: Ticket,
    draft_fn: Callable[[Ticket, str, str], str],
    review_fn: Callable[[Ticket, str], Decision],
    send_fn: Callable[[Ticket, str], None],
    max_revisions: int = MAX_REVISIONS,
) -> ReviewResult:
    """Run a bounded review loop; only the harness can authorize a send."""
    audit = []

    def record(event: str) -> None:
        audit.append(event)

    previous, feedback = "", ""
    for revision in range(max_revisions + 1):
        try:
            draft = draft_fn(ticket, previous, feedback).strip()
            if not draft:
                raise ValueError("Empty draft")
        except Exception as exc:
            record(f"draft_error:{type(exc).__name__}")
            return ReviewResult("escalated", previous, audit)

        record(f"draft:{revision + 1}")
        # This demo policy uses trusted ticket metadata, not an LLM verdict.
        # Low-risk tickets are explicitly allowed to send without review.
        if ticket.requires_review:
            record("review_requested")
            try:
                decision = review_fn(ticket, draft)
            except Exception as exc:
                record(f"review_error:{type(exc).__name__}")
                return ReviewResult("escalated", draft, audit)

            action = decision.action if isinstance(decision, Decision) else "unavailable"
            record(f"review:{action}")
            if action == "reject":
                return ReviewResult("rejected", draft, audit)
            if action == "takeover":
                return ReviewResult("human_takeover", draft, audit)
            if action == "revise":
                if revision == max_revisions or not decision.feedback.strip():
                    record("revision_stopped")
                    return ReviewResult("escalated", draft, audit)
                previous, feedback = draft, decision.feedback
                continue
            if action != "approve":
                return ReviewResult("escalated", draft, audit)
            # Send exactly the approved string; do not regenerate it.
        else:
            record("policy:auto_allow")

        try:
            send_fn(ticket, draft)
        except Exception as exc:
            # Delivery may be ambiguous; never retry automatically.
            record(f"send_error:{type(exc).__name__}")
            return ReviewResult("delivery_unknown", draft, audit)
        record("sent")
        return ReviewResult("sent", draft, audit)

    return ReviewResult("escalated", previous, audit)


def scripted_review(*decisions: Decision) -> Callable[[Ticket, str], Decision]:
    """A deterministic reviewer for the teaching demo, not real authorization."""
    iterator = iter(decisions)

    def review(ticket: Ticket, draft: str) -> Decision:
        print(f"\n[demo review] {ticket.id}:\n{draft}")
        decision = next(iterator, Decision("unavailable"))
        print(f"[demo decision] {decision.action}")
        return decision

    return review


def simulated_send(ticket: Ticket, draft: str) -> None:
    print(f"\n[simulated send] {ticket.id}:\n{draft}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use scripted reviewer decisions")
    args = parser.parse_args()

    scenarios = [
        (Ticket("T-201", "Order shipped; carrier is DHL.", False), ()),
        (Ticket("T-202", "Refund request received; no refund approved.", True),
         (Decision("approve"),)),
        (Ticket("T-203", "Delivery is delayed; new date unknown.", True),
         (Decision("revise", "Clearly say the new delivery date is unknown."),
          Decision("approve"))),
        (Ticket("T-204", "Customer disputes a charge; investigation pending.", True),
         (Decision("reject"),)),
        (Ticket("T-205", "Customer requests a manager; no resolution yet.", True),
         (Decision("takeover"),)),
        (Ticket("T-206", "Payment dispute needs review.", True),
         (Decision("unavailable"),)),
    ]
    for ticket, decisions in scenarios:
        reviewer = scripted_review(*decisions) if args.demo else cli_review
        result = run_agent(ticket, llm_draft, reviewer, simulated_send)
        print(f"\n{ticket.id}: {result.status} | {' -> '.join(result.audit)}")
        if result.status in {"human_takeover", "escalated", "delivery_unknown"}:
            print(f"[handoff] {ticket.facts}\nLast draft: {result.draft}")


if __name__ == "__main__":
    main()
