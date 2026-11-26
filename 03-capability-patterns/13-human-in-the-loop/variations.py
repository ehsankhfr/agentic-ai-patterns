"""HITL variations: human clarification and durable asynchronous approval."""

import json
import sqlite3
import uuid
from contextlib import closing
from tempfile import TemporaryDirectory
from pathlib import Path
from collections.abc import Callable


# ---------------------------------------------------------------------------
# Variation 2: Human clarification before the agent can finish its task.
# The harness decides what information is missing, not the language model.
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("carrier", "tracking_number")


def clarify_and_draft(facts: dict, ask_model: Callable, ask_human: Callable):
    """Return (status, draft, facts); stop if required human input is missing."""
    facts = dict(facts)
    for name in REQUIRED_FIELDS:
        if isinstance(facts.get(name), str) and facts[name].strip():
            continue
        try:
            question = ask_model(
                f"Missing field: {name}. Known facts: {json.dumps(facts)}",
                "Write one short question asking a human operator for the missing field. "
                "Do not invent a value or request other information.",
            ).strip()
            # The field name is displayed separately, even if the model asks poorly.
            answer = ask_human(name, question).strip()
        except (Exception, KeyboardInterrupt):
            return "escalated", "", facts
        if not answer:
            return "escalated", "", facts
        facts[name] = answer
    try:
        draft = ask_model(
            json.dumps(facts),
            "Draft a shipment update using only these facts. Do not claim it was sent.",
        ).strip()
    except Exception:
        return "escalated", "", facts
    return ("drafted", draft, facts) if draft else ("escalated", "", facts)


def clarification_demo(ask_model: Callable, demo: bool) -> None:
    def ask_human(field, question):
        print(f"[clarify {field}] {question}")
        if demo:
            return {"carrier": "DHL", "tracking_number": "DEMO-123"}[field]
        return input(f"Enter {field} (blank to escalate): ")

    status, draft, facts = clarify_and_draft(
        {"order": "4821", "carrier": "DHL"}, ask_model, ask_human,
    )
    print(f"[{status}] {draft or 'Human follow-up required'}")


# ---------------------------------------------------------------------------
# Variation 3: Pause, persist, approve later, then resume exactly once.
# SQLite is the checkpoint and the simulated delivery system. No real send.
# ---------------------------------------------------------------------------

class ApprovalStore:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY, ticket TEXT NOT NULL, draft TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN
                    ('pending', 'approved', 'rejected', 'executed'))
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY, ticket TEXT NOT NULL, draft TEXT NOT NULL
            )""")

    def propose(self, ticket: str, draft: str) -> str:
        if not ticket.strip() or not draft.strip():
            raise ValueError("Ticket and draft must be nonempty")
        request_id = uuid.uuid4().hex
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO approvals VALUES (?, ?, ?, 'pending')",
                       (request_id, ticket, draft))
        return request_id

    def get(self, request_id: str):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT ticket, draft, status FROM approvals WHERE id=?",
                             (request_id,)).fetchone()
        return row

    def decide(self, request_id: str, action: str) -> bool:
        status = {"approve": "approved", "reject": "rejected"}.get(action)
        if status is None:
            return False
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute(
                "UPDATE approvals SET status=? WHERE id=? AND status='pending'",
                (status, request_id),
            ).rowcount == 1

    def execute(self, request_id: str) -> bool:
        """Atomically claim an approved request and record its exact draft."""
        with closing(sqlite3.connect(self.path)) as db, db:
            changed = db.execute(
                "UPDATE approvals SET status='executed' WHERE id=? AND status='approved'",
                (request_id,),
            ).rowcount
            if changed != 1:
                return False
            db.execute("""INSERT INTO outbox SELECT id, ticket, draft
                          FROM approvals WHERE id=?""", (request_id,))
        return True

    def delivered(self, request_id: str):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT ticket, draft FROM outbox WHERE id=?",
                              (request_id,)).fetchone()


def checkpoint_cli(args, ask_model: Callable) -> None:
    def propose(store):
        draft = ask_model(
            "Ticket T-301: delivery delayed; new date unknown.",
            "Draft a customer email using only the facts. Do not promise a date.",
        ).strip()
        return store.propose("T-301", draft)

    if args.demo:
        # Reopen the store at every stage to simulate separate workers/processes.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "approvals.sqlite3"
            request_id = propose(ApprovalStore(path))
            print(f"[paused] {request_id}: {ApprovalStore(path).get(request_id)}")
            print("[demo approval]", ApprovalStore(path).decide(request_id, "approve"))
            print("[resumed]", ApprovalStore(path).execute(request_id))
            print("[simulated delivery]", ApprovalStore(path).delivered(request_id))
            print("[duplicate resume blocked]", not ApprovalStore(path).execute(request_id))
        return

    store = ApprovalStore(args.db)
    action, request_id = args.checkpoint_action, args.request_id
    if action == "propose":
        request_id = propose(store)
        print(f"[pending] {request_id}: {store.get(request_id)}")
        return
    if not request_id:
        raise SystemExit("--request-id is required for this checkpoint action")
    row = store.get(request_id)
    if row is None:
        raise SystemExit("Unknown request ID")
    print(f"[{row[2]}] Ticket {row[0]}\n{row[1]}")
    if action == "decide":
        if row[2] != "pending":
            print("Decision blocked: request is not pending")
            return
        try:
            decision = input("Approve or reject this exact draft? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("No decision recorded")
            return
        print("Decision recorded:", store.decide(request_id, decision))
    elif action == "execute":
        print("Simulated delivery recorded:", store.execute(request_id))
        print("Outbox:", store.delivered(request_id))
