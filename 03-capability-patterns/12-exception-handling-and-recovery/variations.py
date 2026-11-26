"""Two additional recovery strategies, using only simulated, read-only work."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable


@dataclass
class RecoveryResult:
    status: str
    value: object = None
    attempts: int = 0
    events: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Variation 2: Reflection-driven recovery
# A failed generated SQL query is diagnosed, then rewritten and validated.
# ---------------------------------------------------------------------------

SCHEMA = "orders(order_id TEXT, carrier TEXT, eta TEXT)"
ORDER_ID = "4821"


def read_order(query: str) -> dict:
    """Execute a single read-only query in a bounded, disposable demo database."""
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE TABLE orders(order_id TEXT, carrier TEXT, eta TEXT)")
        db.executemany("INSERT INTO orders VALUES (?, ?, ?)",
                       [("4821", "UPS", "Thursday"), ("7310", "DHL", "Monday")])
        # Model-generated SQL is not trusted. No writes, attachments, PRAGMAs,
        # or SQL functions; only reads from the toy orders table.
        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 == "orders" and database == "main":
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        db.set_authorizer(authorize)
        remaining = 100

        def budget():
            nonlocal remaining
            remaining -= 1
            return int(remaining <= 0)

        db.set_progress_handler(budget, 100)
        cursor = db.execute(query)
        columns = [item[0] for item in cursor.description or ()]
        rows = cursor.fetchmany(2)
        if columns != ["order_id", "carrier", "eta"] or len(rows) != 1:
            raise ValueError("Expected exactly one row with order_id, carrier, eta")
        row = dict(zip(columns, rows[0]))
        # Verify the returned values independently against a parameterized lookup.
        expected = db.execute("SELECT order_id, carrier, eta FROM orders WHERE order_id=?",
                              (ORDER_ID,)).fetchone()
        if rows[0] != expected:
            raise ValueError("Result does not match the requested order's recorded facts")
        return row


def reflective_recovery(
    ask: Callable[[str, str], str],
    initial_query: str | None = None,
    max_attempts: int = 3,
) -> RecoveryResult:
    """On an execution/validation failure, diagnose before changing the query."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    events = []
    task = (f"Schema: {SCHEMA}. Return order_id, carrier and eta for order {ORDER_ID}. "
            "Use exactly those column names and return only one row.")
    sql_system = "Generate one read-only SQLite SELECT. Return only raw SQL, no fences."
    try:
        query = initial_query if initial_query is not None else ask(task, sql_system)
    except Exception as exc:
        return RecoveryResult("controlled_failure", events=[f"model_error:{type(exc).__name__}"])

    for attempt in range(1, max_attempts + 1):
        try:
            value = read_order(query)
        except (sqlite3.Error, ValueError, TypeError) as exc:
            events.append(f"detect:{exc}")
            if attempt == max_attempts:
                return RecoveryResult("controlled_failure", attempts=attempt, events=events)
            try:
                reflection = ask(
                    f"{task}\nFailed query: {query}\nException: {exc}",
                    "Explain the likely reason for failure and suggest a concrete correction. "
                    "Do not execute anything or invent schema fields.",
                ).strip()
                if not reflection:
                    raise ValueError("Empty reflection")
                events.append(f"reflect:{reflection}")
                revised = ask(
                    f"{task}\nPrevious query: {query}\nException: {exc}\nReflection: {reflection}",
                    sql_system,
                ).strip()
                if not revised or revised == query.strip():
                    raise ValueError("No changed query proposed")
                query = revised
                events.append("recover:revised_query")
            except Exception as model_error:
                events.append(f"model_error:{type(model_error).__name__}")
                return RecoveryResult("controlled_failure", attempts=attempt, events=events)
        else:
            events.append("recover:validated_result")
            return RecoveryResult("recovered" if attempt > 1 else "completed", value, attempt, events)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Variation 3: Proactive resilience and durable checkpoint recovery
# The demo's stages are read-only/idempotent. Each successful stage and its
# audit event are committed together; a new process can resume later.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    name: str
    healthy: Callable[[], bool]
    run: Callable[[dict], object]


class Checkpoints:
    def __init__(self, path):
        self.path = Path(path)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, plan TEXT NOT NULL, outputs TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, run_id TEXT NOT NULL, event TEXT NOT NULL
            )""")

    def load(self, run_id, plan):
        signature = json.dumps(plan)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?, '{}')", (run_id, signature))
            saved_plan, raw = db.execute("SELECT plan, outputs FROM runs WHERE id=?",
                                         (run_id,)).fetchone()
        if saved_plan != signature:
            raise ValueError("Checkpoint belongs to a different plan")
        return json.loads(raw)

    def event(self, run_id, event):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO events(run_id, event) VALUES (?, ?)", (run_id, event))

    def save(self, run_id, outputs, stage):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE runs SET outputs=? WHERE id=?", (json.dumps(outputs), run_id))
            db.execute("INSERT INTO events(run_id, event) VALUES (?, ?)",
                       (run_id, f"checkpoint:{stage}"))

    def history(self, run_id):
        with closing(sqlite3.connect(self.path)) as db:
            return [row[0] for row in db.execute(
                "SELECT event FROM events WHERE run_id=? ORDER BY seq", (run_id,))]


def monitored_workflow(store: Checkpoints, run_id: str, stages: list[Stage]) -> RecoveryResult:
    """Preflight unfinished dependencies; checkpoint successful read-only stages.

    Single worker per run ID. A failed or unavailable dependency pauses work
    rather than losing progress. Reopening the database restores prior outputs.
    """
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError("Stage names must be unique")
    outputs = store.load(run_id, names)
    # Check every remaining dependency before doing any new work.
    for stage in stages:
        if stage.name in outputs:
            continue
        try:
            ready = stage.healthy()
        except Exception as exc:
            ready = False
            store.event(run_id, f"health_error:{stage.name}:{type(exc).__name__}")
        if not ready:
            store.event(run_id, f"preflight_blocked:{stage.name}")
            return RecoveryResult("paused", outputs, events=store.history(run_id))

    for stage in stages:
        if stage.name in outputs:
            continue
        try:
            # Recheck immediately before the call: preflight is not a guarantee.
            if not stage.healthy():
                raise RuntimeError("Dependency became unavailable")
            value = stage.run(dict(outputs))
            candidate = {**outputs, stage.name: value}
            store.save(run_id, candidate, stage.name)
            outputs = candidate
        except Exception as exc:
            store.event(run_id, f"controlled_failure:{stage.name}:{type(exc).__name__}")
            return RecoveryResult("paused", outputs, events=store.history(run_id))
    store.event(run_id, "completed")
    return RecoveryResult("completed", outputs, events=store.history(run_id))


def reflection_demo(ask):
    print("\n=== Variation 2: Reflection-driven recovery ===")
    # Deliberately invalid column so the model always has an exception to inspect.
    result = reflective_recovery(ask, "SELECT order_id, carrier, delivery_date FROM orders WHERE order_id='4821'")
    for event in result.events:
        print(event)
    print(f"[{result.status}] {result.value}")


def checkpoint_demo(ask, path=None):
    print("\n=== Variation 3: Proactive resilience and checkpoint recovery ===")
    def demonstrate(db_path):
        state = {"up": False, "reported_healthy": False}

        def draft(outputs):
            if not state["up"]:
                raise RuntimeError("Model became unavailable after preflight")
            reply = ask(json.dumps(outputs["lookup"]),
                        "Draft a short shipment update using only the supplied facts.").strip()
            if not reply:
                raise ValueError("Empty model reply")
            return reply

        def stages():
            return [
                Stage("lookup", lambda: True, lambda _: {"carrier": "UPS", "eta": "Thursday"}),
                Stage("draft", lambda: state["reported_healthy"], draft),
            ]

        # First demonstrate that a known outage prevents any new work.
        result = monitored_workflow(Checkpoints(db_path), "shipment-4821", stages())
        print(f"[preflight: {result.status}] {result.value}")
        # A subsequent health check passes, but the dependency fails during execution.
        state["reported_healthy"] = True
        result = monitored_workflow(Checkpoints(db_path), "shipment-4821", stages())
        print(f"[{result.status}] {result.value}")
        state["up"] = True
        # A different store instance reads the saved lookup and only retries the draft.
        result = monitored_workflow(Checkpoints(db_path), "shipment-4821", stages())
        for event in result.events:
            print(event)
        print(f"[{result.status}] {result.value}")

    if path:
        demonstrate(path)
    else:
        with TemporaryDirectory() as temp:
            demonstrate(Path(temp) / "recovery.sqlite3")
