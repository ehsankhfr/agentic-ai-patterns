"""
Goal Setting and Monitoring Pattern

Long-running agents need a way to track progress toward complex objectives
and course-correct when they fall behind. This pattern shows how to structure
goals hierarchically and monitor them with an LLM-driven intervention loop:

  Goal        – a named objective that can hold both direct Milestones and
                nested sub-Goals. Overall progress is the mean of all
                milestone progresses at every level of the tree, so a
                high-level goal accurately reflects the state of its
                components.

  Milestone   – a single measurable target (current vs. target value) with
                a human-readable corrective action that fires if it falls
                below the monitoring threshold. Progress is clamped to
                [0, 1] so partial progress is always comparable across
                milestones with different units.

  GoalMonitor – walks the full milestone tree, collects every milestone
                below a configurable threshold, and calls the LLM once
                with all lagging milestones in a single prompt. The LLM
                returns a targeted corrective action for each lagging
                milestone and a replanning recommendation for the overall
                goal. If every milestone is on track, no LLM call is made.

  run_demo()  – a customer-support automation scenario. A customer sends a
                free-text message; the LLM identifies which known issue types
                it contains (goal setting), and a playbook per issue turns
                that into a goal tree ("Resolve the customer's support
                request" → one subgoal per issue + "Close the ticket").
                It then loops: review → LLM interventions → record simulated
                support-agent progress → review again. The loop stops as soon
                as every milestone is on track, or when the simulated
                progress runs out and the ticket is escalated to a human.
                Each review receives the previous plan so the LLM can adjust
                it rather than repeat itself. Every LLM prompt is printed.
"""

import json
import textwrap
from dataclasses import dataclass, field

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


def traced_llm_call(prompt: str, system: str) -> str:
    """Call the LLM, printing the exact prompt first so every stage is inspectable."""
    print("  ----- system prompt -----")
    print(textwrap.indent(system, "  "))
    print("  ----- user prompt -----")
    print(textwrap.indent(prompt, "  "))
    print("  ----- end of prompt -----")
    return llm_call(prompt, system=system)


@dataclass
class Milestone:
    name: str
    target: float
    current: float = 0.0
    corrective_action: str = "Review the work and revise the next steps."

    @property
    def progress(self) -> float:
        if self.target <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.target))


@dataclass
class Goal:
    title: str
    milestones: list[Milestone] = field(default_factory=list)
    subgoals: list["Goal"] = field(default_factory=list)

    def overall_progress(self) -> float:
        components = [milestone.progress for milestone in self.milestones]
        components.extend(subgoal.overall_progress() for subgoal in self.subgoals)
        if not components:
            return 0.0
        return sum(components) / len(components)

    def iter_milestones(self, path: str = "") -> list[tuple[str, Milestone]]:
        current_path = f"{path} / {self.title}" if path else self.title
        result = [(current_path, milestone) for milestone in self.milestones]
        for subgoal in self.subgoals:
            result.extend(subgoal.iter_milestones(current_path))
        return result


@dataclass
class MonitorReport:
    progress: float
    interventions: str | None

    @property
    def on_track(self) -> bool:
        return self.interventions is None


@dataclass
class GoalMonitor:
    threshold: float = 0.7

    def review(
        self, goal: Goal, previous_interventions: str | None = None
    ) -> MonitorReport:
        # Step 1: walk the goal tree and compare every milestone to the threshold.
        all_milestones = goal.iter_milestones()
        print(
            f"\n[Monitor] Step 1 - Scan {len(all_milestones)} milestones "
            f"(threshold {self.threshold:.0%}, overall {goal.overall_progress():.0%})"
        )
        behind_schedule = []
        for path, milestone in all_milestones:
            lagging = milestone.progress < self.threshold
            print(
                f"  [{'BEHIND' if lagging else 'ok':>6}] {milestone.name} "
                f"({path.rsplit(' / ', 1)[-1]}): "
                f"{milestone.current:g}/{milestone.target:g} = {milestone.progress:.0%}"
            )
            if lagging:
                behind_schedule.append(
                    {
                        "goal": path,
                        "milestone": milestone.name,
                        "current": milestone.current,
                        "target": milestone.target,
                        "progress": f"{milestone.progress:.0%}",
                        "suggested_action": milestone.corrective_action,
                    }
                )

        interventions = None
        if not behind_schedule:
            # Step 2 (skipped): nothing is lagging, so no LLM call is needed.
            print("[Monitor] Step 2 - No lagging milestones; skipping the LLM call.")
        else:
            # Step 2: ask the LLM for one corrective plan covering every lagging milestone.
            names = ", ".join(item["milestone"] for item in behind_schedule)
            print(
                f"[Monitor] Step 2 - {len(behind_schedule)} lagging milestone(s) "
                f"collected: {names}"
            )
            history = ""
            if previous_interventions:
                print("[Monitor]   Including the previous corrective plan in the prompt.")
                history = (
                    "Previous corrective plan (milestones have been re-measured "
                    "since it was applied):\n"
                    f"{previous_interventions}\n\n"
                    "Adjust the plan based on what has moved. Do not repeat "
                    "advice for milestones that are now on track.\n\n"
                )
            system_prompt = (
                "You are a goal-monitoring agent. Use the supplied measurements "
                "to recommend practical corrective actions and a revised plan."
            )
            user_prompt = (
                f"Top-level goal: {goal.title}\n"
                f"Current overall progress: {goal.overall_progress():.0%}\n"
                f"Milestones below the {self.threshold:.0%} progress threshold:\n"
                f"{json.dumps(behind_schedule, indent=2)}\n\n"
                f"{history}"
                "Recommend a concise corrective plan. Give one targeted action "
                "for each lagging milestone, then state how to replan or adjust "
                "priorities across the subgoals. Do not claim the metrics changed."
            )
            print(f"[Monitor] Step 3 - Calling the LLM ({MODEL}) with this prompt:")
            interventions = traced_llm_call(user_prompt, system_prompt)
            print("[Monitor]   LLM response received.")
        return MonitorReport(
            progress=goal.overall_progress(),
            interventions=interventions,
        )


@dataclass
class Playbook:
    """Standard resolution steps for one type of customer issue."""

    description: str  # how the classifier recognises this issue
    title: str  # becomes the subgoal title
    # (milestone name, steps required to complete it, corrective action)
    steps: list[tuple[str, int, str]]


PLAYBOOKS: dict[str, Playbook] = {
    "damaged_item": Playbook(
        description="the item arrived broken, defective or damaged",
        title="Resolve the damaged item",
        steps=[
            ("Damage verified", 2, "Ask for a photo and check the carrier's delivery record."),
            ("Replacement shipped", 3, "Escalate stock allocation and offer expedited shipping."),
        ],
    ),
    "duplicate_charge": Playbook(
        description="the customer was charged more than once for the same order",
        title="Refund the duplicate charge",
        steps=[
            ("Duplicate charge confirmed", 2, "Match both payments to the order in the billing system."),
            ("Refund issued", 3, "Retry the refund or route it to the billing team."),
        ],
    ),
    "delivery_delay": Playbook(
        description="the order is late or has not arrived",
        title="Resolve the delivery delay",
        steps=[
            ("Shipment located", 2, "Contact the carrier and request a trace."),
            ("New delivery date confirmed", 1, "Offer a firm date or a reshipment."),
        ],
    ),
}


def identify_issues(message: str) -> list[str]:
    """Goal setting, step 1: ask the LLM which known issue types the customer raised."""
    catalog = "\n".join(f"- {key}: {pb.description}" for key, pb in PLAYBOOKS.items())
    print(f"[Goal setting] Calling the LLM ({MODEL}) with this prompt:")
    answer = traced_llm_call(
        (
            f'Customer message:\n"{message}"\n\n'
            f"Issue types:\n{catalog}\n\n"
            "Which issue types does the customer describe? Reply with the matching "
            "issue type keys only, separated by commas. If none match, reply NONE."
        ),
        system=(
            "You are a customer support triage agent. Classify the customer's "
            "message using only the issue types provided."
        ),
    )
    print(f"[Goal setting]   LLM answered: {answer}")
    # Iterate over the catalog (not the answer) so the tree order is deterministic
    # and any key the LLM invented is ignored.
    return [key for key in PLAYBOOKS if key in answer]


def build_goal(issue_keys: list[str]) -> Goal:
    """Goal setting, step 2: turn the identified issues into a goal tree."""
    subgoals = [
        Goal(
            title=PLAYBOOKS[key].title,
            milestones=[
                Milestone(name, target=steps, corrective_action=action)
                for name, steps, action in PLAYBOOKS[key].steps
            ],
        )
        for key in issue_keys
    ]
    subgoals.append(
        Goal(
            title="Close the ticket",
            milestones=[
                Milestone(
                    "Customer confirmed resolution",
                    target=1,
                    corrective_action="Send a summary of what was done and ask the customer to confirm.",
                )
            ],
        )
    )
    return Goal(title="Resolve the customer's support request", subgoals=subgoals)


def print_goal_tree(goal: Goal) -> None:
    print(f"Goal: {goal.title} ({goal.overall_progress():.0%} overall)")
    for subgoal in goal.subgoals:
        print(f"- {subgoal.title}: {subgoal.overall_progress():.0%}")
        for milestone in subgoal.milestones:
            print(
                f"  - {milestone.name}: {milestone.current:g}/{milestone.target:g} steps "
                f"= {milestone.progress:.0%}"
            )


def print_report(label: str, report: MonitorReport) -> None:
    print(f"\n{label}: {report.progress:.0%} overall")
    if report.on_track:
        print("Status: On track.")
        return
    print("Status: Intervention needed.")
    print("Recommended corrective plan:")
    print(report.interventions)


def apply_measurements(goal: Goal, measured: dict[str, float]) -> None:
    """Record simulated post-action measurements and show what changed."""
    for _, milestone in goal.iter_milestones():
        if milestone.name not in measured:
            print(f"  - {milestone.name}: no new measurement ({milestone.progress:.0%})")
            continue
        before = milestone.progress
        old_value = milestone.current
        milestone.current = measured[milestone.name]
        print(
            f"  - {milestone.name}: {old_value:g} -> {milestone.current:g} "
            f"(target {milestone.target:g}), progress {before:.0%} -> {milestone.progress:.0%}"
        )


def run_demo() -> None:
    customer_message = (
        "Hi, my order #4821 arrived with a cracked screen and I was also charged "
        "twice for it. I need this sorted before Friday."
    )
    monitor = GoalMonitor(threshold=0.7)

    print("=== Goal Setting and Monitoring: Customer Support Automation ===")

    print("\n=== Stage 1: Customer request ===")
    print(f'Customer: "{customer_message}"')

    print("\n=== Stage 2: Goal setting ===")
    print("[Goal setting] Step 1 - Identify the issues in the customer's message")
    issues = identify_issues(customer_message)
    if not issues:
        print("[Goal setting] No known issue type matched; hand the ticket to a human agent.")
        return
    print(f"[Goal setting] Step 2 - Build the goal tree from playbooks: {', '.join(issues)}")
    goal = build_goal(issues)
    print_goal_tree(goal)
    print(f"Monitoring threshold: {monitor.threshold:.0%} per milestone")

    # Simulated support-agent activity between reviews. Progress is uneven on
    # purpose (the refund fails once before succeeding), so the loop has to
    # iterate before every milestone is back above the threshold. Values are
    # steps completed per milestone.
    activity_by_round = [
        (
            "Agent verified the damage photo and found the duplicate charge.",
            {"Damage verified": 2, "Duplicate charge confirmed": 2},
        ),
        (
            "Replacement order created; refund submitted but the payment API timed out.",
            {"Replacement shipped": 1, "Refund issued": 1},
        ),
        (
            "Stock allocated and replacement shipped; refund retried successfully.",
            {"Replacement shipped": 3, "Refund issued": 3},
        ),
        (
            "Customer replied and confirmed the resolution.",
            {"Customer confirmed resolution": 1},
        ),
    ]

    print("\n=== Stage 3: Initial review ===")
    report = monitor.review(goal)
    print_report("Initial review", report)

    rounds_run = 0
    for round_number, (activity, measured) in enumerate(activity_by_round, start=1):
        if report.on_track:
            print("\nEvery milestone is on track; stopping the loop early.")
            break
        rounds_run = round_number
        print(f"\n=== Stage 4 (round {round_number}): Act, re-measure, review again ===")
        print(f"Support agent activity: {activity}")
        print("Recording progress:")
        apply_measurements(goal, measured)
        report = monitor.review(goal, previous_interventions=report.interventions)
        print_report(f"Review {round_number}", report)

    print("\n=== Stage 5: Outcome ===")
    print(f"Rounds of corrective action: {rounds_run}")
    print(f"Final overall progress: {report.progress:.0%}")
    if report.on_track:
        print("Ticket resolved: every milestone is on track.")
    else:
        print(
            f"Still behind after {len(activity_by_round)} rounds; "
            "escalate the ticket to a human agent."
        )


if __name__ == "__main__":
    run_demo()
