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

  StallDetector – flags a milestone that has started but stopped moving
                across consecutive reviews. Automated retries that keep
                failing look exactly like this, and more LLM advice will not
                fix them.

  Escalation  – human in the loop. When the agent should not continue alone
                it pauses, has the LLM write a handoff brief (what the
                customer wants, what was tried, what is blocked), and asks a
                human to decide. The human can:
                  - HANDLE the blocked step themselves and report the details
                    back (for example a refund reference). The agent records
                    the milestone as done, uses the details in its revised
                    plan, and the loop resumes;
                  - GUIDE the agent, whose revised plan is built around the
                    advice before the loop resumes;
                  - TAKE OVER the ticket (the loop ends).
                Escalation fires when:
                  - a milestone stalls (handle, guide or take over),
                  - the round budget runs out while still behind (take over),
                  - triage cannot match the message to any known issue
                    (take over; there is no goal to resume).

  run_demo()  – a customer-support automation scenario. A customer sends a
                free-text message; the LLM identifies which known issue types
                it contains (goal setting), and a playbook per issue turns
                that into a goal tree ("Resolve the customer's support
                request" → one subgoal per issue + "Close the ticket").
                It then loops: review → LLM interventions → record simulated
                support-agent progress → review again. The loop stops as soon
                as every milestone is on track, or when a human takes over.
                In the demo the refund keeps failing, so the stall detector
                escalates, a human issues the refund themselves and reports
                the reference back to the agent. The human is simulated so
                the demo runs unattended; pass --interactive to answer the
                escalation yourself. Each review receives the previous plan
                so the LLM can adjust it rather than repeat itself. Every
                LLM prompt is printed.
"""

import argparse
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"


def llm_call(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Send one prompt to the model and return the reply text.

    The system message is optional; when `system` is empty none is sent.
    """
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
    """A single measurable target, tracked as a current value against a target.

    `corrective_action` is the suggested fix handed to the LLM when this
    milestone falls below the monitoring threshold.
    """

    name: str
    target: float
    current: float = 0.0
    corrective_action: str = "Review the work and revise the next steps."

    @property
    def progress(self) -> float:
        """Fraction complete, clamped to [0, 1]; 0 if the target is not positive."""
        if self.target <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.target))


@dataclass
class Goal:
    """A named objective holding direct milestones and nested sub-goals."""

    title: str
    milestones: list[Milestone] = field(default_factory=list)
    subgoals: list["Goal"] = field(default_factory=list)

    def overall_progress(self) -> float:
        """Mean progress of this goal's direct milestones and its sub-goals.

        Recursive: each sub-goal contributes its own overall progress as a single
        component, whatever the number of milestones inside it. Returns 0 if the
        goal has neither milestones nor sub-goals.
        """
        components = [milestone.progress for milestone in self.milestones]
        components.extend(subgoal.overall_progress() for subgoal in self.subgoals)
        if not components:
            return 0.0
        return sum(components) / len(components)

    def iter_milestones(self, path: str = "") -> list[tuple[str, Milestone]]:
        """Flatten the tree into (path, milestone) pairs, this goal's own first.

        `path` is the chain of goal titles joined by " / " (for example
        "Top goal / Sub-goal"). Callers leave it empty; it only carries the
        parent path through the recursion.
        """
        current_path = f"{path} / {self.title}" if path else self.title
        result = [(current_path, milestone) for milestone in self.milestones]
        for subgoal in self.subgoals:
            result.extend(subgoal.iter_milestones(current_path))
        return result


@dataclass
class MonitorReport:
    """The outcome of one review: overall progress and the LLM's corrective plan.

    `interventions` is None when no milestone was lagging.
    """

    progress: float
    interventions: str | None

    @property
    def on_track(self) -> bool:
        """True when nothing was lagging, so no corrective plan was needed."""
        return self.interventions is None


@dataclass
class GoalMonitor:
    """Checks a goal tree against a progress threshold and asks the LLM for fixes.

    `threshold` is the minimum per-milestone progress (0-1) counted as on track.
    """

    threshold: float = 0.7

    def review(
        self,
        goal: Goal,
        previous_interventions: str | None = None,
        human_input: str | None = None,
    ) -> MonitorReport:
        """Scan every milestone and, if any lag, get one LLM plan covering them all.

        `previous_interventions` is the last plan, so the LLM adjusts it rather
        than repeating itself. `human_input` is an update from a human, which the
        prompt marks as authoritative. No LLM call is made when nothing is lagging.
        """
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
            if human_input:
                print("[Monitor]   Including the human agent's input in the prompt.")
                history += (
                    "Update from a human support lead (authoritative; it "
                    "overrides any conflicting earlier advice):\n"
                    f"{human_input}\n\n"
                    "Build the revised plan around this update. If an action "
                    "involves telling the customer what was done, include the "
                    "details the human reported (such as reference numbers) "
                    "word for word. Only plan actions for the milestones listed "
                    "above.\n\n"
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
class StallDetector:
    """Flags milestones that have started but stopped moving between reviews.

    A milestone counts as idle for one review when it is in progress (above 0%),
    still below the threshold, and unchanged since the previous review. Milestones
    at 0% are not flagged: they are usually waiting on earlier work (for example
    the customer cannot confirm a resolution that has not happened yet), and the
    round budget catches work that never starts.
    """

    max_idle_reviews: int = 2
    # Both dicts are keyed by (goal path, milestone name).
    _last_progress: dict[tuple[str, str], float] = field(default_factory=dict)  # progress at the previous review
    _idle_reviews: dict[tuple[str, str], int] = field(default_factory=dict)  # consecutive idle reviews so far

    def observe(self, goal: Goal, threshold: float) -> list[str]:
        """Record the latest measurements; return names of milestones now stalled."""
        stalled = []
        for path, milestone in goal.iter_milestones():
            key = (path, milestone.name)
            idle = (
                0 < milestone.progress < threshold
                and self._last_progress.get(key) == milestone.progress
            )
            self._idle_reviews[key] = self._idle_reviews.get(key, 0) + 1 if idle else 0
            self._last_progress[key] = milestone.progress
            if self._idle_reviews[key] >= self.max_idle_reviews:
                stalled.append(milestone.name)
        return stalled

    def reset(self) -> None:
        """Clear idle counts after a human has intervened, so the same stall
        is not reported again before the agent has had a chance to act on it."""
        self._idle_reviews.clear()


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


def format_goal_tree(goal: Goal) -> str:
    """Render the goal, its sub-goals and their milestones as an indented outline.

    Used for console output and as the progress snapshot in handoff briefs.
    """
    lines = [f"Goal: {goal.title} ({goal.overall_progress():.0%} overall)"]
    for subgoal in goal.subgoals:
        lines.append(f"- {subgoal.title}: {subgoal.overall_progress():.0%}")
        for milestone in subgoal.milestones:
            lines.append(
                f"  - {milestone.name}: {milestone.current:g}/{milestone.target:g} steps "
                f"= {milestone.progress:.0%}"
            )
    return "\n".join(lines)


def print_goal_tree(goal: Goal) -> None:
    """Print the outline produced by `format_goal_tree`."""
    print(format_goal_tree(goal))


def print_report(label: str, report: MonitorReport) -> None:
    """Print a review's overall progress, then 'On track' or the corrective plan.

    `label` titles the output (for example "Review 2").
    """
    print(f"\n{label}: {report.progress:.0%} overall")
    if report.on_track:
        print("Status: On track.")
        return
    print("Status: Intervention needed.")
    print("Recommended corrective plan:")
    print(report.interventions)


# ---------------------------------------------------------------------------
# Human-in-the-loop escalation
# ---------------------------------------------------------------------------


@dataclass
class HumanDecision:
    """A human's answer to an escalation: the action they chose and their note."""

    # "handled":  the human completed the blocked step; the note carries the details
    # "guide":    the human advised the agent; the note is the advice
    # "take_over": the human owns the ticket from here; the note is the handoff
    action: str
    note: str


# A reviewer receives the handoff brief and whether the agent can resume (if not,
# the only valid answer is "take_over"), and returns the human's decision.
# Swapping this function is the only thing that changes between the unattended
# demo and a real console or ticketing UI.
Reviewer = Callable[[str, bool], HumanDecision]


def simulated_human(brief: str, can_resume: bool) -> HumanDecision:
    """Stand-in for a support lead so the demo runs unattended."""
    if can_resume:
        return HumanDecision(
            "handled",
            "The payment API is down, so I issued the refund manually in the "
            "billing console for order #4821. Refund reference RF-20931.",
        )
    return HumanDecision("take_over", "I will contact the customer directly.")


def console_human(brief: str, can_resume: bool) -> HumanDecision:
    """Ask the person running the script (the brief has already been printed)."""
    choice = "t"
    if can_resume:
        choice = ""
        while choice not in ("h", "g", "t"):
            choice = input(
                "  [Human] [h]andle it yourself and report back, [g]uide the agent, "
                "or [t]ake over the ticket? "
            ).strip().lower()
    if choice in ("h", "g"):
        prompt = (
            "  [Human] Details to report back to the agent (for example a refund reference): "
            if choice == "h"
            else "  [Human] Guidance for the agent: "
        )
        note = ""
        while not note:
            note = input(prompt).strip()
        return HumanDecision("handled" if choice == "h" else "guide", note)
    note = input("  [Human] Handoff note (optional): ").strip()
    return HumanDecision("take_over", note or "No note provided.")


def describe_decision(decision: HumanDecision, stalled: list[str]) -> str:
    """Phrase a human's answer as facts the monitor can plan around."""
    if decision.action == "handled":
        return (
            f"The human completed {', '.join(stalled)} themselves, so it is now done. "
            f"Details they reported: {decision.note}"
        )
    return f"Guidance for the agent: {decision.note}"


def ticket_context(message: str, goal: Goal, actions: list[str]) -> str:
    """The facts a human needs to pick the ticket up cold.

    The monitor's latest LLM-written plan is left out on purpose: it is long,
    it is advice rather than fact, and for a stalled milestone it is the advice
    that has already failed, so it only pushes the brief toward repeating it.
    """
    taken = "\n".join(f"- {action}" for action in actions) or "- none yet"
    return (
        f'Customer message: "{message}"\n\n'
        f"Goal progress:\n{format_goal_tree(goal)}\n\n"
        f"Actions taken so far:\n{taken}"
    )


def escalate(
    reason: str, context: str, reviewer: Reviewer, can_resume: bool
) -> HumanDecision:
    """Pause the agent, brief a human, and return their decision."""
    print(f"[Escalation] Reason: {reason}")
    print(f"[Escalation] Calling the LLM ({MODEL}) to write the handoff brief:")
    brief = traced_llm_call(
        f"Why this is being escalated: {reason}\n\n{context}\n\n"
        "Write the handoff brief for the human agent.",
        system=(
            "You write handoff briefs for human customer-support agents. Use only "
            "the facts supplied; do not invent actions or results. Reply with the "
            "brief only, no preamble, in at most 5 short lines: what the customer "
            "needs, what has been done, what is blocked, and a final line starting "
            "'Decision needed:' stating what you need the human to decide."
        ),
    )
    print("[Escalation] Handoff brief for the human:")
    print(textwrap.indent(brief, "  "))
    print("[Escalation] Agent paused; waiting for the human's decision.")
    decision = reviewer(brief, can_resume)
    print(f"[Escalation] Human decision: {decision.action} - {decision.note}")
    return decision


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


def run_demo(interactive: bool = False) -> None:
    """Run the customer-support scenario end to end.

    Triage the message, build the goal tree, then loop (act, re-measure, review,
    detect stalls) until every milestone is on track, a human takes over, or the
    rounds run out. With `interactive=True` the person at the console answers
    escalations; otherwise a simulated human does.
    """
    customer_message = (
        "Hi, my order #4821 arrived with a cracked screen and I was also charged "
        "twice for it. I need this sorted before Friday."
    )
    monitor = GoalMonitor(threshold=0.7)
    stalls = StallDetector(max_idle_reviews=2)
    reviewer: Reviewer = console_human if interactive else simulated_human

    print("=== Goal Setting and Monitoring: Customer Support Automation ===")

    print("\n=== Stage 1: Customer request ===")
    print(f'Customer: "{customer_message}"')

    print("\n=== Stage 2: Goal setting ===")
    print("[Goal setting] Step 1 - Identify the issues in the customer's message")
    issues = identify_issues(customer_message)
    if not issues:
        print("[Goal setting] No known issue type matched, so there is no goal to pursue.")
        print("\n=== Escalation: triage failed ===")
        decision = escalate(
            "Triage could not match the message to any known issue type.",
            f'Customer message: "{customer_message}"\n\nNo goal was created.',
            reviewer,
            can_resume=False,
        )
        print(f"\nTicket handed to a human agent. Note: {decision.note}")
        return
    print(f"[Goal setting] Step 2 - Build the goal tree from playbooks: {', '.join(issues)}")
    goal = build_goal(issues)
    print_goal_tree(goal)
    print(f"Monitoring threshold: {monitor.threshold:.0%} per milestone")
    print(
        f"Escalation policy: a milestone that is in progress but unchanged for "
        f"{stalls.max_idle_reviews} reviews, or still behind when the "
        "rounds run out, goes to a human."
    )

    # Simulated support-agent activity between reviews. Progress is uneven on
    # purpose: the payment API keeps timing out, so the refund stalls at 1 of 3
    # steps. Retrying is not working and more LLM advice will not change that,
    # which is the case for a human. After the escalation (between rounds 4 and
    # 5) the refund is done, either by the human (who reports the details back)
    # or by the agent following the human's guidance, and the customer confirms.
    # Values are steps completed per milestone.
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
            "Stock allocated and replacement shipped; refund retried but the payment API timed out again.",
            {"Replacement shipped": 3, "Refund issued": 1},
        ),
        (
            "Refund retried a third time; the payment API is still timing out.",
            {"Refund issued": 1},
        ),
        (
            ("Refund completed and the customer was sent a summary of what was done; "
            "customer replied and confirmed the resolution."),
            {"Refund issued": 3, "Customer confirmed resolution": 1},
        ),
    ]
    actions_taken: list[str] = []

    print("\n=== Stage 3: Initial review ===")
    report = monitor.review(goal)
    print_report("Initial review", report)
    stalls.observe(goal, monitor.threshold)  # baseline for stall detection

    rounds_run = 0
    escalations = 0
    handover: HumanDecision | None = None
    for round_number, (activity, measured) in enumerate(activity_by_round, start=1):
        if report.on_track:
            print("\nEvery milestone is on track; stopping the loop early.")
            break
        rounds_run = round_number
        print(f"\n=== Stage 4 (round {round_number}): Act, re-measure, review again ===")
        print(f"Support agent activity: {activity}")
        actions_taken.append(activity)
        print("Recording progress:")
        apply_measurements(goal, measured)
        report = monitor.review(goal, previous_interventions=report.interventions)
        print_report(f"Review {round_number}", report)

        stalled = stalls.observe(goal, monitor.threshold)
        if not stalled:
            continue

        print(f"\n=== Escalation (after round {round_number}): human in the loop ===")
        escalations += 1
        decision = escalate(
            f"No progress for {stalls.max_idle_reviews} consecutive reviews on: "
            f"{', '.join(stalled)}.",
            ticket_context(customer_message, goal, actions_taken),
            reviewer,
            # Handling or guiding only helps if another round remains.
            can_resume=round_number < len(activity_by_round),
        )
        if decision.action == "take_over":
            handover = decision
            break
        if decision.action == "handled":
            # The human did the blocked work: record it as progress, then let the
            # agent plan the rest using the details they reported.
            print("\nRecording the human's work:")
            apply_measurements(
                goal,
                {m.name: m.target for _, m in goal.iter_milestones() if m.name in stalled},
            )
            actions_taken.append(
                f"A human completed {', '.join(stalled)} and reported: {decision.note}"
            )
        else:
            actions_taken.append(f"Human guidance received: {decision.note}")
        stalls.reset()
        print("\nRe-planning with the human's input:")
        report = monitor.review(goal, human_input=describe_decision(decision, stalled))
        print_report("Revised review", report)

    if handover is None and not report.on_track:
        print("\n=== Escalation (round budget exhausted): human in the loop ===")
        escalations += 1
        handover = escalate(
            f"Still behind after {len(activity_by_round)} rounds.",
            ticket_context(customer_message, goal, actions_taken),
            reviewer,
            can_resume=False,
        )

    print("\n=== Stage 5: Outcome ===")
    print(f"Rounds of corrective action: {rounds_run}")
    print(f"Human escalations: {escalations}")
    print(f"Final overall progress: {report.progress:.0%}")
    if handover is not None:
        print(f"Ticket handed to a human agent. Note: {handover.note}")
    elif escalations:
        print("Ticket resolved after human help: every milestone is on track.")
    else:
        print("Ticket resolved: every milestone is on track.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Goal setting and monitoring demo with human-in-the-loop escalation."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="answer escalations yourself instead of using the simulated human",
    )
    run_demo(interactive=parser.parse_args().interactive)
