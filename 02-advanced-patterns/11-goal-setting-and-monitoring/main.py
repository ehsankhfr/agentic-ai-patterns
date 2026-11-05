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

  run_demo()  – builds a two-level goal tree ("Ship a reliable support
                assistant" → "Improve answer quality" / "Increase adoption"),
                then loops: review → LLM interventions → record simulated
                post-action progress → review again. The loop stops as soon
                as every milestone is on track, or when the simulated
                measurements run out. Each review receives the previous
                plan so the LLM can adjust it rather than repeat itself.
"""

import json
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
        behind_schedule = [
            {
                "goal": path,
                "milestone": milestone.name,
                "current": milestone.current,
                "target": milestone.target,
                "progress": f"{milestone.progress:.0%}",
                "suggested_action": milestone.corrective_action,
            }
            for path, milestone in goal.iter_milestones()
            if milestone.progress < self.threshold
        ]
        interventions = None
        if behind_schedule:
            history = ""
            if previous_interventions:
                history = (
                    "Previous corrective plan (milestones have been re-measured "
                    "since it was applied):\n"
                    f"{previous_interventions}\n\n"
                    "Adjust the plan based on what has moved. Do not repeat "
                    "advice for milestones that are now on track.\n\n"
                )
            interventions = llm_call(
                (
                    f"Top-level goal: {goal.title}\n"
                    f"Current overall progress: {goal.overall_progress():.0%}\n"
                    f"Milestones below the {self.threshold:.0%} progress threshold:\n"
                    f"{json.dumps(behind_schedule, indent=2)}\n\n"
                    f"{history}"
                    "Recommend a concise corrective plan. Give one targeted action "
                    "for each lagging milestone, then state how to replan or adjust "
                    "priorities across the subgoals. Do not claim the metrics changed."
                ),
                system=(
                    "You are a goal-monitoring agent. Use the supplied measurements "
                    "to recommend practical corrective actions and a revised plan."
                ),
            )
        return MonitorReport(
            progress=goal.overall_progress(),
            interventions=interventions,
        )


def print_report(label: str, report: MonitorReport) -> None:
    print(f"\n{label}: {report.progress:.0%} overall")
    if report.on_track:
        print("Status: On track.")
        return
    print("Status: Intervention needed.")
    print(report.interventions)


def run_demo() -> None:
    goal = Goal(
        title="Ship a reliable support assistant",
        subgoals=[
            Goal(
                title="Improve answer quality",
                milestones=[
                    Milestone(
                        "Intent routing accuracy",
                        target=0.90,
                        current=0.60,
                        corrective_action="Review misrouted examples and update routing guidance.",
                    ),
                    Milestone(
                        "Resolved conversations",
                        target=200,
                        current=110,
                        corrective_action="Replan the rollout and prioritize unresolved cases.",
                    ),
                ],
            ),
            Goal(
                title="Increase adoption",
                milestones=[
                    Milestone(
                        "Active support teams",
                        target=80,
                        current=48,
                        corrective_action="Interview inactive teams and address onboarding blockers.",
                    ),
                ],
            ),
        ],
    )
    monitor = GoalMonitor(threshold=0.7)

    print("=== Goal Setting and Monitoring ===")
    print(f"Goal: {goal.title}")
    for subgoal in goal.subgoals:
        print(f"- {subgoal.title}: {subgoal.overall_progress():.0%}")
        for milestone in subgoal.milestones:
            print(f"  - {milestone.name}: {milestone.progress:.0%}")

    # Simulated measurements taken after each round of corrective actions.
    # Progress is uneven on purpose: some milestones recover quickly, others
    # need several rounds, so the loop has to iterate before everything is
    # back above the threshold.
    measurements_by_round = [
        {"Intent routing accuracy": 0.75, "Resolved conversations": 120, "Active support teams": 52},
        {"Resolved conversations": 135, "Active support teams": 55},
        {"Resolved conversations": 150, "Active support teams": 60},
    ]

    report = monitor.review(goal)
    print_report("Initial review", report)

    for round_number, measured in enumerate(measurements_by_round, start=1):
        if report.on_track:
            break
        print(f"\n--- Round {round_number}: record progress after the recommended actions ---")
        for _, milestone in goal.iter_milestones():
            if milestone.name in measured:
                milestone.current = measured[milestone.name]
        report = monitor.review(goal, previous_interventions=report.interventions)
        print_report(f"Review {round_number}", report)

    if report.on_track:
        print("\nAll milestones are back on track.")
    else:
        print(f"\nStill behind after {len(measurements_by_round)} rounds; escalate to a human.")


if __name__ == "__main__":
    run_demo()
