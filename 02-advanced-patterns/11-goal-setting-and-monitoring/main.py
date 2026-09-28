"""
Goal Setting and Monitoring Pattern

Shows how an agent can define measurable goals, track progress, and trigger
corrective actions when progress drifts from targets.
"""

from dataclasses import dataclass, field


@dataclass
class Milestone:
    name: str
    target: float
    current: float = 0.0

    @property
    def progress(self) -> float:
        if self.target <= 0:
            return 0.0
        return min(1.0, self.current / self.target)


@dataclass
class Goal:
    title: str
    milestones: list[Milestone] = field(default_factory=list)

    def overall_progress(self) -> float:
        if not self.milestones:
            return 0.0
        return sum(m.progress for m in self.milestones) / len(self.milestones)

    def needs_intervention(self, threshold: float = 0.6) -> bool:
        return any(m.progress < threshold for m in self.milestones)


def run_demo() -> None:
    goal = Goal(
        title="Ship reliable support assistant",
        milestones=[
            Milestone("Intent routing accuracy", target=0.90, current=0.88),
            Milestone("Average response latency (s)", target=2.0, current=1.5),
            Milestone("Resolved conversations", target=200, current=112),
        ],
    )

    print("=== Goal Setting and Monitoring ===")
    print(f"Goal: {goal.title}")

    for milestone in goal.milestones:
        print(
            f"- {milestone.name}: current={milestone.current}, "
            f"target={milestone.target}, progress={milestone.progress:.0%}"
        )

    print(f"\nOverall progress: {goal.overall_progress():.0%}")

    if goal.needs_intervention():
        print("Status: Attention needed -> trigger corrective actions and replanning.")
    else:
        print("Status: On track.")


if __name__ == "__main__":
    run_demo()
