"""
Learning and Adaptation Pattern

Agents that repeat similar tasks should improve over time. This pattern
demonstrates three complementary mechanisms for doing so — none of which
require retraining the underlying model:

  1. Feedback-based rule learning – After a failed attempt the agent asks
                                    the LLM to distil the failure and its
                                    feedback into one concise, reusable
                                    lesson. Accumulated lessons are injected
                                    into the prompt when planning the next
                                    attempt, so the same mistake is not
                                    repeated.

  2. Example-based reuse          – Successful past attempts are stored with
                                    their approach description. When a new
                                    task arrives, the closest successful
                                    attempt (measured by bag-of-words cosine
                                    similarity) is retrieved and its approach
                                    is reused as a starting point, rather
                                    than starting from scratch.

  3. Strategy selection           – Multiple strategies are tried across
                                    attempts and their quality scores are
                                    recorded. Before each new attempt the
                                    agent picks the strategy with the best
                                    observed mean score, trying any untried
                                    strategy first to ensure fair comparison.

The three mechanisms operate at different levels of abstraction:
feedback-based learning captures what went wrong in a specific attempt;
example-based reuse transfers a proven method to a similar task; strategy
selection tracks which broad approach works best across many tasks and
routes future attempts accordingly.
"""

from collections import Counter
from dataclasses import dataclass, field
import math

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
class Attempt:
    task: str
    output: str
    outcome: str
    feedback: str = ""
    approach: str = ""


@dataclass
class FeedbackLearner:
    lessons: list[str] = field(default_factory=list)

    def learn(self, attempt: Attempt) -> str:
        lesson = llm_call(
            (
                f"Task: {attempt.task}\n"
                f"Outcome: {attempt.outcome}\n"
                f"Output: {attempt.output}\n"
                f"Feedback: {attempt.feedback or 'none'}\n\n"
                "Return one concise lesson to improve the next attempt."
            ),
            system="You distill practical lessons from agent outcomes.",
        )
        self.lessons.append(lesson)
        return lesson

    def adapt_plan(self, task: str) -> str:
        context = "\n".join(f"- {lesson}" for lesson in self.lessons) or "- No lessons yet"
        return llm_call(
            f"Task: {task}\nPast lessons:\n{context}\n\nCreate an improved 3-step plan.",
            system="You create short, actionable plans that apply prior lessons.",
        )


def _tokens(text: str) -> Counter[str]:
    return Counter(word.lower() for word in text.split())


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    shared = left_tokens.keys() & right_tokens.keys()
    dot = sum(left_tokens[token] * right_tokens[token] for token in shared)
    norm_left = math.sqrt(sum(count * count for count in left_tokens.values()))
    norm_right = math.sqrt(sum(count * count for count in right_tokens.values()))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


@dataclass
class ExampleBasedLearner:
    """Reuse a successful task approach, not user-specific long-term memory."""

    attempts: list[Attempt] = field(default_factory=list)
    minimum_similarity: float = 0.1

    def recommend(self, task: str) -> Attempt | None:
        candidates = [
            (_similarity(task, attempt.task), attempt)
            for attempt in self.attempts
            if attempt.outcome == "success" and attempt.approach
        ]
        if not candidates:
            return None
        similarity, attempt = max(candidates, key=lambda candidate: candidate[0])
        return attempt if similarity >= self.minimum_similarity else None

    def adapt_plan(self, task: str) -> str:
        example = self.recommend(task)
        precedent = (
            f"Similar successful approach: {example.approach}"
            if example
            else "No sufficiently similar successful attempt is available."
        )
        return llm_call(
            f"Task: {task}\n{precedent}\n\nCreate an improved 3-step plan.",
            system="You adapt a plan by reusing relevant approaches from successful attempts.",
        )


@dataclass
class StrategySelector:
    """Select the strategy with the best observed mean score, trying each once."""

    scores: dict[str, list[float]] = field(default_factory=dict)

    def select(self, strategies: list[str]) -> str:
        if not strategies:
            raise ValueError("At least one strategy is required.")
        untried = [strategy for strategy in strategies if not self.scores.get(strategy)]
        if untried:
            return untried[0]
        return max(
            strategies,
            key=lambda strategy: sum(self.scores[strategy]) / len(self.scores[strategy]),
        )

    def record(self, strategy: str, score: float) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError("Strategy scores must be between 0 and 1.")
        self.scores.setdefault(strategy, []).append(score)


def run_feedback_learning_demo() -> None:
    learner = FeedbackLearner()
    failed = Attempt(
        task="Summarise customer feedback for product decisions",
        output="A generic summary with no themes or priorities.",
        outcome="failure",
        feedback="Needs clear themes, evidence, and prioritized actions.",
    )
    lesson = learner.learn(failed)

    print("=== Strategy 1: Learn a rule from feedback ===")
    print(f"Learned lesson: {lesson}")
    print("\nAdapted plan:")
    print(learner.adapt_plan("Summarise new feedback batch for weekly planning"))


def run_example_reuse_demo() -> None:
    learner = ExampleBasedLearner(
        attempts=[
            Attempt(
                task="Summarize customer feedback for product decisions",
                output="Themes with evidence and priorities.",
                outcome="success",
                approach=(
                    "Group comments into themes, cite representative quotes, "
                    "and rank actions by frequency and customer impact."
                ),
            ),
            Attempt(
                task="Prepare a weekly engineering release",
                output="Release checklist.",
                outcome="success",
                approach="Verify tests, document changes, and coordinate deployment.",
            ),
        ]
    )
    task = "Summarise support feedback and prioritize product improvements"
    example = learner.recommend(task)

    print("\n=== Strategy 2: Reuse a similar successful approach ===")
    if example:
        print(f"Retrieved approach: {example.approach}")
    print("\nAdapted plan:")
    print(learner.adapt_plan(task))


def run_strategy_selection_demo() -> None:
    strategies = ["feedback_rules", "successful_examples"]
    measured_outcomes = [
        ("feedback_rules", 0.55),
        ("successful_examples", 0.82),
        ("feedback_rules", 0.68),
        ("successful_examples", 0.76),
    ]
    selector = StrategySelector()

    print("\n=== Strategy 3: Choose using measured outcomes ===")
    print("Illustrative evaluation history:")
    for strategy, score in measured_outcomes:
        selector.record(strategy, score)
        print(f"- {strategy}: quality score {score:.2f}")

    selected = selector.select(strategies)
    print(f"Selected for the next task: {selected}")


def run_demo() -> None:
    run_feedback_learning_demo()
    run_example_reuse_demo()
    run_strategy_selection_demo()


if __name__ == "__main__":
    run_demo()
