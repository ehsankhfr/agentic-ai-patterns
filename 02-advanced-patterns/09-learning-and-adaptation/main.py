"""
Learning and Adaptation Pattern

Demonstrates how an agent can improve over repeated attempts by:
1) recording outcomes,
2) extracting reusable lessons,
3) adapting the next plan using those lessons.
"""

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
class Attempt:
    task: str
    output: str
    outcome: str
    feedback: str = ""


@dataclass
class AdaptiveMemory:
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


def run_demo() -> None:
    memory = AdaptiveMemory()

    failed = Attempt(
        task="Summarise customer feedback for product decisions",
        output="A generic summary with no themes or priorities.",
        outcome="failure",
        feedback="Needs clear themes, evidence, and prioritized actions.",
    )
    lesson = memory.learn(failed)

    print("=== Learning and Adaptation ===")
    print("Learned lesson:")
    print(f"- {lesson}\n")

    improved_plan = memory.adapt_plan("Summarise new feedback batch for weekly planning")
    print("Adapted plan:")
    print(improved_plan)


if __name__ == "__main__":
    run_demo()
