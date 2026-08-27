"""
Planning Pattern with Ollama (OpenAI-compatible API)

Planning separates *what to do* from *how to do it*. A planner LLM first
produces a structured sequence of steps for a goal, and then an executor
carries out each step in order, feeding results forward through the plan.

This makes the agent's reasoning transparent and auditable, and lets you
add replanning logic when a step produces unexpected results.

Two strategies are demonstrated:

  1. Static plan-and-execute  – The planner creates a full plan upfront as a
                                JSON list of steps; the executor runs each step
                                in sequence, passing prior results as context.

  2. Dynamic replanning       – After each step the replanner inspects the
                                result and either continues with the original
                                plan or revises the remaining steps before
                                proceeding. Handles surprises gracefully.
"""

import json
import re
from dataclasses import dataclass, field
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def llm_call(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Single LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def extract_json(text: str) -> list[dict]:
    """Extract the first JSON array from a string, tolerating markdown fences."""
    # Strip ```json ... ``` fences if present
    fenced = re.search(r"```(?:json)?\s*(\[[^\]]*(?:\][^\]]*)*])\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text

    # Find the outermost JSON array
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in response:\n{text}")
    candidate = raw[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Local models often emit trailing commas before a closing ] or };
        # strip those and retry before giving up.
        repaired = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSON array from response:\n{text}"
            ) from exc


@dataclass
class Step:
    index: int
    description: str
    action: str
    result: str = ""
    completed: bool = False


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)

    def pending(self) -> list[Step]:
        return [s for s in self.steps if not s.completed]

    def completed_summary(self) -> str:
        lines = []
        for s in self.steps:
            if s.completed:
                lines.append(f"Step {s.index}: {s.description}\nResult: {s.result}")
        return "\n\n".join(lines) if lines else "No steps completed yet."


# ---------------------------------------------------------------------------
# Strategy 1: Static plan-and-execute
# The planner creates the full plan once; the executor runs every step.
# ---------------------------------------------------------------------------

def create_plan(goal: str) -> Plan:
    """Ask the LLM to produce a structured plan for the given goal."""
    system = (
        "You are an expert planner. When given a goal, break it down into a "
        "clear sequence of concrete steps. "
        "Respond ONLY with a valid JSON array where every element has exactly "
        "these keys: \"step\" (int), \"description\" (str), \"action\" (str). "
        "Do not include any text outside the JSON array."
    )
    prompt = (
        f"Create a step-by-step plan to accomplish the following goal:\n\n"
        f"Goal: {goal}\n\n"
        "Return a JSON array of steps. Example format:\n"
        '[{"step": 1, "description": "...", "action": "..."}, ...]'
    )
    response = llm_call(prompt, system)
    raw_steps = extract_json(response)

    steps = [
        Step(
            index=item.get("step", i + 1),
            description=item.get("description", ""),
            action=item.get("action", ""),
        )
        for i, item in enumerate(raw_steps)
    ]
    return Plan(goal=goal, steps=steps)


def execute_step(step: Step, plan: Plan) -> str:
    """Execute a single step using prior results as context."""
    system = (
        "You are a precise, focused executor. You will be given a single step "
        "to perform as part of a larger plan. Carry out only that step, "
        "provide a concrete and concise result, and stop."
    )
    prior = plan.completed_summary()
    prompt = (
        f"Overall goal: {plan.goal}\n\n"
        f"Progress so far:\n{prior}\n\n"
        f"Current step {step.index}: {step.description}\n"
        f"Action to take: {step.action}\n\n"
        "Provide the result of this step only:"
    )
    return llm_call(prompt, system)


def static_plan_and_execute(goal: str) -> str:
    """
    Create a full plan upfront and execute each step in order.

    Args:
        goal: The high-level objective to accomplish.

    Returns:
        A final summary produced after all steps complete.
    """
    print(f"Goal: {goal}\n")

    plan = create_plan(goal)
    print(f"Plan ({len(plan.steps)} steps):")
    for s in plan.steps:
        print(f"  {s.index}. {s.description}")
    print()

    for step in plan.steps:
        print(f"--- Executing step {step.index}: {step.description} ---")
        step.result = execute_step(step, plan)
        step.completed = True
        print(f"Result: {step.result}\n")

    # Final synthesis
    system = "You are a concise summariser. Synthesise the completed steps into a final answer."
    prompt = (
        f"Goal: {plan.goal}\n\n"
        f"Completed steps and results:\n{plan.completed_summary()}\n\n"
        "Provide a concise final answer or summary:"
    )
    return llm_call(prompt, system)


# ---------------------------------------------------------------------------
# Strategy 2: Dynamic replanning
# After each step, a replanner may revise the remaining steps based on the
# actual result before proceeding.
# ---------------------------------------------------------------------------

def replan(plan: Plan, last_step: Step) -> list[Step]:
    """
    Inspect the last result and return a (possibly revised) list of remaining
    steps. Returns the existing pending steps unchanged if no revision needed.
    """
    pending = plan.pending()
    if not pending:
        return []

    system = (
        "You are an adaptive planner. Review the goal, what has been done, "
        "and the latest result. Decide whether the remaining steps are still "
        "appropriate, or whether they need to be updated.\n"
        "Respond with a JSON array of the remaining steps to execute "
        "(same format as before). If no changes are needed, return the "
        "original remaining steps unchanged."
    )
    remaining_json = json.dumps(
        [{"step": s.index, "description": s.description, "action": s.action} for s in pending]
    )
    prompt = (
        f"Goal: {plan.goal}\n\n"
        f"Progress so far:\n{plan.completed_summary()}\n\n"
        f"Latest step ({last_step.index}): {last_step.description}\n"
        f"Latest result: {last_step.result}\n\n"
        f"Current remaining steps:\n{remaining_json}\n\n"
        "Return the updated (or unchanged) remaining steps as a JSON array:"
    )
    response = llm_call(prompt, system)
    try:
        raw = extract_json(response)
    except ValueError:
        return pending  # fall back to original remaining steps

    return [
        Step(
            index=item.get("step", i + 1),
            description=item.get("description", ""),
            action=item.get("action", ""),
        )
        for i, item in enumerate(raw)
    ]


def dynamic_plan_and_execute(goal: str, max_steps: int = 10) -> str:
    """
    Create an initial plan, then replan after each step if needed.

    Args:
        goal:      The high-level objective to accomplish.
        max_steps: Safety cap on total steps executed.

    Returns:
        A final summary produced after all steps complete or the cap is hit.
    """
    print(f"Goal: {goal}\n")

    plan = create_plan(goal)
    print(f"Initial plan ({len(plan.steps)} steps):")
    for s in plan.steps:
        print(f"  {s.index}. {s.description}")
    print()

    executed = 0
    while plan.pending() and executed < max_steps:
        step = plan.pending()[0]
        print(f"--- Executing step {step.index}: {step.description} ---")
        step.result = execute_step(step, plan)
        step.completed = True
        executed += 1
        print(f"Result: {step.result}\n")

        # Replan with the updated information
        revised = replan(plan, step)
        if revised:
            # Replace pending steps with the (possibly updated) list
            completed = [s for s in plan.steps if s.completed]
            plan.steps = completed + revised
            if len(revised) != len(plan.pending()):
                print(f"  [replanned] Remaining steps updated to {len(revised)} step(s).\n")

    # Final synthesis
    system = "You are a concise summariser. Synthesise the completed steps into a final answer."
    prompt = (
        f"Goal: {plan.goal}\n\n"
        f"Completed steps and results:\n{plan.completed_summary()}\n\n"
        "Provide a concise final answer or summary:"
    )
    return llm_call(prompt, system)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY 1: Static plan-and-execute")
    print("=" * 60)
    print()

    result = static_plan_and_execute(
        "Write a short competitive analysis of Python vs JavaScript for "
        "building backend web services."
    )
    print("=== Final Answer ===")
    print(result)

    print()
    print("=" * 60)
    print("STRATEGY 2: Dynamic replanning")
    print("=" * 60)
    print()

    result = dynamic_plan_and_execute(
        "Explain how transformers work in machine learning, then describe "
        "three real-world applications and evaluate their impact."
    )
    print("=== Final Answer ===")
    print(result)
