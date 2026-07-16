"""
Reflection Pattern with OpenAI

Reflection lets an LLM critique and improve its own output over one or more
iterations. Instead of accepting the first response, the model (or a second
"critic" model) evaluates the draft, identifies weaknesses, and the generator
then produces a refined version.

Two reflection strategies are demonstrated:

  1. Self-reflection  – A single model generates a draft, critiques it, then
                        revises based on its own feedback (simple, low-overhead).

  2. Two-agent loop   – A dedicated "critic" agent and a "generator" agent take
                        turns: the generator produces output, the critic provides
                        structured feedback, and the generator revises until the
                        critic is satisfied or a max-iteration limit is reached.
"""

import os
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def llm_call(prompt: str, system: str = "", model: str = "llama3.2") -> str:
    """Single LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Strategy 1: Self-reflection
# The same model generates a draft, critiques it, then rewrites.
# ---------------------------------------------------------------------------

def self_reflect_and_revise(task: str, iterations: int = 1) -> str:
    """
    Generate a response, have the model critique it, then revise.

    Args:
        task:       The original task or question.
        iterations: Number of reflect-and-revise cycles to perform.

    Returns:
        The final revised response.
    """
    # Step 1: Initial generation
    generator_system = (
        "You are a knowledgeable assistant. Provide clear, accurate responses."
    )
    draft = llm_call(task, generator_system)
    print("=== Initial Draft ===")
    print(draft)
    print()

    for i in range(iterations):
        # Step 2: Self-critique
        critic_system = (
            "You are a critical reviewer. Evaluate the following response for "
            "accuracy, clarity, completeness, and logical consistency. "
            "List specific weaknesses and suggest concrete improvements. "
            "Be concise and direct."
        )
        critique_prompt = (
            f"Original task:\n{task}\n\n"
            f"Response to critique:\n{draft}\n\n"
            "Provide your critique:"
        )
        critique = llm_call(critique_prompt, critic_system)
        print(f"=== Critique (iteration {i + 1}) ===")
        print(critique)
        print()

        # Step 3: Revision
        revise_system = (
            "You are a knowledgeable assistant. Revise your response based on "
            "the critique provided. Address every point raised."
        )
        revise_prompt = (
            f"Original task:\n{task}\n\n"
            f"Your previous response:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            "Write an improved response that addresses the critique:"
        )
        draft = llm_call(revise_prompt, revise_system)
        print(f"=== Revised Response (iteration {i + 1}) ===")
        print(draft)
        print()

    return draft


# ---------------------------------------------------------------------------
# Strategy 2: Two-agent reflection loop
# A dedicated critic and generator take turns until the critic approves or
# the iteration limit is reached.
# ---------------------------------------------------------------------------

APPROVAL_SIGNAL = "APPROVED"


def critic_evaluate(task: str, response: str) -> str:
    """
    Evaluate a response and return either APPROVED or structured feedback.

    Returns APPROVED (exact string) when the response meets the quality bar,
    otherwise returns specific feedback the generator should act on.
    """
    system = (
        "You are a strict quality reviewer. "
        "Evaluate whether the response fully and accurately addresses the task. "
        f"If it does, respond with exactly one word: {APPROVAL_SIGNAL}. "
        "If not, respond with a concise numbered list of improvements needed "
        "(no preamble, no approval signal)."
    )
    prompt = f"Task:\n{task}\n\nResponse:\n{response}"
    return llm_call(prompt, system)


def generator_revise(task: str, response: str, feedback: str) -> str:
    """Revise a response given structured critic feedback."""
    system = (
        "You are a skilled writer and analyst. "
        "Revise your response to fully address the feedback provided."
    )
    prompt = (
        f"Task:\n{task}\n\n"
        f"Previous response:\n{response}\n\n"
        f"Reviewer feedback:\n{feedback}\n\n"
        "Write the improved response:"
    )
    return llm_call(prompt, system)


def two_agent_reflection_loop(task: str, max_iterations: int = 3) -> str:
    """
    Run a critic–generator loop until the critic approves or max_iterations
    is reached.

    Args:
        task:           The original task or question.
        max_iterations: Maximum number of revision cycles.

    Returns:
        The final (approved or best-effort) response.
    """
    # Initial generation
    generator_system = (
        "You are a knowledgeable assistant. Provide clear, accurate responses."
    )
    response = llm_call(task, generator_system)
    print("=== Initial Response ===")
    print(response)
    print()

    for i in range(max_iterations):
        feedback = critic_evaluate(task, response)

        if APPROVAL_SIGNAL in feedback:
            print(f"=== Critic approved on iteration {i + 1} ===")
            break

        print(f"=== Critic Feedback (iteration {i + 1}) ===")
        print(feedback)
        print()

        response = generator_revise(task, response, feedback)
        print(f"=== Revised Response (iteration {i + 1}) ===")
        print(response)
        print()
    else:
        print(f"=== Max iterations ({max_iterations}) reached ===")

    return response


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Strategy 1: Self-reflection ---
    print("=" * 60)
    print("STRATEGY 1: SELF-REFLECTION")
    print("=" * 60)
    print()

    task_1 = (
        "Explain the CAP theorem in distributed systems and give "
        "a real-world example of a system that chooses CP and one that chooses AP."
    )

    final_1 = self_reflect_and_revise(task_1, iterations=2)
    print("=== Final Answer (Self-reflection) ===")
    print(final_1)
    print()

    # --- Strategy 2: Two-agent reflection loop ---
    print("=" * 60)
    print("STRATEGY 2: TWO-AGENT REFLECTION LOOP")
    print("=" * 60)
    print()

    task_2 = (
        "Write a Python function that checks whether a string is a valid "
        "IPv4 address without using the `ipaddress` module. "
        "Include edge cases in the docstring."
    )

    final_2 = two_agent_reflection_loop(task_2, max_iterations=3)
    print("=== Final Answer (Two-agent loop) ===")
    print(final_2)
