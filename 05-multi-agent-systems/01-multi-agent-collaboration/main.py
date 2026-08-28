"""
Multi-Agent Collaboration Pattern with Ollama (OpenAI-compatible API)

Multi-agent collaboration assigns different parts of a task to agents with
complementary roles, rather than relying on one general-purpose prompt to
balance every perspective at once.

Six collaboration topologies are demonstrated:

  1. Sequential pipeline   – Each agent receives the previous agent's handoff
                             and extends it before passing it on.

  2. Supervisor-worker     – A supervisor delegates focused assignments to
                             workers and reconciles their results.

  3. Parallel council      – Independent agents propose solutions at the same
                             time; a council chair selects or combines them.

  4. Peer debate           – Opposing agents challenge each other's proposals
                             across rounds; a moderator decides the outcome.

  5. Hierarchical team     – Specialists report to domain leads, who report
                             to a coordinator (a two-level supervisor tree).

  6. Blackboard            – Agents contribute to a shared workspace that a
                             reviewer inspects before a synthesizer answers.
"""

import asyncio
from dataclasses import dataclass, field
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv(find_dotenv())

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
async_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Shared workspace primitives (used by Strategy 6: Blackboard)
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    name: str
    role: str
    output: str = ""


@dataclass
class SharedWorkspace:
    task: str
    contributions: list[Agent] = field(default_factory=list)
    review: str = ""

    def context(self) -> str:
        """Return the artifacts currently available to collaborating agents."""
        contributions = "\n\n".join(
            f"[{agent.name} - {agent.role}]\n{agent.output}"
            for agent in self.contributions
            if agent.output
        )
        review = f"\n\n[Reviewer]\n{self.review}" if self.review else ""
        return contributions + review or "No contributions yet."


# ---------------------------------------------------------------------------
# Shared helpers used by every strategy
# ---------------------------------------------------------------------------

def llm_call(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Single LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The model returned a response without text content.")
    return content.strip()


async def llm_call_async(
    prompt: str, system: str = "", model: str = MODEL
) -> str:
    """Async LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The model returned a response without text content.")
    return content.strip()


def ask_role(task: str, role: str, context: str = "") -> str:
    """Ask one role to work on a task with optional handoff context."""
    system = (
        f"You are a specialist agent responsible for {role}. Be concrete, "
        "state uncertainty, and produce work another agent can use."
    )
    prompt = f"Task:\n{task}\n\n"
    if context:
        prompt += f"Handoff from the team:\n{context}\n\n"
    prompt += f"Your assignment: {role}\n\nProvide your contribution."
    return llm_call(prompt, system)


async def ask_role_async(task: str, role: str, context: str = "") -> str:
    """Async counterpart to ask_role for independent collaboration tasks."""
    system = (
        f"You are a specialist agent responsible for {role}. Be concrete, "
        "state uncertainty, and produce work another agent can use."
    )
    prompt = f"Task:\n{task}\n\n"
    if context:
        prompt += f"Handoff from the team:\n{context}\n\n"
    prompt += f"Your assignment: {role}\n\nProvide your contribution."
    return await llm_call_async(prompt, system)


def synthesize(task: str, evidence: str, role: str = "lead synthesizer") -> str:
    """Combine agent artifacts into a final answer."""
    system = (
        f"You are the {role}. Produce a clear, accurate answer to the task. "
        "Resolve conflicts using evidence, and do not mention the internal "
        "agent workflow."
    )
    prompt = f"Task:\n{task}\n\nAgent artifacts:\n{evidence}\n\nWrite the final answer."
    return llm_call(prompt, system)


# ---------------------------------------------------------------------------
# Strategy 1: Sequential pipeline
# Each agent extends the previous agent's handoff before passing it on.
# ---------------------------------------------------------------------------

def sequential_collaboration(task: str) -> str:
    """
    Run a linear handoff pipeline where each agent builds on the last.

    Args:
        task: The question or objective for the team.

    Returns:
        A final answer synthesised from the pipeline's handoffs.
    """
    handoff = ""
    artifacts = []
    roles = [
        "research the relevant facts and constraints",
        "critique the previous contribution and identify tradeoffs",
        "turn the research and critique into a practical draft",
    ]
    for index, role in enumerate(roles, start=1):
        handoff = ask_role(task, role, handoff)
        artifacts.append(f"Agent {index}:\n{handoff}")
    return synthesize(task, "\n\n".join(artifacts), "pipeline editor")


# ---------------------------------------------------------------------------
# Strategy 2: Supervisor-worker
# A supervisor delegates focused assignments to workers and merges results.
# ---------------------------------------------------------------------------

def supervisor_worker_collaboration(task: str) -> str:
    """
    Delegate focused assignments to workers, then reconcile their results.

    Args:
        task: The question or objective for the team.

    Returns:
        A final answer synthesised from the workers' assignments.
    """
    assignments = [
        "analyze technical feasibility and implementation constraints",
        "analyze cost, operational risk, and failure modes",
        "analyze the user or business impact and success criteria",
    ]
    artifacts = []
    for assignment in assignments:
        output = ask_role(task, assignment)
        artifacts.append(f"Worker assignment: {assignment}\n{output}")

    evidence = "\n\n".join(artifacts)
    return synthesize(
        task,
        evidence,
        "supervisor. Check that every assignment is covered and reconcile conflicts",
    )


# ---------------------------------------------------------------------------
# Strategy 3: Parallel council
# Independent agents propose solutions at once; a chair picks or combines them.
# ---------------------------------------------------------------------------

async def _parallel_council_collaboration(task: str) -> str:
    """Collect independent council perspectives concurrently."""
    roles = [
        "solve the task independently using a conservative approach",
        "solve the task independently using an innovative approach",
        "solve the task independently as a risk-focused reviewer",
    ]
    artifacts = await asyncio.gather(
        *(ask_role_async(task, role) for role in roles)
    )
    evidence = "\n\n".join(
        f"Council member {index}:\n{output}"
        for index, output in enumerate(artifacts, start=1)
    )
    return synthesize(
        task,
        evidence,
        "council chair. Compare the independent proposals and select or combine the strongest ideas",
    )


def parallel_council_collaboration(task: str) -> str:
    """
    Collect independent perspectives concurrently, then let a council chair judge them.

    Args:
        task: The question or objective for the team.

    Returns:
        A final answer selected or combined from the independent proposals.
    """
    return asyncio.run(_parallel_council_collaboration(task))


# ---------------------------------------------------------------------------
# Strategy 4: Peer debate
# Opposing agents challenge each other's proposals; a moderator decides.
# ---------------------------------------------------------------------------

def debate_collaboration(task: str, rounds: int = 1) -> str:
    """
    Run an affirmative-vs-negative debate before a moderator decides.

    Args:
        task:   The question or objective for the team.
        rounds: Number of rebuttal rounds after the opening arguments.

    Returns:
        A final answer decided by the impartial moderator.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative")

    affirmative = ask_role(task, "argue for the strongest solution")
    negative = ask_role(task, "argue against likely solutions and propose safeguards")
    debate = f"Affirmative:\n{affirmative}\n\nNegative:\n{negative}"

    for round_number in range(rounds):
        affirmative = ask_role(
            task,
            f"defend your proposal against the other side in debate round {round_number + 1}",
            debate,
        )
        negative = ask_role(
            task,
            f"rebut the proposal and expose remaining weaknesses in debate round {round_number + 1}",
            debate,
        )
        debate = f"Affirmative:\n{affirmative}\n\nNegative:\n{negative}"

    return synthesize(task, debate, "impartial debate moderator")


# ---------------------------------------------------------------------------
# Strategy 5: Hierarchical team
# Specialists report to domain leads, who report to a coordinator.
# ---------------------------------------------------------------------------

def hierarchical_collaboration(task: str) -> str:
    """
    Run a two-level team: specialists report to leads, who report to a coordinator.

    Args:
        task: The question or objective for the team.

    Returns:
        A final answer synthesised from each lead's departmental report.
    """
    departments = {
        "Engineering lead": [
            "analyze architecture and technical constraints",
            "identify reliability and security concerns",
        ],
        "Product lead": [
            "analyze user needs and product tradeoffs",
            "define measurable success criteria and rollout risks",
        ],
    }
    lead_reports = []
    for lead, specialist_roles in departments.items():
        specialist_outputs = [ask_role(task, role) for role in specialist_roles]
        evidence = "\n\n".join(
            f"Specialist {index}:\n{output}"
            for index, output in enumerate(specialist_outputs, start=1)
        )
        lead_reports.append(f"{lead}:\n{synthesize(task, evidence, lead)}")

    return synthesize(task, "\n\n".join(lead_reports), "hierarchical team coordinator")


# ---------------------------------------------------------------------------
# Strategy 6: Blackboard
# Agents contribute to a shared workspace; a reviewer and synthesizer inspect it.
# ---------------------------------------------------------------------------

def run_agent(agent: Agent, workspace: SharedWorkspace) -> str:
    """Ask an agent to contribute while exposing the current shared context."""
    system = (
        f"You are {agent.name}, acting as {agent.role}. "
        "Work only within your role, make your reasoning concrete, and clearly "
        "label assumptions or uncertainty."
    )
    prompt = (
        f"Shared task:\n{workspace.task}\n\n"
        f"Existing contributions from other agents:\n{workspace.context()}\n\n"
        f"Your responsibility:\n{agent.role}\n\n"
        "Produce a concise contribution that the next agents can build on."
    )
    agent.output = llm_call(prompt, system)
    workspace.contributions.append(agent)
    return agent.output


def review_contributions(workspace: SharedWorkspace) -> str:
    """Have a reviewer identify conflicts, gaps, and the strongest evidence."""
    system = (
        "You are the reviewing agent in a multi-agent team. Compare the shared "
        "contributions against the task. Identify conflicts, unsupported claims, "
        "and important omissions. End with concrete priorities for the synthesizer."
    )
    prompt = (
        f"Task:\n{workspace.task}\n\n"
        f"Team workspace:\n{workspace.context()}\n\n"
        "Review the work and give actionable guidance to the synthesizer."
    )
    workspace.review = llm_call(prompt, system)
    return workspace.review


def collaborate(task: str) -> str:
    """
    Coordinate specialist agents, a reviewer, and a final synthesizer.

    Args:
        task: The question or objective for the team.

    Returns:
        A final answer grounded in the team's shared workspace.
    """
    workspace = SharedWorkspace(task=task)
    agents = [
        Agent("Researcher", "Gather the key facts, definitions, and examples."),
        Agent("Skeptic", "Look for risks, counterexamples, tradeoffs, and edge cases."),
        Agent("Practicalist", "Turn the task into concrete recommendations for the intended audience."),
    ]

    for agent in agents:
        print(f"=== {agent.name} ===")
        print(run_agent(agent, workspace))
        print()

    print("=== Reviewer ===")
    print(review_contributions(workspace))
    print()

    return synthesize(workspace.task, workspace.context())


if __name__ == "__main__":
    task = (
        "Recommend an approach for a small startup choosing between a managed "
        "PostgreSQL database and a document database for a multi-tenant SaaS "
        "product. Explain the tradeoffs and give a practical recommendation."
    )

    print("=" * 60)
    print("MULTI-AGENT COLLABORATION DEMO")
    print("=" * 60)
    print(f"Task: {task}\n")

    strategies = [
        ("STRATEGY 1: Sequential pipeline", sequential_collaboration),
        ("STRATEGY 2: Supervisor-worker", supervisor_worker_collaboration),
        ("STRATEGY 3: Parallel council", parallel_council_collaboration),
        ("STRATEGY 4: Peer debate", debate_collaboration),
        ("STRATEGY 5: Hierarchical team", hierarchical_collaboration),
        ("STRATEGY 6: Blackboard", collaborate),
    ]

    for title, strategy in strategies:
        print("=" * 60)
        print(title)
        print("=" * 60)
        print()
        final_answer = strategy(task)
        print("=== Final Answer ===")
        print(final_answer)
        print()
        print("-" * 60)
        print()
