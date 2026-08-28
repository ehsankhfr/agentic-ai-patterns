"""
Parallelisation Pattern with OpenAI

Parallelisation runs multiple LLM calls concurrently rather than sequentially.
This cuts total latency when sub-tasks are independent of each other.

Two parallelisation strategies are demonstrated:

  1. Sectioning   – Split a large task into independent chunks processed in
                    parallel (e.g., summarise each chapter of a book at once).

  2. Voting       – Run the same prompt multiple times in parallel and
                    aggregate the results for higher reliability / consensus
                    (e.g., N independent evaluations of a piece of text).

  3. Map-Reduce   – Apply the same transformation to many independent inputs
                    in parallel (Map), then send all intermediate results to a
                    final LLM call that combines them into a single answer
                    (Reduce).
                    Example: extract key facts from many documents at once,
                    then synthesise those facts into one research summary.
"""

import asyncio
import os
import time
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv())

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


async def llm_call_async(
    prompt: str,
    system: str = "",
    model: str = "llama3.2",
) -> str:
    """Async LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Strategy 1: Sectioning — parallel independent chunks
# ---------------------------------------------------------------------------

async def summarise_section(section: str, index: int) -> dict:
    """Summarise a single section of text."""
    system = "You are a concise summariser. Produce a 1-2 sentence summary."
    prompt = f"Summarise the following text:\n\n{section}"
    summary = await llm_call_async(prompt, system)
    return {"index": index, "summary": summary}


async def parallel_summarise(sections: list[str]) -> list[str]:
    """Summarise all sections in parallel, then return in original order."""
    tasks = [summarise_section(section, i) for i, section in enumerate(sections)]
    results = await asyncio.gather(*tasks)
    # Sort by original index to preserve order
    results.sort(key=lambda r: r["index"])
    return [r["summary"] for r in results]


async def parallel_summarise_demo() -> None:
    """Demo: summarise multiple independent text passages simultaneously."""
    passages = [
        (
            "The Renaissance was a period of cultural and intellectual flourishing in Europe "
            "from the 14th to 17th centuries. It saw the revival of classical Greek and Roman "
            "ideas and produced masterpieces in art, literature, and science."
        ),
        (
            "Machine learning is a subset of artificial intelligence where systems learn "
            "patterns from data without being explicitly programmed. It powers applications "
            "like image recognition, recommendation engines, and language models."
        ),
        (
            "The Amazon rainforest covers over 5.5 million square kilometres and is home to "
            "an estimated 10% of all species on Earth. It plays a critical role in regulating "
            "the global climate by absorbing vast quantities of CO₂."
        ),
        (
            "Quantum computing leverages quantum mechanical phenomena such as superposition "
            "and entanglement to perform computations that would be intractable on classical "
            "computers. Applications include cryptography, drug discovery, and optimisation."
        ),
    ]

    print("--- Sectioning: parallel summarisation ---\n")
    start = time.perf_counter()
    summaries = await parallel_summarise(passages)
    elapsed = time.perf_counter() - start

    for i, summary in enumerate(summaries, 1):
        print(f"Section {i}: {summary}")
    print(f"\nAll {len(passages)} sections summarised in {elapsed:.2f}s (parallel)\n")


# ---------------------------------------------------------------------------
# Strategy 2: Voting — run the same prompt N times and aggregate
# ---------------------------------------------------------------------------

async def single_evaluation(text: str, criteria: str, run_id: int) -> dict:
    """Run one evaluation pass and return a score + reasoning."""
    system = (
        "You are a strict content evaluator. "
        "Evaluate the text against the given criteria and respond with a JSON object: "
        '{"score": <integer 1-10>, "reasoning": "<one sentence>"}. '
        "Return only valid JSON, no markdown."
    )
    prompt = f"Criteria: {criteria}\n\nText to evaluate:\n{text}"
    import json
    raw = await llm_call_async(prompt, system)
    try:
        result = json.loads(raw)
        score = int(result.get("score", 5))
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        score = 5
        reasoning = "parse error"
    return {"run_id": run_id, "score": score, "reasoning": reasoning}


async def voting_evaluate(text: str, criteria: str, n: int = 3) -> dict:
    """Run N parallel evaluations and aggregate by averaging scores."""
    tasks = [single_evaluation(text, criteria, i) for i in range(n)]
    results = await asyncio.gather(*tasks)

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)

    print(f"Individual scores: {scores}")
    print(f"Average score    : {avg_score:.1f}/10")
    for r in results:
        print(f"  Run {r['run_id']}: {r['score']}/10 — {r['reasoning']}")

    return {"average_score": avg_score, "individual_results": results}


async def voting_demo() -> None:
    """Demo: evaluate the same essay with 3 parallel calls for robust scoring."""
    essay = (
        "Climate change represents the defining challenge of our era. Rising temperatures, "
        "extreme weather events, and melting ice caps are not distant threats but present "
        "realities affecting millions. Immediate, coordinated global action through renewable "
        "energy adoption, reforestation, and carbon pricing is essential to prevent "
        "catastrophic outcomes for future generations."
    )
    criteria = "Clarity, logical structure, use of evidence, and persuasiveness"

    print("--- Voting: 3 parallel evaluations of the same essay ---\n")
    print(f"Essay (truncated): {essay[:100]}...\n")
    start = time.perf_counter()
    result = await voting_evaluate(essay, criteria, n=3)
    elapsed = time.perf_counter() - start

    avg_score = result["average_score"]

    if avg_score >= 8:
        verdict = "Strong submission"
    elif avg_score >= 6:
        verdict = "Acceptable but needs improvement"
    else:
        verdict = "Needs major revision"

    print(f"\nFinal verdict: {verdict}")
    print(f"\nAggregated in {elapsed:.2f}s (parallel)\n")


# ---------------------------------------------------------------------------
# Strategy 3: Map-Reduce — parallel map then sequential reduce
# ---------------------------------------------------------------------------

async def extract_key_facts(passage: str, topic: str) -> str:
    """Map step: extract key facts relevant to a topic from a passage."""
    system = "You are a research assistant. Extract only the most relevant facts as bullet points."
    prompt = f"From the text below, extract key facts about '{topic}':\n\n{passage}"
    return await llm_call_async(prompt, system)


async def synthesise_facts(facts_list: list[str], topic: str) -> str:
    """Reduce step: synthesise all extracted facts into a coherent summary."""
    system = "You are a research analyst. Synthesise the provided facts into a clear, coherent paragraph."
    combined = "\n\n---\n\n".join(facts_list)
    prompt = f"Synthesise these extracted facts about '{topic}' into a single coherent summary:\n\n{combined}"
    return await llm_call_async(prompt, system)


async def map_reduce_demo() -> None:
    """Demo: extract facts from multiple documents in parallel, then reduce."""
    topic = "renewable energy"
    documents = [
        (
            "Solar panel efficiency has improved dramatically over the past decade, "
            "with modern panels achieving over 22% efficiency. The cost of solar "
            "power has dropped by 90% since 2010, making it the cheapest electricity "
            "source in history in many regions."
        ),
        (
            "Wind energy capacity has tripled globally over the last ten years. "
            "Offshore wind farms generate more consistent power than onshore due to "
            "stronger and more reliable winds. The UK leads in offshore wind capacity."
        ),
        (
            "Battery storage technology is key to enabling renewable energy at scale. "
            "Lithium-ion battery costs have fallen by 97% since 1991. Grid-scale "
            "storage allows excess solar and wind power to be stored and dispatched "
            "when generation is low."
        ),
    ]

    print("--- Map-Reduce: parallel extraction then synthesis ---\n")
    start = time.perf_counter()

    # Map: parallel fact extraction
    map_tasks = [extract_key_facts(doc, topic) for doc in documents]
    facts_list = await asyncio.gather(*map_tasks)

    print("Extracted facts per document:")
    for i, facts in enumerate(facts_list, 1):
        print(f"\n  Document {i}:\n{facts}")

    # Reduce: single synthesis call
    print("\nSynthesising...")
    synthesis = await synthesise_facts(facts_list, topic)
    elapsed = time.perf_counter() - start

    print(f"\nFinal synthesis:\n{synthesis}")
    print(f"\nCompleted map-reduce in {elapsed:.2f}s\n")


async def main() -> None:
    print("=" * 60)
    print("PARALLELISATION PATTERN DEMO")
    print("=" * 60)
    print()

    await parallel_summarise_demo()
    print("-" * 60)
    print()

    await voting_demo()
    print("-" * 60)
    print()

    await map_reduce_demo()


if __name__ == "__main__":
    asyncio.run(main())
