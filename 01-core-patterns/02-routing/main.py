"""
Routing Pattern with OpenAI

Routing classifies an input and directs it to a specialised handler.
This keeps prompts focused: rather than one giant prompt that tries to handle
every case, a lightweight router decides which expert sub-prompt to invoke.

Common use-cases:
  - Customer support triage (billing, technical, general)
  - Multi-language query dispatch
  - Content moderation routing
  - Intent-based chatbot flows

Two routing strategies are shown here:
  1. LLM-based router  – the model classifies the intent in natural language
  2. Structured router – the model returns JSON so classification is unambiguous
"""

import json
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
# Strategy 1: LLM-based router
# A lightweight classifier decides the category, then the right handler runs.
# ---------------------------------------------------------------------------

SUPPORT_CATEGORIES = ["billing", "technical", "general", "refund"]


def route_support_query(query: str) -> str:
    """Classify a support query into one of the known categories."""
    system = (
        "You are a customer support classifier. "
        f"Classify the user's message into exactly one of: {', '.join(SUPPORT_CATEGORIES)}. "
        "Respond with only the category name in lowercase."
    )
    category = llm_call(query, system).lower().strip()
    # Normalise – if the model returns something unexpected, fall back to 'general'
    return category if category in SUPPORT_CATEGORIES else "general"


# Specialised handlers
def handle_billing(query: str) -> str:
    system = (
        "You are a billing support specialist. "
        "Help customers with payment issues, invoices, and subscription changes. "
        "Be empathetic and solution-focused."
    )
    return llm_call(query, system)


def handle_technical(query: str) -> str:
    system = (
        "You are a technical support engineer. "
        "Diagnose and resolve technical issues step-by-step. "
        "Ask clarifying questions when needed."
    )
    return llm_call(query, system)


def handle_refund(query: str) -> str:
    system = (
        "You are a refund specialist. "
        "Guide customers through the refund process following company policy: "
        "refunds are available within 30 days of purchase with a valid reason."
    )
    return llm_call(query, system)


def handle_general(query: str) -> str:
    system = (
        "You are a friendly general customer support agent. "
        "Answer questions clearly and helpfully."
    )
    return llm_call(query, system)


HANDLERS = {
    "billing": handle_billing,
    "technical": handle_technical,
    "refund": handle_refund,
    "general": handle_general,
}


def support_router(query: str) -> dict:
    """Full routing pipeline: classify → dispatch → return result."""
    category = route_support_query(query)
    handler = HANDLERS.get(category, handle_general)
    response = handler(query)
    return {"category": category, "response": response}


# ---------------------------------------------------------------------------
# Strategy 2: Structured router using JSON output
# Returns a structured decision, enabling richer metadata (confidence, reason).
# ---------------------------------------------------------------------------

CONTENT_TYPES = ["code", "creative_writing", "data_analysis", "factual_qa", "other"]


def structured_route(user_input: str) -> dict:
    """Use structured JSON output to route content to the right expert."""
    system = (
        "You are a query classifier. Analyse the user's input and return a JSON object with:\n"
        '  "content_type": one of ' + str(CONTENT_TYPES) + "\n"
        '  "confidence": float between 0 and 1\n'
        '  "reason": one sentence explanation\n'
        "Return only valid JSON, no markdown fences."
    )
    raw = llm_call(user_input, system)
    try:
        classification = json.loads(raw)
    except json.JSONDecodeError:
        classification = {"content_type": "other", "confidence": 0.0, "reason": "parse error"}

    content_type = classification.get("content_type", "other")
    if content_type not in CONTENT_TYPES:
        content_type = "other"

    # Dispatch to a specialised system prompt
    expert_systems = {
        "code": "You are an expert software engineer. Provide clean, well-commented code.",
        "creative_writing": "You are a creative writing assistant. Be imaginative and vivid.",
        "data_analysis": "You are a data analyst. Provide structured, quantitative insights.",
        "factual_qa": "You are a knowledgeable assistant. Provide accurate, concise facts.",
        "other": "You are a helpful assistant.",
    }
    expert_response = llm_call(user_input, expert_systems[content_type])

    return {
        "classification": classification,
        "content_type": content_type,
        "response": expert_response,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ROUTING PATTERN DEMO")
    print("=" * 60)

    # Demo 1: Support query router
    print("\n--- Demo 1: Customer Support Router (LLM classifier) ---\n")
    queries = [
        "My invoice shows a charge I don't recognise from last month.",
        "The app crashes whenever I try to upload a file larger than 10 MB.",
        "I bought the premium plan yesterday by mistake, can I get my money back?",
        "What are your business hours?",
    ]
    for q in queries:
        result = support_router(q)
        print(f"Query    : {q}")
        print(f"Category : {result['category']}")
        print(f"Response : {result['response'][:120]}...")
        print()

    print("-" * 60)

    # Demo 2: Structured content router
    print("\n--- Demo 2: Structured Content Router (JSON output) ---\n")
    inputs = [
        "Write a Python function to find all prime numbers up to n.",
        "Describe a sunset on a distant alien world.",
        "What was the global GDP in 2023?",
    ]
    for inp in inputs:
        result = structured_route(inp)
        cls = result["classification"]
        print(f"Input          : {inp}")
        print(f"Content type   : {result['content_type']}  (confidence: {cls.get('confidence', '?')})")
        print(f"Reason         : {cls.get('reason', '')}")
        print(f"Response       : {result['response'][:120]}...")
        print()
