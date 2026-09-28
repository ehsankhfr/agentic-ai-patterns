"""
Tool Use Pattern with OpenAI

Tool use (function calling) lets the LLM decide when to invoke external
functions and how to interpret their results. Instead of hallucinating facts,
the model can call real tools – APIs, calculators, databases – and weave the
results into its final answer.

Two strategies are demonstrated:

  1. Single-turn tool call  – The model calls one tool, gets the result, and
                              returns a final response.  Good for simple,
                              well-scoped tasks.

  2. Agentic tool loop      – The model may call multiple tools across multiple
                              turns until it can answer the user.  The loop runs
                              until the model stops emitting tool-call requests
                              or a max-step limit is hit.

Simulated tools used here (no external APIs needed):
  - get_weather(city)              – returns fake weather data
  - calculate(expression)          – safely evaluates a maths expression
  - search_knowledge_base(query)   – returns canned facts for demo purposes
"""

import ast
import json
import math
from typing import Any
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """Return simulated weather data for a city."""
    fake_data = {
        "london":    {"city": "London",    "temperature_c": 14, "condition": "Cloudy",  "humidity_pct": 78},
        "tokyo":     {"city": "Tokyo",     "temperature_c": 28, "condition": "Sunny",   "humidity_pct": 55},
        "new york":  {"city": "New York",  "temperature_c": 22, "condition": "Partly cloudy", "humidity_pct": 60},
        "sydney":    {"city": "Sydney",    "temperature_c": 19, "condition": "Rainy",   "humidity_pct": 85},
        "paris":     {"city": "Paris",     "temperature_c": 17, "condition": "Overcast","humidity_pct": 72},
    }
    key = city.lower().strip()
    return fake_data.get(key, {"city": city, "error": "City not found in database"})


# Allowed AST node types for the safe math evaluator
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.UAdd, ast.USub,
)
_MATH_FUNCS: dict[str, Any] = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
_BINOPS: dict[type, Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def _validate_ast(tree: ast.Expression) -> None:
    """Raise ValueError if the AST contains any disallowed nodes or calls."""
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCS:
                raise ValueError(f"Disallowed function call: {ast.unparse(node.func)}")


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate a validated AST node to a numeric result."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call):
        fn = _MATH_FUNCS[node.func.id]  # type: ignore[attr-defined]
        return fn(*[_eval_node(a) for a in node.args])
    raise ValueError(f"Unsupported node: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    """Parse and evaluate a math expression using AST without calling eval()."""
    tree = ast.parse(expression, mode="eval")
    _validate_ast(tree)
    return _eval_node(tree.body)


def calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression and return the result."""
    try:
        result = _safe_eval(expression)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


def search_knowledge_base(query: str) -> dict:
    """Return a canned fact relevant to the query keyword (demo only)."""
    facts = {
        "python":       "Python was created by Guido van Rossum and first released in 1991.",
        "openai":       "OpenAI was founded in 2015 and is known for the GPT series of models.",
        "llm":          "Large Language Models (LLMs) are trained on massive text corpora using transformer architectures.",
        "agentic":      "Agentic AI systems can plan, use tools, and act autonomously to complete multi-step tasks.",
        "tool use":     "Tool use allows LLMs to call external functions, enabling access to real-time data and computation.",
        "transformer":  "The Transformer architecture, introduced in 2017, underpins modern LLMs and uses attention mechanisms.",
    }
    query_lower = query.lower()
    for keyword, fact in facts.items():
        if keyword in query_lower:
            return {"query": query, "result": fact}
    return {"query": query, "result": "No specific information found for that query."}


# ---------------------------------------------------------------------------
# Tool registry – maps tool names to callables and JSON schema definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city, e.g. 'London'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression, e.g. '2 ** 10' or 'sqrt(144)'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base for factual information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic or question to look up"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_CALLABLES: dict[str, Any] = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
}


def dispatch_tool(name: str, arguments: str) -> str:
    """Call the named tool with JSON-encoded arguments and return a JSON string result."""
    fn = TOOL_CALLABLES.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments)
        result = fn(**args)
    except Exception as exc:
        result = {"error": str(exc)}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Strategy 1: Single-turn tool call
# The model calls at most one round of tools, then gives a final answer.
# ---------------------------------------------------------------------------

def single_turn_tool_call(user_query: str) -> str:
    """
    Send a query to the model with tools available.
    If it calls a tool, execute it and send the result back for a final answer.

    Args:
        user_query: The user's question or task.

    Returns:
        The model's final text response.
    """
    messages = [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = response.choices[0].message

    # No tool call – return the direct answer
    if not msg.tool_calls:
        return msg.content.strip()

    # Execute every tool call the model requested
    messages.append(msg)  # assistant message with tool_calls
    for tc in msg.tool_calls:
        result = dispatch_tool(tc.function.name, tc.function.arguments)
        print(f"  [tool] {tc.function.name}({tc.function.arguments}) -> {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

    # Second LLM call to synthesise the tool results into a final answer
    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Strategy 2: Agentic tool loop
# The model may invoke tools multiple times until it can answer or hits the
# step limit.
# ---------------------------------------------------------------------------

def agentic_tool_loop(user_query: str, max_steps: int = 6) -> str:
    """
    Run an agentic loop: the model can call tools repeatedly until it either
    produces a final text response (no more tool calls) or max_steps is reached.

    Args:
        user_query: The user's question or task.
        max_steps:  Maximum number of tool-call rounds to allow.

    Returns:
        The model's final text response.
    """
    messages = [{"role": "user", "content": user_query}]
    step = 0

    while step < max_steps:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        # No tool calls → the model is done
        if not msg.tool_calls:
            return msg.content.strip()

        # Execute all tool calls in this round
        step += 1
        print(f"\n  --- Step {step} ---")
        for tc in msg.tool_calls:
            result = dispatch_tool(tc.function.name, tc.function.arguments)
            print(f"  [tool] {tc.function.name}({tc.function.arguments}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Fallback: ask the model to wrap up with what it knows
    messages.append({
        "role": "user",
        "content": "Please provide your best answer based on the information gathered so far.",
    })
    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY 1: Single-turn tool call")
    print("=" * 60)

    queries_single = [
        "What is the weather like in Tokyo right now?",
        "What is the square root of 1764?",
        "Tell me a fact about transformers in machine learning.",
    ]

    for q in queries_single:
        print(f"\nUser: {q}")
        answer = single_turn_tool_call(q)
        print(f"Assistant: {answer}")
        print()

    print("=" * 60)
    print("STRATEGY 2: Agentic tool loop (multi-tool, multi-step)")
    print("=" * 60)

    queries_agentic = [
        (
            "I'm planning a trip. Compare the weather in London and Sydney, "
            "then calculate the average temperature between the two cities."
        ),
        (
            "What is the area of a circle with radius 7? "
            "Also, find a fact about LLMs and summarise both results."
        ),
    ]

    for q in queries_agentic:
        print(f"\nUser: {q}")
        answer = agentic_tool_loop(q)
        print(f"\nAssistant: {answer}")
        print()
