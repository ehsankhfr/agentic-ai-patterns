# Agentic AI Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com)

Practical Python examples of agentic AI patterns, from core LLM workflows to advanced memory, adaptation, and protocol designs. Every pattern uses an LLM; the examples connect to Ollama's OpenAI-compatible API and use the `llama3.2` model by default.

## Patterns

### Core patterns

| Pattern | What it demonstrates |
| --- | --- |
| [Prompt Chaining](01-core-patterns/01-prompt-chaining/main.py) | Composes focused LLM calls into a pipeline, with optional validation gates. |
| [Routing](01-core-patterns/02-routing/main.py) | Classifies a request and dispatches it to a specialized handler, using natural-language or structured routing. |
| [Parallelization](01-core-patterns/03-parallelization/main.py) | Uses concurrent calls for sectioning, voting, and map-reduce workflows. |
| [Reflection](01-core-patterns/04-reflection/main.py) | Improves a draft through self-critique or a separate critic-generator loop. |
| [Tool Use](01-core-patterns/05-tool-use/main.py) | Demonstrates single-turn and multi-step tool calling with simulated weather, calculator, and knowledge-base tools. |
| [Planning](01-core-patterns/06-planning/main.py) | Creates and executes plans, with both static execution and dynamic replanning. |
| [Multi-Agent Collaboration](01-core-patterns/07-multi-agent/main.py) | Shows sequential, supervisor-worker, parallel council, debate, hierarchical, and blackboard team topologies. |

### Advanced patterns

| Pattern | What it demonstrates |
| --- | --- |
| [Memory Management](02-advanced-patterns/08-memory-management/main.py) | Manages conversation context, retrieves long-term memories, and learns procedural rules from experience. |
| [Learning and Adaptation](02-advanced-patterns/09-learning-and-adaptation/main.py) | Compares feedback-derived rules, reuse of similar successful approaches, and strategy selection based on measured outcomes. |
| [Model Context Protocol (MCP)](02-advanced-patterns/10-model-context-protocol/main.py) | Uses an LLM to select discovered tool, resource, and prompt capabilities across multiple in-process servers; it is not a wire-compatible MCP implementation. |
| [Goal Setting and Monitoring](02-advanced-patterns/11-goal-setting-and-monitoring/main.py) | Uses an LLM to recommend corrective actions for lagging milestones in a hierarchical goal, then demonstrates a follow-up review. |

## Setup

### 1. Install Ollama and download models

Install [Ollama](https://ollama.com/download) for your operating system, then download the default chat model:

```bash
ollama pull llama3.2
```

The **Memory Management** pattern (`08-memory-management`) also uses a dedicated embedding model for semantic recall. Pull it once before running that pattern:

```bash
ollama pull nomic-embed-text
```

`nomic-embed-text` is a small (~274 MB) model that converts text to dense vectors. It enables true semantic search so that a query like "dinner suggestions" can match a stored fact like "user is vegetarian" — something bag-of-words similarity cannot do. If the model is not present, the pattern falls back to keyword-based cosine similarity automatically, but recall quality will be lower.

Make sure Ollama is running before launching any example. On macOS with Homebrew, you can manage it as a background service:

```bash
brew services start ollama
brew services info ollama
brew services stop ollama
```

The examples connect to `http://localhost:11434/v1` and use `llama3.2` by default. To use another model available in Ollama, change the model value at the top of the selected example.

### 2. Create and activate a virtual environment

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install a pattern's dependencies

Each pattern has its own `requirements.txt`. For example:

```bash
pip install -r 01-core-patterns/01-prompt-chaining/requirements.txt
```

Install the requirements file from the pattern's folder when running a different example. All patterns make LLM calls and require Ollama to be running.

### 4. Run an example

Run a pattern from the repository root. For example:

```bash
python 01-core-patterns/01-prompt-chaining/main.py
python 01-core-patterns/03-parallelization/main.py
python 02-advanced-patterns/08-memory-management/main.py
python 02-advanced-patterns/10-model-context-protocol/main.py
```

Replace the path with the `main.py` path for any of the patterns listed above.

## Requirements

- Python 3.10 or later
- Ollama running locally for all patterns
- The `llama3.2` model (default), or another chat model configured in the example
- The `nomic-embed-text` model — required for semantic recall in the Memory Management pattern; optional for all others
