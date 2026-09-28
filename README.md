# Agentic AI Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com)

Practical Python examples of agentic AI patterns, from core LLM workflows to advanced memory, adaptation, and protocol designs. The LLM-backed examples use Ollama's OpenAI-compatible API and the `llama3.2` model by default.

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
| [Learning and Adaptation](02-advanced-patterns/09-learning-and-adaptation/main.py) | Records outcomes, extracts lessons, and uses them to adapt later plans. |
| [Model Context Protocol (MCP)](02-advanced-patterns/10-model-context-protocol/main.py) | Implements a small protocol-style client/server interface for tools, resources, and prompts. |
| [Goal Setting and Monitoring](02-advanced-patterns/11-goal-setting-and-monitoring/main.py) | Tracks measurable milestones and flags goals that need intervention. |

## Setup

### 1. Install Ollama and download a model

Install [Ollama](https://ollama.com/download) for your operating system, then download the default model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before launching an LLM-backed example. On macOS with Homebrew, you can manage it as a background service:

```bash
brew services start ollama
brew services info ollama
brew services stop ollama
```

The examples connect to `http://localhost:11434/v1` and use `llama3.2`. To use another model available in Ollama, change the model value in the selected example.

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

Install the requirements file from the pattern's folder when running a different example. The MCP and goal-setting demos are standalone and do not make LLM calls.

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
- Ollama running locally for the LLM-backed examples
- The `llama3.2` model, or another model configured in the example
