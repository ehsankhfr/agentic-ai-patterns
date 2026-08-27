# Agentic AI Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com)

A collection of practical agentic AI design patterns implemented with the OpenAI API (or any OpenAI-compatible local model via [Ollama](https://ollama.com)).

## Patterns

The patterns are grouped by the role they play in an agentic system.

### 1. Workflow Control

These patterns control how work flows between steps and agents.

#### [Prompt Chaining](01-workflow-control/01-prompt-chaining/main.py)

Decomposes a task into a sequence of steps where each LLM call processes the output of the previous one. Useful for structured pipelines like Research → Draft → Edit.

#### [Routing](01-workflow-control/02-routing/main.py)

Classifies an input and directs it to a specialised handler. Keeps prompts focused by using a lightweight router. Demonstrates two strategies:

- **LLM-based router** — the model classifies intent in natural language
- **Structured router** — the model returns JSON for unambiguous classification

#### [Parallelisation](01-workflow-control/03-parallelisation/main.py)

Runs multiple LLM calls concurrently to reduce latency when sub-tasks are independent. Demonstrates sectioning, voting, and map-reduce strategies.

#### [Planning](01-workflow-control/04-planning/main.py)

Separates what to do from how to do it: a planner creates a structured sequence of steps and an executor carries them out. Dynamic replanning handles unexpected results.

### 2. Quality and Reliability

These patterns improve outputs through critique, revision, consensus, and recovery.

#### [Reflection](02-quality-reliability/01-reflection/main.py)

Lets an LLM critique and iteratively improve its own output through self-reflection or a dedicated critic-generator loop.

Parallelisation's voting strategy and Planning's dynamic replanning also provide reliability mechanisms.

### 3. Tools and Memory

These patterns connect agents to external capabilities and persistent context.

#### [Tool Use](03-tools-and-memory/01-tool-use/main.py)

Lets the LLM invoke external functions such as APIs, calculators, and databases, either in a single turn or through an agentic multi-step tool loop.

#### [Memory Management](03-tools-and-memory/02-memory-management/main.py)

Provides short-term context management, long-term persistent memory, and learning from experience across sessions.

### 4. Multi-Agent Organization

#### [Multi-Agent Collaboration](04-multi-agent/01-multi-agent-collaboration/main.py)

Shows sequential pipelines, supervisor-worker teams, parallel councils, peer debates, hierarchical teams, and blackboard collaboration.

## Setup

All patterns are configured to use a local **llama3.2** model via Ollama. No API key required.

### 1. Install and start Ollama

```bash
brew install ollama
ollama pull llama3.2
```

**Start Ollama as a background service** (auto-restarts at login):

```bash
brew services start ollama    # start in background
brew services stop ollama     # stop
brew services restart ollama  # restart
brew services info ollama     # check status
```

**List installed models and pull a specific version:**

```bash
ollama list                   # show locally installed models
ollama pull llama3.2          # default (3b)
ollama pull llama3.2:1b       # smaller/faster variant
ollama pull llama3.2:3b       # explicit 3b variant
ollama pull llama3.1:8b       # larger, more capable
ollama pull mistral           # recommended for multilingual / translation tasks
```

To use a different model, update the `model` parameter in the `llm_call` function of any pattern, or pass it at the call site.

### 2. Install dependencies

Each pattern has its own `requirements.txt`:

```bash
pip install -r 01-workflow-control/01-prompt-chaining/requirements.txt
pip install -r 01-workflow-control/02-routing/requirements.txt
pip install -r 01-workflow-control/03-parallelisation/requirements.txt
pip install -r 01-workflow-control/04-planning/requirements.txt
pip install -r 02-quality-reliability/01-reflection/requirements.txt
pip install -r 03-tools-and-memory/01-tool-use/requirements.txt
pip install -r 03-tools-and-memory/02-memory-management/requirements.txt
pip install -r 04-multi-agent/01-multi-agent-collaboration/requirements.txt
```

### 3. Run a pattern

```bash
python 01-workflow-control/01-prompt-chaining/main.py
python 01-workflow-control/02-routing/main.py
python 01-workflow-control/03-parallelisation/main.py
python 01-workflow-control/04-planning/main.py
python 02-quality-reliability/01-reflection/main.py
python 03-tools-and-memory/01-tool-use/main.py
python 03-tools-and-memory/02-memory-management/main.py
python 04-multi-agent/01-multi-agent-collaboration/main.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running as a background service (`brew services start ollama`) with `llama3.2` pulled
