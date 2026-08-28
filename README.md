# Agentic AI Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com)

A collection of practical agentic AI design patterns implemented with the OpenAI API (or any OpenAI-compatible local model via [Ollama](https://ollama.com)).

## Patterns

The patterns are grouped by the capability they add to an agentic system:
**reason → control → act → remember → collaborate**.

### 1. Reasoning and Task Decomposition

These patterns help agents break down tasks, plan solutions, and improve their
own reasoning.

#### [Prompt Chaining](01-reasoning-task-decomposition/01-prompt-chaining/main.py)

Decomposes a task into a sequence of steps where each LLM call processes the output of the previous one. Useful for structured pipelines like Research → Draft → Edit.

#### [Planning](01-reasoning-task-decomposition/02-planning/main.py)

Separates what to do from how to do it: a planner creates a structured sequence of steps and an executor carries them out. Dynamic replanning handles unexpected results.

#### [Reflection](01-reasoning-task-decomposition/03-reflection/main.py)

Lets an LLM critique and iteratively improve its own output through self-reflection or a dedicated critic-generator loop.

### 2. Control Flow and Coordination

These patterns decide how work is routed, split, and coordinated across steps.

#### [Routing](02-control-flow-coordination/01-routing/main.py)

Classifies an input and directs it to a specialised handler. Keeps prompts focused by using a lightweight router. Demonstrates two strategies:

- **LLM-based router** — the model classifies intent in natural language
- **Structured router** — the model returns JSON for unambiguous classification

#### [Parallelisation](02-control-flow-coordination/02-parallelisation/main.py)

Runs multiple LLM calls concurrently to reduce latency when sub-tasks are independent. Demonstrates sectioning, voting, and map-reduce strategies.

### 3. Tools and Environment

This pattern connects an agent to capabilities outside the language model.

#### [Tool Use](03-tools-and-environment/01-tool-use/main.py)

Lets the LLM invoke external functions such as APIs, calculators, and databases, either in a single turn or through an agentic multi-step tool loop.

### 4. Memory and Learning

This pattern lets agents retain context, user preferences, and lessons across sessions.

#### [Memory Management](04-memory-and-learning/01-memory-management/main.py)

Provides short-term context management, long-term persistent memory, and learning from experience across sessions.

### 5. Multi-Agent Systems

This pattern organizes multiple agents into teams that can collaborate on complex work.

#### [Multi-Agent Collaboration](05-multi-agent-systems/01-multi-agent-collaboration/main.py)

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

### 2. Create a virtual environment

Create and activate a project-local virtual environment before installing the
dependencies:

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

When you are finished, deactivate the environment with:

```bash
deactivate
```

### 3. Install dependencies

Each pattern has its own `requirements.txt`:

```bash
pip install -r 01-reasoning-task-decomposition/01-prompt-chaining/requirements.txt
pip install -r 01-reasoning-task-decomposition/02-planning/requirements.txt
pip install -r 01-reasoning-task-decomposition/03-reflection/requirements.txt
pip install -r 02-control-flow-coordination/01-routing/requirements.txt
pip install -r 02-control-flow-coordination/02-parallelisation/requirements.txt
pip install -r 03-tools-and-environment/01-tool-use/requirements.txt
pip install -r 04-memory-and-learning/01-memory-management/requirements.txt
pip install -r 05-multi-agent-systems/01-multi-agent-collaboration/requirements.txt
```

### 4. Run a pattern

```bash
python 01-reasoning-task-decomposition/01-prompt-chaining/main.py
python 01-reasoning-task-decomposition/02-planning/main.py
python 01-reasoning-task-decomposition/03-reflection/main.py
python 02-control-flow-coordination/01-routing/main.py
python 02-control-flow-coordination/02-parallelisation/main.py
python 03-tools-and-environment/01-tool-use/main.py
python 04-memory-and-learning/01-memory-management/main.py
python 05-multi-agent-systems/01-multi-agent-collaboration/main.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running as a background service (`brew services start ollama`) with `llama3.2` pulled
