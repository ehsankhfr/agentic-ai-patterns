# Agentic AI Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com)

A collection of practical agentic AI design patterns implemented with the OpenAI API (or any OpenAI-compatible local model via [Ollama](https://ollama.com)).

## Patterns

### Core Patterns

#### Chapter 1: [Prompt Chaining](core-patterns/01-prompt-chaining/main.py)
Sequential task decomposition.

#### Chapter 2: [Routing](core-patterns/02-routing/main.py)
Dynamic path selection.

#### Chapter 3: [Parallelization](core-patterns/03-parallelization/main.py)
Concurrent processing.

#### Chapter 4: [Reflection](core-patterns/04-reflection/main.py)
Self-improvement mechanisms.

#### Chapter 5: [Tool Use](core-patterns/05-tool-use/main.py)
External capability integration.

#### Chapter 6: [Planning](core-patterns/06-planning/main.py)
Strategic task management.

#### Chapter 7: [Multi-Agent](core-patterns/07-multi-agent/main.py)
Collaborative systems.

### Advanced Patterns

#### Chapter 8: [Memory Management](advanced-patterns/08-memory-management/main.py)
State persistence.

#### Chapter 9: [Learning and Adaptation](advanced-patterns/09-learning-and-adaptation/main.py)
Dynamic improvement.

#### Chapter 10: [Model Context Protocol (MCP)](advanced-patterns/10-model-context-protocol/main.py)
Standardized interfaces.

#### Chapter 11: [Goal Setting and Monitoring](advanced-patterns/11-goal-setting-and-monitoring/main.py)
Objective tracking.

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
pip install -r core-patterns/01-prompt-chaining/requirements.txt
pip install -r core-patterns/02-routing/requirements.txt
pip install -r core-patterns/03-parallelization/requirements.txt
pip install -r core-patterns/04-reflection/requirements.txt
pip install -r core-patterns/05-tool-use/requirements.txt
pip install -r core-patterns/06-planning/requirements.txt
pip install -r core-patterns/07-multi-agent/requirements.txt
pip install -r advanced-patterns/08-memory-management/requirements.txt
pip install -r advanced-patterns/09-learning-and-adaptation/requirements.txt
pip install -r advanced-patterns/10-model-context-protocol/requirements.txt
pip install -r advanced-patterns/11-goal-setting-and-monitoring/requirements.txt
```

### 4. Run a pattern

```bash
python core-patterns/01-prompt-chaining/main.py
python core-patterns/02-routing/main.py
python core-patterns/03-parallelization/main.py
python core-patterns/04-reflection/main.py
python core-patterns/05-tool-use/main.py
python core-patterns/06-planning/main.py
python core-patterns/07-multi-agent/main.py
python advanced-patterns/08-memory-management/main.py
python advanced-patterns/09-learning-and-adaptation/main.py
python advanced-patterns/10-model-context-protocol/main.py
python advanced-patterns/11-goal-setting-and-monitoring/main.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running as a background service (`brew services start ollama`) with `llama3.2` pulled
