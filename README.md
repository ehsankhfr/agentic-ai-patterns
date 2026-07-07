# Agentic AI Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com)

A collection of practical agentic AI design patterns implemented with the OpenAI API (or any OpenAI-compatible local model via [Ollama](https://ollama.com)).

## Patterns

### [Prompt Chaining](prompt-chaining/main.py)

Decomposes a task into a sequence of steps where each LLM call processes the output of the previous one. Useful for structured pipelines like Research → Draft → Edit.

### [Parallelisation](parallelisation/main.py)

Runs multiple LLM calls concurrently to reduce latency when sub-tasks are independent. Demonstrates two strategies:

- **Sectioning** — split a large task into independent chunks processed in parallel
- **Voting** — run the same prompt N times and aggregate results for higher reliability

### [Routing](routing/main.py)

Classifies an input and directs it to a specialised handler. Keeps prompts focused by using a lightweight router. Demonstrates two strategies:

- **LLM-based router** — the model classifies intent in natural language
- **Structured router** — the model returns JSON for unambiguous classification

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
pip install -r prompt-chaining/requirements.txt
pip install -r parallelisation/requirements.txt
pip install -r routing/requirements.txt
```

### 3. Run a pattern

```bash
python prompt-chaining/main.py
python parallelisation/main.py
python routing/main.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running as a background service (`brew services start ollama`) with `llama3.2` pulled
