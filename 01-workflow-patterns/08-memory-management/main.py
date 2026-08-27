"""
Memory Management Pattern with Ollama (OpenAI-compatible API)

Agents need two complementary kinds of memory, much like humans:

  - Short-Term (Contextual) Memory: the recent turns of the current
    conversation, held within the LLM's context window. It is fast and
    free to access but ephemeral and size-limited.

  - Long-Term (Persistent) Memory: facts, preferences, past events, and
    reusable strategies
    that must survive across sessions. It lives outside the context
    window (a database, a vector store, ...) and is retrieved on demand
    and merged back into the short-term context.

Three strategies are demonstrated:

  1. Short-term memory management – a Session holds a scoped `state`
                                    dict (session/user/app/temp prefixes,
                                    mirroring ADK's convention) plus a
                                    sliding window of turns. Once the
                                    window overflows, older turns are
                                    summarised so the context stays small
                                    without losing information.

  2. Long-term memory management  – an InMemoryMemoryStore persists
                                    facts across sessions and exposes a
                                    `search` method that ranks stored
                                    memories by similarity to a query
                                    (a lightweight bag-of-words cosine
                                    score stands in for a real embedding
                                    / vector database). A new session
                                    recalls relevant memories and folds
                                    them into its short-term context
                                    before answering.

  3. Learning from experience     – episodic memories record task outcomes,
                                    feedback is converted into procedural
                                    rules, and duplicate memories are
                                    consolidated before retrieval.
"""

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "llama3.2"


def llm_call(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Single LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Strategy 1: Short-term memory management
# A Session keeps a scoped state dict and a bounded window of turns.
# Older turns are summarised instead of dropped so context stays small.
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    """Tracks one chat thread: scoped state plus a sliding turn window.

    Keys in `state` follow ADK-style prefixes:
      - "user:"  -> persists for this user across sessions
      - "app:"   -> shared by all users of the application
      - "temp:"  -> valid only for the current turn, never persisted
      - (none)   -> scoped to this session only
    """

    session_id: str
    max_turns: int = 6
    state: dict = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role, content))
        # Turns tagged temp: only matter for the current exchange.
        self.state = {k: v for k, v in self.state.items() if not k.startswith("temp:")}
        if len(self.turns) > self.max_turns:
            self._compact()

    def _compact(self) -> None:
        """Summarise the oldest half of the window to keep context bounded."""
        overflow = len(self.turns) - self.max_turns
        stale, self.turns = self.turns[:overflow], self.turns[overflow:]
        stale_text = "\n".join(f"{t.role}: {t.content}" for t in stale)
        prior_summary = f"{self.summary}\n" if self.summary else ""
        system = "You compress conversation history into a short factual summary."
        prompt = (
            "Summarise the following exchange in 1-3 sentences, keeping any facts, "
            f"names, or decisions a future turn might need:\n\n{prior_summary}{stale_text}"
        )
        self.summary = llm_call(prompt, system)

    def context(self) -> str:
        """Render summary + recent turns as the short-term context block."""
        parts = []
        if self.summary:
            parts.append(f"[Earlier summary] {self.summary}")
        parts.extend(f"{t.role}: {t.content}" for t in self.turns)
        return "\n".join(parts)


def run_short_term_demo() -> None:
    print("=== Strategy 1: Short-Term Memory (sliding window + summarisation) ===\n")
    session = Session(session_id="s1", max_turns=4)
    session.state["user:name"] = "Maya"
    session.state["task_status"] = "idle"

    exchanges = [
        ("user", "Hi, I'm Maya. I'm planning a trip to Lisbon."),
        ("assistant", "Nice to meet you, Maya! When are you thinking of travelling?"),
        ("user", "Sometime in October, for about a week."),
        ("assistant", "Got it — a week in Lisbon in October. Any must-see interests?"),
        ("user", "I love seafood and old bookshops."),
        ("assistant", "Great, I'll keep seafood spots and bookshops in mind."),
        ("user", "Also, what was the date range I mentioned again?"),
    ]
    for role, content in exchanges:
        session.add_turn(role, content)
        session.state["temp:last_role"] = role

    print("Session state:", session.state)
    print("\nContext handed to the LLM:")
    print(session.context())
    print()

    answer = llm_call(
        f"Conversation so far:\n{session.context()}\n\nAnswer the user's last question.",
        system="You are a helpful travel assistant. Use only the given context.",
    )
    print(f"\nAssistant: {answer}\n")


# ---------------------------------------------------------------------------
# Strategy 2: Long-term memory management
# Facts persist across sessions in a store and are retrieved by similarity
# (a bag-of-words cosine score stands in for a real embedding/vector DB).
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Memory:
    text: str
    user_id: str
    memory_type: str = "semantic"  # semantic, episodic, or procedural
    source_session_id: str = ""
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    vector: Counter = field(default_factory=Counter)


class InMemoryMemoryStore:
    """Minimal stand-in for a MemoryService / vector database.

    Mirrors the two operations agent frameworks expect from long-term
    memory: `add_session_to_memory` (persist facts extracted from a
    session) and `search_memory` (semantic-style retrieval by query).
    """

    def __init__(self) -> None:
        self._memories: list[Memory] = []

    def add(
        self,
        text: str,
        user_id: str,
        memory_type: str = "semantic",
        source_session_id: str = "",
        confidence: float = 1.0,
    ) -> Memory:
        """Add a memory unless an equivalent memory already exists."""
        normalized = text.strip().lower()
        for memory in self._memories:
            if (
                memory.user_id == user_id
                and memory.memory_type == memory_type
                and memory.text.strip().lower() == normalized
            ):
                memory.confidence = max(memory.confidence, confidence)
                return memory

        memory = Memory(
            text=text.strip(),
            user_id=user_id,
            memory_type=memory_type,
            source_session_id=source_session_id,
            confidence=max(0.0, min(1.0, confidence)),
            vector=_tokenize(text),
        )
        self._memories.append(memory)
        return memory

    def add_session_to_memory(self, session: Session, user_id: str) -> None:
        """Extract durable facts from a finished session and store them."""
        transcript = session.context()
        system = "Extract durable facts about the user worth remembering long-term."
        prompt = (
            "From this conversation, list only the durable facts/preferences worth "
            "remembering in future sessions, one per line (no commentary):\n\n"
            f"{transcript}"
        )
        facts = llm_call(prompt, system)
        for line in facts.splitlines():
            line = line.strip("-• ").strip()
            if line:
                self.add(line, user_id, source_session_id=session.session_id)

    def record_episode(
        self,
        user_id: str,
        session: Session,
        outcome: str,
        feedback: str = "",
    ) -> Memory:
        """Store a task experience so similar future tasks can reuse it."""
        system = "Extract one concise, reusable lesson from an agent experience."
        prompt = (
            "Summarise this experience as one sentence containing the task, "
            f"outcome ({outcome}), and reusable lesson. Feedback: {feedback or 'none'}\n\n"
            f"{session.context()}"
        )
        lesson = llm_call(prompt, system)
        return self.add(
            lesson,
            user_id,
            memory_type="episodic",
            source_session_id=session.session_id,
            confidence=0.8 if outcome == "success" else 0.9,
        )

    def learn_from_feedback(self, user_id: str, task: str, feedback: str) -> Memory:
        """Turn explicit feedback into a reusable procedural rule."""
        system = "Convert user feedback into one concise instruction for future tasks."
        prompt = (
            f"Task: {task}\nFeedback: {feedback}\n\n"
            "Return only one imperative rule, without commentary."
        )
        rule = llm_call(prompt, system)
        return self.add(rule, user_id, memory_type="procedural", confidence=0.95)

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 3,
        memory_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[str]:
        query_vec = _tokenize(query)
        scored = [
            (m, _cosine_similarity(query_vec, m.vector))
            for m in self._memories
            if m.user_id == user_id
            and (memory_types is None or m.memory_type in memory_types)
        ]
        scored = [pair for pair in scored if pair[1] > min_score]
        scored.sort(key=lambda pair: pair[1] * pair[0].confidence, reverse=True)
        for memory, _ in scored[:top_k]:
            memory.last_used_at = time.time()
        return [memory.text for memory, _ in scored[:top_k]]

    def list_memories(self, user_id: str) -> list[Memory]:
        """Return a user's memories without exposing the internal list."""
        return [m for m in self._memories if m.user_id == user_id]

    def forget(self, user_id: str, text: str) -> bool:
        """Explicitly forget an exact memory; return whether one was removed."""
        for index, memory in enumerate(self._memories):
            if memory.user_id == user_id and memory.text == text:
                del self._memories[index]
                return True
        return False


def run_long_term_demo() -> None:
    print("=== Strategy 2: Long-Term Memory (persist + semantic recall) ===\n")
    memory_store = InMemoryMemoryStore()
    user_id = "user-maya"

    # Session A: user shares preferences; facts are distilled into long-term memory.
    session_a = Session(session_id="a")
    for role, content in [
        ("user", "I'm vegetarian and I'm allergic to peanuts."),
        ("assistant", "Noted — vegetarian meals, no peanuts."),
        ("user", "Also, I prefer window seats when flying."),
    ]:
        session_a.add_turn(role, content)
    memory_store.add_session_to_memory(session_a, user_id)
    print("Facts stored in long-term memory:")
    for m in memory_store.list_memories(user_id):
        print(f"  - [{m.memory_type}] {m.text}")

    # Session B: a brand-new session recalls relevant memories before answering.
    print("\n--- New session starts later ---")
    session_b = Session(session_id="b")
    new_message = "Can you suggest a dinner and book my flight seat?"
    recalled = memory_store.search(new_message, user_id)
    print("Recalled memories:", recalled)

    session_b.add_turn("user", new_message)
    context = session_b.context()
    if recalled:
        context = "[Recalled long-term memory]\n" + "\n".join(recalled) + "\n\n" + context

    answer = llm_call(
        f"{context}\n\nRespond to the user's request using the recalled memory where relevant.",
        system="You are a helpful assistant with access to the user's long-term preferences.",
    )
    print(f"\nAssistant: {answer}\n")


# ---------------------------------------------------------------------------
# Strategy 3: Learning from experience
# Failed outcomes become episodes, explicit feedback becomes a procedure,
# and both are retrieved before the next attempt.
# ---------------------------------------------------------------------------

def run_learning_demo() -> None:
    print("=== Strategy 3: Learning from Experience ===\n")
    memory_store = InMemoryMemoryStore()
    user_id = "user-maya"

    failed_session = Session(session_id="failed-trip-plan")
    failed_session.add_turn("user", "Plan a Lisbon dinner itinerary.")
    failed_session.add_turn(
        "assistant",
        "Here are several popular restaurants, including a steakhouse.",
    )
    episode = memory_store.record_episode(
        user_id,
        failed_session,
        outcome="failure",
        feedback="The recommendation ignored that I am vegetarian.",
    )
    rule = memory_store.learn_from_feedback(
        user_id,
        task="recommend a dinner",
        feedback="The recommendation ignored that I am vegetarian.",
    )
    print(f"Stored episode: [{episode.memory_type}] {episode.text}")
    print(f"Learned rule: [{rule.memory_type}] {rule.text}")

    new_task = "Recommend a dinner in Lisbon"
    recalled = memory_store.search(
        new_task,
        user_id,
        top_k=5,
        memory_types=["episodic", "procedural"],
    )
    print("\nMemories retrieved for the next attempt:")
    for memory in recalled:
        print(f"  - {memory}")

    answer = llm_call(
        "Relevant lessons from previous attempts:\n"
        + "\n".join(recalled)
        + f"\n\nNew task: {new_task}\nApply the lessons and answer the user.",
        system="You are an assistant that learns from previous outcomes.",
    )
    print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY MANAGEMENT DEMO")
    print("=" * 60)
    print()

    run_short_term_demo()
    print("-" * 60)
    run_long_term_demo()
    print("-" * 60)
    run_learning_demo()
