"""Deterministic unit tests; no live model required."""
import unittest
from unittest.mock import Mock, patch

import main as m


class PatternTests(unittest.TestCase):
    def test_session_clears_temporary_state_and_compacts(self):
        session = m.Session("s", max_turns=2, state={"temp:key": 1, "user:key": 2})
        with patch.object(m, "llm_call", return_value="earlier summary") as call:
            session.add_turn("user", "old fact")
            session.add_turn("assistant", "reply")
            session.add_turn("user", "new fact")
        self.assertEqual(session.state, {"user:key": 2})
        self.assertEqual(len(session.turns), 2)
        self.assertIn("earlier summary", session.context())
        self.assertIn("old fact", call.call_args.args[0])

    def test_similarity_handles_empty_and_orthogonal_vectors(self):
        self.assertEqual(m._cosine_similarity_dense([], []), 0)
        self.assertEqual(m._cosine_similarity_dense([1, 0], [0, 1]), 0)
        self.assertAlmostEqual(m._cosine_similarity_dense([2, 0], [1, 0]), 1)
        self.assertAlmostEqual(m._cosine_similarity_sparse(m._tokenize("CAT"), m._tokenize("cat")), 1)

    def test_deduplication_and_user_isolation(self):
        store = m.InMemoryMemoryStore()
        with patch.object(m, "_embed", return_value=[]):
            first = store.add("Cat", "u1", confidence=0.2)
            self.assertIs(store.add(" cat ", "u1", confidence=0.9), first)
            store.add("Cat", "u2")
        self.assertEqual(first.confidence, 0.9)
        self.assertEqual(len(store.list_memories("u1")), 1)
        self.assertTrue(store.forget("u1", "Cat"))
        self.assertEqual(store.list_memories("u1"), [])
        self.assertEqual(len(store.list_memories("u2")), 1)

    def test_sparse_retrieval_filters_user_and_memory_type(self):
        store = m.InMemoryMemoryStore()
        with patch.object(m, "_embed", return_value=[]):
            store.add("cat food", "u1")
            store.add("cat rule", "u1", memory_type="procedural")
            store.add("cat secret", "u2")
            self.assertEqual(store.search("cat", "u1", memory_types=["semantic"], expand_query=False), ["cat food"])
            self.assertEqual(store.search("unrelated", "u1", expand_query=False), [])
        self.assertIsNotNone(store.list_memories("u1")[0].last_used_at)

    def test_feedback_creates_procedural_memory(self):
        store = m.InMemoryMemoryStore()
        with patch.object(m, "_embed", return_value=[]), patch.object(m, "llm_call", return_value="Verify sources"):
            rule = store.learn_from_feedback("u1", "task", "check facts")
        self.assertEqual((rule.text, rule.memory_type), ("Verify sources", "procedural"))


if __name__ == "__main__":
    unittest.main()
