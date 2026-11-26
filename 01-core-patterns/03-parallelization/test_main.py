"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_summaries_preserve_order(self):
        async def summarize(section, index):
            await asyncio.sleep(0)
            return {"index": index, "summary": section.upper()}
        with patch.object(m, "summarise_section", side_effect=summarize):
            self.assertEqual(await m.parallel_summarise(["a", "b"]), ["A", "B"])
            self.assertEqual(await m.parallel_summarise([]), [])

    async def test_voting_averages_scores(self):
        with patch.object(m, "llm_call_async", side_effect=['{"score":3}', '{"score":6}', '{"score":9}']):
            result = await m.voting_evaluate("text", "quality")
        self.assertEqual(result["average_score"], 6)
        self.assertEqual(len(result["individual_results"]), 3)

    async def test_invalid_vote_uses_fallback(self):
        with patch.object(m, "llm_call_async", return_value="invalid"):
            result = await m.single_evaluation("text", "criteria", 1)
        self.assertEqual(result, {"run_id": 1, "score": 5, "reasoning": "parse error"})

    async def test_map_reduce_combines_evidence(self):
        with patch.object(m, "llm_call_async", side_effect=["fact A", "fact B", "summary"]) as call:
            facts = await asyncio.gather(m.extract_key_facts("A", "topic"), m.extract_key_facts("B", "topic"))
            self.assertEqual(await m.synthesise_facts(facts, "topic"), "summary")
            self.assertIn("fact A", call.call_args.args[0])
            self.assertIn("fact B", call.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
