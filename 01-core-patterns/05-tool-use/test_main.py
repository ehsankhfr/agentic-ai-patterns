"""Deterministic unit tests; no live model required."""
import unittest
import asyncio
import json
from unittest.mock import patch

import main as m


class PatternTests(unittest.TestCase):
    def test_weather_lookup_and_unknown_city(self):
        self.assertEqual(m.get_weather(" London ")["temperature_c"], 14)
        self.assertIn("error", m.get_weather("unknown"))

    def test_calculator_precedence_and_invalid_expression(self):
        self.assertEqual(m.calculate("2+3*4")["result"], 14)
        for expression in ("1/0", "unknown", "(1).__class__"):
            with self.subTest(expression=expression):
                self.assertIn("error", m.calculate(expression))

    def test_dispatch_valid_json(self):
        result = json.loads(m.dispatch_tool("calculate", '{"expression":"3*7"}'))
        self.assertEqual(result["result"], 21)

    def test_dispatch_unknown_or_bad_arguments(self):
        for name, args in [("missing", "{}"), ("calculate", "bad JSON"), ("calculate", "{}")]:
            with self.subTest(name=name, args=args):
                self.assertIn("error", json.loads(m.dispatch_tool(name, args)))

    def test_knowledge_base_match_and_fallback(self):
        self.assertIn("1991", m.search_knowledge_base("PYTHON")["result"])
        self.assertIn("No specific information", m.search_knowledge_base("unknown")["result"])


if __name__ == "__main__":
    unittest.main()
