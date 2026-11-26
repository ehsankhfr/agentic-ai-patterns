"""Deterministic unit tests; no live model required."""
import unittest
from unittest.mock import Mock, patch

import main as m


class PatternTests(unittest.TestCase):
    def test_discovery_aggregates_all_capability_kinds(self):
        client = m.MCPClient(m.build_servers())
        items = {(item.server, item.kind, item.name) for item in client.discover()}
        self.assertEqual(items, {
            ("knowledge", "tool", "calculator.add"),
            ("knowledge", "resource", "knowledge.lookup"),
            ("knowledge", "prompt", "report.summary"),
            ("workspace", "resource", "release.notes"),
        })

    def test_tool_resource_and_prompt_dispatch(self):
        client = m.MCPClient(m.build_servers())
        result = client.call("knowledge", "tool", "calculator.add", {"a": 2, "b": 3})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"result": 5})
        self.assertEqual(client.call("workspace", "resource", "release.notes", {}).data["version"], "2.4")
        self.assertIn("sales", client.call("knowledge", "prompt", "report.summary", {"report_type": "sales"}).data["prompt"])

    def test_unknown_server_capability_and_handler_error(self):
        client = m.MCPClient(m.build_servers())
        for args in [
            ("missing", "tool", "calculator.add", {}),
            ("knowledge", "tool", "missing", {}),
            ("knowledge", "tool", "calculator.add", {"a": 1}),
        ]:
            with self.subTest(args=args):
                result = client.call(*args)
                self.assertFalse(result.ok)
                self.assertTrue(result.error)
                self.assertEqual(result.data, {})

    def test_duplicate_registration_and_server_names_rejected(self):
        server = m.MCPServer("demo")
        server.register("tool", "echo", "echo", lambda p: p)
        with self.assertRaises(ValueError):
            server.register("tool", "echo", "again", lambda p: p)
        with self.assertRaises(ValueError):
            m.MCPClient([server, m.MCPServer("demo")])


if __name__ == "__main__":
    unittest.main()
