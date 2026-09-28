"""
Model Context Protocol (MCP) Pattern

Demonstrates a standardized interface between a model client and external
capabilities (tools/resources/prompts) using protocol-like request contracts.
"""

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class MCPRequest:
    capability: str
    operation: str
    payload: dict[str, Any]


@dataclass
class MCPResponse:
    ok: bool
    data: dict[str, Any]
    error: str | None = None


class MCPServer:
    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register(self, capability: str, operation: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._handlers[(capability, operation)] = handler

    def handle(self, request: MCPRequest) -> MCPResponse:
        handler = self._handlers.get((request.capability, request.operation))
        if handler is None:
            return MCPResponse(
                ok=False,
                data={},
                error=f"Unsupported MCP operation: {request.capability}.{request.operation}",
            )
        try:
            return MCPResponse(ok=True, data=handler(request.payload))
        except Exception as exc:  # noqa: BLE001 - demonstration pattern
            return MCPResponse(ok=False, data={}, error=str(exc))


class MCPClient:
    def __init__(self, server: MCPServer) -> None:
        self.server = server

    def call(self, capability: str, operation: str, payload: dict[str, Any]) -> MCPResponse:
        return self.server.handle(MCPRequest(capability=capability, operation=operation, payload=payload))


def build_server() -> MCPServer:
    server = MCPServer()

    def calculator_add(payload: dict[str, Any]) -> dict[str, Any]:
        return {"result": payload["a"] + payload["b"]}

    def knowledge_lookup(payload: dict[str, Any]) -> dict[str, Any]:
        topic = payload.get("topic", "unknown")
        return {"topic": topic, "summary": f"Standardized response for '{topic}'."}

    server.register("tool", "calculator.add", calculator_add)
    server.register("resource", "knowledge.lookup", knowledge_lookup)
    return server


def run_demo() -> None:
    client = MCPClient(build_server())

    add_result = client.call("tool", "calculator.add", {"a": 13, "b": 29})
    kb_result = client.call("resource", "knowledge.lookup", {"topic": "MCP"})
    missing = client.call("tool", "calendar.create", {"title": "Sprint review"})

    print("=== MCP Demo ===")
    print("calculator.add ->", add_result)
    print("knowledge.lookup ->", kb_result)
    print("unsupported operation ->", missing)


if __name__ == "__main__":
    run_demo()
