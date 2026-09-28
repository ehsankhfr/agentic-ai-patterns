"""
Model Context Protocol (MCP) Pattern

An LLM uses an in-process, protocol-style simulation of tool, resource, and
prompt capabilities, including client-side discovery across multiple servers.
This is not a wire-compatible MCP implementation and does not use the MCP SDK.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"


def llm_call(
    messages: list[dict[str, Any]],
    model: str = MODEL,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message


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


@dataclass(frozen=True)
class Capability:
    kind: str
    name: str
    description: str


@dataclass(frozen=True)
class DiscoveredCapability:
    server: str
    kind: str
    name: str
    description: str


class MCPServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._handlers: dict[
            tuple[str, str],
            tuple[Callable[[dict[str, Any]], dict[str, Any]], Capability],
        ] = {}

    def register(
        self,
        kind: str,
        operation: str,
        description: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        key = (kind, operation)
        if key in self._handlers:
            raise ValueError(f"Capability already registered: {kind}.{operation}")
        capability = Capability(kind=kind, name=operation, description=description)
        self._handlers[key] = (handler, capability)

    def discover(self) -> list[Capability]:
        return [capability for _, capability in self._handlers.values()]

    def handle(self, request: MCPRequest) -> MCPResponse:
        registered = self._handlers.get((request.capability, request.operation))
        if registered is None:
            return MCPResponse(
                ok=False,
                data={},
                error=f"Unsupported capability: {request.capability}.{request.operation}",
            )
        handler, _ = registered
        try:
            return MCPResponse(ok=True, data=handler(request.payload))
        except Exception as exc:  # noqa: BLE001 - demonstration of an error response
            return MCPResponse(ok=False, data={}, error=str(exc))


class MCPClient:
    def __init__(self, servers: list[MCPServer]) -> None:
        self.servers = {server.name: server for server in servers}
        if len(self.servers) != len(servers):
            raise ValueError("Server names must be unique.")

    def discover(self) -> list[DiscoveredCapability]:
        return [
            DiscoveredCapability(
                server=server.name,
                kind=capability.kind,
                name=capability.name,
                description=capability.description,
            )
            for server in self.servers.values()
            for capability in server.discover()
        ]

    def call(
        self,
        server_name: str,
        capability: str,
        operation: str,
        payload: dict[str, Any],
    ) -> MCPResponse:
        server = self.servers.get(server_name)
        if server is None:
            return MCPResponse(
                ok=False,
                data={},
                error=f"Unknown server: {server_name}",
            )
        request = MCPRequest(
            capability=capability,
            operation=operation,
            payload=payload,
        )
        return server.handle(request)


def build_servers() -> list[MCPServer]:
    knowledge_server = MCPServer("knowledge")
    knowledge_server.register(
        "tool",
        "calculator.add",
        "Add two numbers.",
        lambda payload: {"result": payload["a"] + payload["b"]},
    )
    knowledge_server.register(
        "resource",
        "knowledge.lookup",
        "Look up a short topic summary.",
        lambda payload: {
            "topic": payload.get("topic", "unknown"),
            "summary": f"Standardized response for '{payload.get('topic', 'unknown')}'.",
        },
    )
    knowledge_server.register(
        "prompt",
        "report.summary",
        "Build a prompt for summarizing a report.",
        lambda payload: {
            "prompt": (
                f"Summarize the {payload.get('report_type', 'project')} report "
                "with findings and next steps."
            )
        },
    )

    workspace_server = MCPServer("workspace")
    workspace_server.register(
        "resource",
        "release.notes",
        "Read the latest release notes.",
        lambda _: {"version": "2.4", "notes": ["Improved routing", "Fixed login retries"]},
    )
    return [knowledge_server, workspace_server]


def run_demo() -> None:
    client = MCPClient(build_servers())
    capabilities = client.discover()
    capability_catalog = [
        {
            "server": capability.server,
            "kind": capability.kind,
            "name": capability.name,
            "description": capability.description,
        }
        for capability in capabilities
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an agent connected to protocol-style capabilities. "
                "Use the available capabilities to complete the user's request. "
                "Call calculator.add for the requested arithmetic, knowledge.lookup "
                "for the topic, report.summary to obtain a summarization prompt, and "
                "release.notes to retrieve the notes. Use only the listed capabilities.\n"
                f"Discovered capabilities: {json.dumps(capability_catalog)}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Add 13 and 29, look up MCP, retrieve the release notes, and prepare "
                "a concise release summary using the report-summary prompt."
            ),
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "call_capability",
                "description": (
                    "Invoke one discovered capability on one of the in-process servers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server": {"type": "string"},
                        "kind": {"type": "string", "enum": ["tool", "resource", "prompt"]},
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["server", "kind", "name", "arguments"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    print("=== MCP-Style In-Process Simulation ===")
    print("\nClient discovery across servers:")
    for capability in capabilities:
        print(
            f"- {capability.server}: {capability.kind}.{capability.name} "
            f"- {capability.description}"
        )

    print("\nLLM-selected capability calls:")
    for _ in range(9):
        message = llm_call(messages, tools=tools)
        if not message.tool_calls:
            if message.content:
                messages.append({"role": "assistant", "content": message.content})
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )
        for tool_call in message.tool_calls:
            if tool_call.function.name != "call_capability":
                raise ValueError(
                    f"Unsupported LLM tool call: {tool_call.function.name}"
                )
            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("LLM capability call arguments must be a JSON object.")
            server_name = arguments.get("server")
            kind = arguments.get("kind")
            name = arguments.get("name")
            payload = arguments.get("arguments")
            if not isinstance(server_name, str) or not isinstance(kind, str):
                raise ValueError("LLM capability call must include server and kind strings.")
            if not isinstance(name, str) or not isinstance(payload, dict):
                raise ValueError("LLM capability call must include a name and arguments object.")
            if not any(
                capability.server == server_name
                and capability.kind == kind
                and capability.name == name
                for capability in capabilities
            ):
                raise ValueError(
                    f"LLM selected an undiscovered capability: {server_name}.{kind}.{name}"
                )
            result = client.call(server_name, kind, name, payload)
            print(f"- {server_name}.{kind}.{name} -> {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"ok": result.ok, "data": result.data, "error": result.error}
                    ),
                }
            )
    else:
        raise RuntimeError("LLM exceeded the maximum number of tool-calling rounds.")

    if messages[-1].get("role") == "assistant":
        print("\nLLM response:")
        print(messages[-1]["content"])
    else:
        raise RuntimeError("LLM did not provide a final response after capability calls.")


if __name__ == "__main__":
    run_demo()
