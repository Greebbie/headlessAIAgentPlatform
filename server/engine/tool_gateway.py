"""Tool Gateway — unified tool invocation with auth, retry, audit."""

from __future__ import annotations

import time
import asyncio
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.models.tool import ToolDefinition
from server.engine.audit_logger import AuditLogger


class ToolInvocationError(Exception):
    def __init__(self, tool_name: str, message: str, recoverable: bool = True):
        self.tool_name = tool_name
        self.recoverable = recoverable
        super().__init__(f"Tool '{tool_name}': {message}")


# ── Built-in function tool registry ─────────────────────────────
# Maps function tool names to their mock-tool router paths
_BUILTIN_FUNCTION_TOOLS: dict[str, str] = {
    "calculator": "/api/v1/mock-tools/calculator",
    "weather": "/api/v1/mock-tools/weather",
    "unit_converter": "/api/v1/mock-tools/unit_converter",
    "timestamp": "/api/v1/mock-tools/timestamp",
}


class ToolGateway:
    """Invoke any registered tool with retry, timeout, auth, and audit."""

    def __init__(self, db: AsyncSession, audit: AuditLogger | None = None):
        self.db = db
        self.audit = audit

    async def get_tool(self, tool_id: str) -> ToolDefinition | None:
        result = await self.db.execute(
            select(ToolDefinition).where(ToolDefinition.id == tool_id, ToolDefinition.enabled == True)
        )
        return result.scalar_one_or_none()

    async def invoke(self, tool_id: str, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a tool by id. Handles auth, retry, timeout."""
        tool = await self.get_tool(tool_id)
        if tool is None:
            raise ToolInvocationError(tool_id, "Tool not found or disabled", recoverable=False)

        t0 = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(tool.max_retries + 1):
            try:
                result = await self._call(tool, input_data or {})
                latency = (time.perf_counter() - t0) * 1000

                if self.audit:
                    self.audit.log(
                        "tool_call",
                        tool_meta={
                            "tool_id": tool.id,
                            "tool_name": tool.name,
                            "input": input_data,
                            "output": result,
                            "success": True,
                            "attempt": attempt + 1,
                        },
                        latency_ms=latency,
                    )
                return result

            except Exception as e:
                last_error = e
                if attempt < tool.max_retries:
                    await asyncio.sleep(tool.retry_backoff_ms / 1000 * (2 ** attempt))

        # All retries exhausted
        latency = (time.perf_counter() - t0) * 1000
        if self.audit:
            self.audit.log(
                "tool_call",
                tool_meta={
                    "tool_id": tool.id,
                    "tool_name": tool.name,
                    "input": input_data,
                    "error": str(last_error),
                    "success": False,
                    "attempts": tool.max_retries + 1,
                },
                latency_ms=latency,
            )
        raise ToolInvocationError(tool.name, f"Failed after {tool.max_retries + 1} attempts: {last_error}")

    async def _call(self, tool: ToolDefinition, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call — dispatches to HTTP or built-in function."""
        # Function-type tools: route to built-in mock tools via local HTTP
        if tool.category == "function":
            return await self._call_function_tool(tool, input_data)

        # HTTP-type tools (api, webhook, rpc)
        return await self._call_http_tool(tool, input_data)

    async def _call_function_tool(self, tool: ToolDefinition, input_data: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a function-type tool to a built-in handler via local HTTP."""
        # Try to resolve the tool name to a built-in path
        tool_name_lower = tool.name.lower().replace(" ", "_").replace("-", "_")
        local_path = _BUILTIN_FUNCTION_TOOLS.get(tool_name_lower)

        if not local_path and tool.endpoint:
            # If tool has a custom endpoint, use it as HTTP
            return await self._call_http_tool(tool, input_data)

        if not local_path:
            raise ToolInvocationError(
                tool.name,
                f"Function tool '{tool.name}' has no built-in handler and no endpoint configured",
                recoverable=False,
            )

        # Call the local FastAPI endpoint
        timeout = tool.timeout_ms / 1000
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"http://127.0.0.1:8000{local_path}",
                json=input_data,
            )
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}

    async def _call_http_tool(self, tool: ToolDefinition, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute an HTTP tool call."""
        if not tool.endpoint:
            raise ToolInvocationError(tool.name, "No endpoint configured for HTTP tool", recoverable=False)

        headers = {"Content-Type": "application/json"}

        # Apply auth
        if tool.auth_config:
            auth_type = tool.auth_config.get("type", "none")
            if auth_type == "bearer":
                headers["Authorization"] = f"Bearer {tool.auth_config.get('token', '')}"
            elif auth_type == "api_key":
                key_name = tool.auth_config.get("header", "X-API-Key")
                headers[key_name] = tool.auth_config.get("token", "")

        timeout = tool.timeout_ms / 1000

        async with httpx.AsyncClient(timeout=timeout) as client:
            if tool.method.upper() == "GET":
                resp = await client.get(tool.endpoint, params=input_data, headers=headers)
            elif tool.method.upper() == "POST":
                resp = await client.post(tool.endpoint, json=input_data, headers=headers)
            elif tool.method.upper() == "PUT":
                resp = await client.put(tool.endpoint, json=input_data, headers=headers)
            elif tool.method.upper() == "DELETE":
                resp = await client.delete(tool.endpoint, params=input_data, headers=headers)
            else:
                resp = await client.post(tool.endpoint, json=input_data, headers=headers)

            resp.raise_for_status()

            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}

    async def test_connectivity(self, tool_id: str) -> dict[str, Any]:
        """Test if a tool endpoint is reachable."""
        tool = await self.get_tool(tool_id)
        if tool is None:
            return {"success": False, "error": "Tool not found"}

        if tool.category == "function":
            # For function tools, check if the built-in handler exists
            tool_name_lower = tool.name.lower().replace(" ", "_").replace("-", "_")
            if tool_name_lower in _BUILTIN_FUNCTION_TOOLS:
                return {"success": True, "status_code": 200, "latency_ms": 0, "note": "Built-in function tool"}
            elif tool.endpoint:
                pass  # Fall through to HTTP test
            else:
                return {"success": False, "error": "No built-in handler or endpoint configured"}

        if not tool.endpoint:
            return {"success": False, "error": "No endpoint configured"}

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.request(tool.method.upper(), tool.endpoint, headers={"Content-Type": "application/json"})
                return {
                    "success": True,
                    "status_code": resp.status_code,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
