"""Unified LLM adapter — abstracts away provider differences.

Supports: OpenAI-compatible (Ollama, vLLM, etc.), DashScope (通义千问),
ZhipuAI (GLM), and any local model with an OpenAI-compatible endpoint.

Handles two reasoning model patterns:
1. Qwen3/DeepSeek-R1 via Ollama: empty content + separate `reasoning` field
2. MiniMax-M2.1/DeepSeek-R1 via vLLM: `<think>...</think>` tags in content
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from server.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    role: str  # system | user | assistant | tool
    content: str
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool result messages
    tool_call_id: str | None = None


@dataclass
class ToolCallRequest:
    """Parsed tool call from an LLM response."""
    id: str
    function_name: str
    arguments: dict[str, Any]
    raw: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)
    # Tool calls requested by the LLM (None if no tool calling)
    tool_calls: list[ToolCallRequest] | None = None
    # Reasoning/thinking content (Qwen3, DeepSeek-R1, etc.)
    reasoning: str | None = None


# Models known to use reasoning/thinking fields or <think> tags
_REASONING_MODEL_PATTERNS = re.compile(
    r"qwen3|deepseek.*r1|o1|o3|minimax|MiniMax-M1", re.IGNORECASE
)

# Pattern to match <think>...</think> blocks in content (vLLM-served models)
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model is known to produce reasoning output."""
    return bool(_REASONING_MODEL_PATTERNS.search(model_name))


def _strip_think_tags(content: str) -> tuple[str, str | None]:
    """Strip <think>...</think> blocks from content.

    Many models served via vLLM (MiniMax-M2.1, DeepSeek-R1, etc.) embed their
    reasoning inside <think> XML tags within the content field itself.

    Returns:
        (clean_content, thinking_text) — thinking_text is None if no tags found
    """
    if not content or "<think>" not in content:
        return content, None

    # Extract all thinking blocks
    thinking_parts = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking_text = "\n".join(t.strip() for t in thinking_parts if t.strip()) or None

    # Remove <think>...</think> blocks from content
    clean = _THINK_TAG_PATTERN.sub("", content).strip()
    return clean, thinking_text


def _extract_content_from_reasoning(reasoning: str) -> str:
    """Extract the final answer from reasoning text when content is empty.

    Reasoning models sometimes put the entire response in the reasoning field.
    We try to find the "conclusion" or "final answer" portion.
    """
    if not reasoning:
        return ""

    # Look for common conclusion markers in reasoning
    for marker in ["最终回答：", "最终答案：", "回答：", "结论：", "总结：",
                    "所以，", "因此，", "综上，"]:
        idx = reasoning.rfind(marker)
        if idx >= 0:
            conclusion = reasoning[idx + len(marker):].strip()
            if len(conclusion) > 5:
                return conclusion

    # If reasoning is short enough (<200 chars), use it directly
    if len(reasoning) < 200:
        return reasoning.strip()

    # Take the last paragraph as the conclusion
    paragraphs = [p.strip() for p in reasoning.split("\n") if p.strip()]
    if paragraphs:
        last = paragraphs[-1]
        if len(last) > 5:
            return last

    return reasoning[:500].strip()


class LLMAdapter:
    """Unified async LLM client.

    Handles reasoning models (Qwen3, DeepSeek-R1) where the response may
    contain a `reasoning` field instead of or in addition to `content`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or getattr(settings, "llm_timeout", 60)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _make_client(self, timeout: int | None = None) -> httpx.AsyncClient:
        """Create an httpx client, bypassing proxy for local endpoints."""
        t = timeout or self.timeout
        # Bypass system proxy for local/private endpoints (Ollama, vLLM, etc.)
        is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))
        return httpx.AsyncClient(timeout=t, trust_env=not is_local)

    def _serialize_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Serialize LLMMessage list to OpenAI API format.

        Handles special message types: tool calls (assistant) and tool results.
        """
        result = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls is not None:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            result.append(msg)
        return result

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Non-streaming chat completion.

        Handles reasoning models where content may be empty and the
        actual answer lives in the `reasoning` field.
        """
        t0 = time.perf_counter()
        model = kwargs.get("model", self.model)
        payload = {
            "model": model,
            "messages": self._serialize_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        async with self._make_client() as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        content = message.get("content") or ""
        reasoning = message.get("reasoning") or ""

        # Pattern 1: <think> tags in content (vLLM-served MiniMax, DeepSeek-R1, etc.)
        content, think_text = _strip_think_tags(content)
        if think_text:
            logger.info(f"Stripped <think> tags from content ({len(think_text)} chars thinking)")
            reasoning = think_text if not reasoning else reasoning

        # Pattern 2: Empty content + separate reasoning field (Qwen3/Ollama)
        if not content.strip() and reasoning.strip():
            logger.info(f"Content empty, extracting from reasoning ({len(reasoning)} chars)")
            content = _extract_content_from_reasoning(reasoning)

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=data,
            reasoning=reasoning if reasoning else None,
        )

    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict],
        **kwargs,
    ) -> LLMResponse:
        """Chat completion with function calling / tool use.

        Args:
            messages: Conversation messages.
            tools: List of tool definitions in OpenAI format.

        Returns:
            LLMResponse with tool_calls populated if the LLM wants to call tools,
            or with content populated if the LLM gave a final answer.
        """
        t0 = time.perf_counter()
        model = kwargs.get("model", self.model)
        payload = {
            "model": model,
            "messages": self._serialize_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "tools": tools,
            "stream": False,
        }
        async with self._make_client() as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        content = message.get("content") or ""
        reasoning = message.get("reasoning") or ""

        # Strip <think> tags from content (vLLM-served models)
        content, think_text = _strip_think_tags(content)
        if think_text:
            logger.info(f"Tool-call: stripped <think> tags ({len(think_text)} chars thinking)")
            reasoning = think_text if not reasoning else reasoning

        # Parse tool calls if present
        tool_calls = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCallRequest(
                    id=tc.get("id", ""),
                    function_name=func.get("name", ""),
                    arguments=args,
                    raw=tc,
                ))

        # If no tool calls and content is empty, extract from reasoning
        if not tool_calls and not content.strip() and reasoning.strip():
            logger.info("Tool-call response: content empty, extracting from reasoning")
            content = _extract_content_from_reasoning(reasoning)

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=data,
            tool_calls=tool_calls,
            reasoning=reasoning if reasoning else None,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        """Streaming chat completion — yields content deltas.

        Handles two reasoning patterns:
        1. Separate `reasoning` field in delta (Ollama) — buffer and skip
        2. `<think>` tags in content delta (vLLM) — buffer until </think>, then yield
        """
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": self._serialize_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        has_content = False
        reasoning_buffer = []
        # State machine for <think> tag stripping in streaming
        in_think_block = False

        async with self._make_client(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})

                    # Pattern 1: Separate reasoning field (Ollama/Qwen3)
                    if "reasoning" in delta and delta["reasoning"]:
                        reasoning_buffer.append(delta["reasoning"])

                    # Content deltas — check for <think> tags (vLLM pattern)
                    if "content" in delta and delta["content"]:
                        text = delta["content"]

                        if in_think_block:
                            # Inside <think> block — buffer as reasoning
                            if "</think>" in text:
                                # End of thinking block
                                parts = text.split("</think>", 1)
                                reasoning_buffer.append(parts[0])
                                in_think_block = False
                                # Yield remaining content after </think>
                                remainder = parts[1].lstrip()
                                if remainder:
                                    has_content = True
                                    yield remainder
                            else:
                                reasoning_buffer.append(text)
                        elif "<think>" in text:
                            # Start of thinking block
                            parts = text.split("<think>", 1)
                            # Yield content before <think> tag
                            before = parts[0]
                            if before.strip():
                                has_content = True
                                yield before
                            in_think_block = True
                            # Check if </think> also appears in this chunk
                            after = parts[1]
                            if "</think>" in after:
                                think_parts = after.split("</think>", 1)
                                reasoning_buffer.append(think_parts[0])
                                in_think_block = False
                                remainder = think_parts[1].lstrip()
                                if remainder:
                                    has_content = True
                                    yield remainder
                            else:
                                reasoning_buffer.append(after)
                        else:
                            # Normal content — yield directly
                            has_content = True
                            yield text

        # Fallback: If no content was yielded but reasoning exists
        if not has_content and reasoning_buffer:
            full_reasoning = "".join(reasoning_buffer)
            extracted = _extract_content_from_reasoning(full_reasoning)
            if extracted:
                yield extracted


# Thread-safe singleton default adapter
_default_adapter: LLMAdapter | None = None
_adapter_lock = threading.Lock()


def get_llm_adapter(**kwargs) -> LLMAdapter:
    global _default_adapter
    if kwargs:
        return LLMAdapter(**kwargs)
    if _default_adapter is None:
        with _adapter_lock:
            if _default_adapter is None:
                _default_adapter = LLMAdapter()
    return _default_adapter
