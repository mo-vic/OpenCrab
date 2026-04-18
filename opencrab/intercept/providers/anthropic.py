"""Anthropic provider adapter."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import Provider


class AnthropicProvider(Provider):
    """Anthropic API provider adapter."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "anthropic"

    def supports(self, request: dict[str, Any]) -> bool:
        """Check if this provider can handle the request.

        Anthropic provider supports messages API requests with messages array.
        Validates that the request has a valid messages structure.
        """
        if not isinstance(request, dict):
            return False
        messages = request.get("messages")
        if not messages or not isinstance(messages, list):
            return False
        if len(messages) == 0:
            return False
        # Validate message structure - Anthropic supports content as list or string
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg:
                return False
            # Anthropic allows content to be a list (multi-modal) or string
            content = msg.get("content")
            if content is not None and not isinstance(content, (str, list)):
                return False
        return True

    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": _convert_to_anthropic_format(messages, kwargs.get("tool_calls")),
            "stream": stream,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                if stream:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            yield _parse_sse(data)
                else:
                    result = await response.json()
                    yield _convert_from_anthropic_format(result)

    def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Transform request to Anthropic format."""
        messages = request["messages"]
        system = None
        if messages and messages[0].get("role") == "system":
            system = messages[0].pop("content")
        transformed = {
            "model": request.get("model", "claude-sonnet-4-20250514"),
            "messages": _convert_to_anthropic_format(messages),
            "stream": request.get("stream", False),
        }
        if system:
            transformed["system"] = system
        return {
            **transformed,
            **{k: v for k, v in request.items() if k not in ("model", "messages")},
        }

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Transform Anthropic response to internal format."""
        return {
            "id": response.get("id", ""),
            "model": response["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response["content"][0]["text"]},
                    "finish_reason": response.get("stop_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": sum(response.get("usage", {}).values()),
            },
        }


def _convert_to_anthropic_format(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert OpenAI-style messages to Anthropic format."""
    result = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        content = msg.get("content", "")
        # Handle tool calls - convert to Anthropic tool_use block format
        msg_tool_calls = msg.get("tool_calls")
        if msg_tool_calls:
            for tc in msg_tool_calls:
                if isinstance(tc, dict) and tc.get("name"):
                    tool_use = {
                        "type": "tool_use",
                        "id": tc.get("id", f"tool_{tc['name']}"),
                        "name": tc["name"],
                        "input": tc.get("arguments", {}),
                    }
                    result.append({"role": msg["role"], "content": None, "tool_use": tool_use})
                    continue
        # Handle tool results in content
        if isinstance(content, str) and content:
            result.append({"role": msg["role"], "content": content})
        elif msg.get("tool_use"):
            # Pass through tool_use blocks
            result.append({"role": msg["role"], "content": None, "tool_use": msg["tool_use"]})
        elif msg.get("role") == "user":
            result.append({"role": msg["role"], "content": content or ""})
    return result


def _convert_from_anthropic_format(response: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic response to OpenAI-style format."""
    return {
        "id": response.get("id", ""),
        "model": response["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response["content"][0]["text"]},
                "finish_reason": response.get("stop_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": sum(response.get("usage", {}).values()),
        },
    }


def _parse_sse(line: str) -> dict[str, Any]:
    """Parse SSE data line."""
    import json

    if line.startswith("data: "):
        line = line[6:]
    return json.loads(line)
