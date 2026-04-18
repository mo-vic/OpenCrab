"""OpenAI provider adapter."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import Provider


class OpenAIProvider(Provider):
    """OpenAI API provider adapter."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "openai"

    def supports(self, request: dict[str, Any]) -> bool:
        """Check if this provider can handle the request.

        OpenAI provider supports chat completions requests with messages.
        Validates that the request has a valid messages structure.
        """
        if not isinstance(request, dict):
            return False
        messages = request.get("messages")
        if not messages or not isinstance(messages, list):
            return False
        if len(messages) == 0:
            return False
        # Validate message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg:
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
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages, "stream": stream, **kwargs}

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
                    data = await response.json()
                    yield data

    def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Transform request to OpenAI format."""
        return {
            "model": request.get("model", "gpt-4o"),
            "messages": request["messages"],
            "stream": request.get("stream", False),
            **{k: v for k, v in request.items() if k not in ("model", "messages")},
        }

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Transform OpenAI response to internal format."""
        return {
            "id": response["id"],
            "model": response["model"],
            "choices": response["choices"],
            "usage": response.get("usage"),
        }


def _parse_sse(line: str) -> dict[str, Any]:
    """Parse SSE data line."""
    import json

    return json.loads(line)
