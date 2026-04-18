"""Intercept proxy server — captures trajectories and routes requests."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from opencrab.config import intercept_config

from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .storage import TrajectoryStore

logger = structlog.get_logger()

_provider_registry: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

_providers: dict[str, Any] = {}
_storage: TrajectoryStore | None = None
_serving_url: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _storage, _providers, _serving_url
    _storage = TrajectoryStore()
    await _storage.init()

    # Load intercept config (supports env var overrides: OPENCRAB_INTERCEPT_*)
    config = intercept_config()

    openai_config = config.get("providers", {}).get("openai", {})
    api_key = os.environ.get("OPENAI_API_KEY", "") or openai_config.get("api_key", "")
    if api_key:
        base_url = openai_config.get("base_url", "https://api.openai.com/v1")
        _providers["openai"] = OpenAIProvider(api_key, base_url=base_url)

    anthropic_config = config.get("providers", {}).get("anthropic", {})
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or anthropic_config.get("api_key", "")
    if anthropic_key:
        base_url = anthropic_config.get("base_url", "https://api.anthropic.com/v1")
        _providers["anthropic"] = AnthropicProvider(anthropic_key, base_url=base_url)

    _serving_url = config.get(
        "serving_url", os.environ.get("OPENCRAB_SERVING_URL", "http://localhost:8081")
    )

    logger.info(
        "intercept_server_started", providers=list(_providers.keys()), serving_url=_serving_url
    )
    yield
    if _storage:
        await _storage.close()


app = FastAPI(title="OpenCrab Intercept", lifespan=lifespan)


async def _should_route_to_distilled(messages: list[dict[str, Any]]) -> tuple[bool, float, str]:
    """Ask the serving layer if the distilled model should handle this request.

    Returns:
        Tuple of (can_handle_locally, confidence, reasoning)
    """
    if not _serving_url:
        return False, 0.0, "Serving URL not configured"

    try:
        import httpx

        payload = {
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{_serving_url}/router/classify", json=payload)
            if response.status_code == 200:
                result = response.json()
                return (
                    result.get("can_handle_locally", False),
                    result.get("confidence", 0.0),
                    result.get("reasoning", ""),
                )
    except Exception as e:
        logger.warning("routing_classifier_error", error=str(e))
        return False, 0.0, f"Routing error: {str(e)}"
    return False, 0.0, "Unexpected routing error"


def _extract_usage_from_response(response: dict[str, Any]) -> tuple[int, int, int]:
    """Extract usage statistics from provider response.

    Returns:
        Tuple of (prompt_tokens, completion_tokens, total_tokens).
    """
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    return prompt_tokens, completion_tokens, total_tokens


def _extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract tool calls from provider response.

    Returns:
        List of tool call dicts with id, name, arguments, or None if no tool calls.
    """
    # OpenAI format: response.choices[0].message.tool_calls
    choices = response.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                    "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", {})),
                }
                for tc in tool_calls
            ]
    # Anthropic format: response.content[].type == "tool_use"
    content = response.get("content", [])
    if isinstance(content, list):
        tool_uses = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_uses.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    }
                )
        if tool_uses:
            return tool_uses
    return None


def _extract_tool_feedback(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Extract tool feedback from user/tool messages following tool calls.

    Looks for messages after tool calls that contain tool results.

    Returns:
        List of tool feedback dicts with tool_call_id, output, feedback_type, or None.
    """
    feedback = []
    for msg in messages:
        role = msg.get("role", "")
        # Tool result can be in user messages with tool_call_id, or role=tool messages
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                # Check if this looks like tool output (simple heuristic)
                feedback.append(
                    {
                        "output": content,
                        "feedback_type": "stdout",
                    }
                )
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "") or msg.get("output", "")
            feedback.append(
                {
                    "tool_call_id": tool_call_id,
                    "output": content,
                    "feedback_type": "stdout",
                }
            )
    return feedback if feedback else None


async def _call_distilled_model(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    request_params: dict[str, Any],
) -> dict[str, Any] | None:
    """Call the distilled model via serving layer."""
    if not _serving_url:
        return None

    try:
        import httpx

        payload = {
            "messages": messages,
            "model": model,
            "stream": stream,
            **request_params,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{_serving_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning("distilled_model_error", error=str(e))
        return None


async def _stream_from_distilled(
    messages: list[dict[str, Any]],
    model: str,
    request_params: dict[str, Any],
) -> AsyncIterator[str]:
    """Stream response from distilled model."""
    if not _serving_url:
        return

    try:
        import httpx

        payload = {
            "messages": messages,
            "model": model,
            "stream": True,
            **request_params,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{_serving_url}/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                    elif line == "":
                        continue
                    else:
                        yield f"{line}\n\n"
    except Exception as e:
        logger.warning("distilled_stream_error", error=str(e))
        yield f'data: {{"error": "{str(e)}"}}\n\n'
    finally:
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    start_time = time.monotonic()
    body = await request.json()
    # Provider is determined by the endpoint, not client input
    provider_name = "openai"

    provider = _providers.get(provider_name)
    if not provider:
        # Fall back to first available provider
        if _providers:
            provider_name, provider = next(iter(_providers.items()))
        else:
            return Response(content="Provider not configured", status_code=500)

    # Validate provider can handle this request format
    if provider and not provider.supports(body):
        # Fall back to first available provider that supports this request
        provider = None
        for name, p in _providers.items():
            if p.supports(body):
                provider = p
                provider_name = name
                break
        if not provider:
            return Response(content="No provider supports this request format", status_code=400)

    messages = body.get("messages", [])
    model = body.get("model", "gpt-4o")
    stream = body.get("stream", False)

    request_params = {k: v for k, v in body.items() if k not in ("messages",)}
    request_params["model"] = model

    routed_to_distilled = False
    routing_confidence = 0.0
    routing_reasoning = ""

    def _store(response=None, error_str=None, is_stream=False):
        latency = (time.monotonic() - start_time) * 1000
        prompt_tokens = completion_tokens = total_tokens = 0
        tool_calls = None
        tool_feedback = None
        if response:
            prompt_tokens, completion_tokens, total_tokens = _extract_usage_from_response(response)
            tool_calls = _extract_tool_calls(response)
        # Extract tool feedback from messages (looks for tool result messages)
        tool_feedback = _extract_tool_feedback(messages)
        return _storage.store(
            provider=provider_name,
            model=model,
            messages=messages,
            request_params=request_params,
            response=response,
            error=error_str,
            routed_to_distilled=routed_to_distilled,
            tool_calls=tool_calls,
            tool_feedback=tool_feedback,
            metadata={
                "routing_confidence": routing_confidence,
                "routing_reasoning": routing_reasoning,
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
        )

    can_handle, routing_confidence, routing_reasoning = await _should_route_to_distilled(messages)
    if can_handle:
        routed_to_distilled = True

        if stream:
            return StreamingResponse(
                _stream_from_distilled(messages, model, request_params),
                media_type="text/event-stream",
            )
        else:
            response = await _call_distilled_model(messages, model, stream, request_params)
            if response:
                if _storage:
                    await _store(response=response)
                return Response(content=json.dumps(response), media_type="application/json")

    if _storage:
        await _store()

    try:
        if stream:
            stream_iter, full_response = await _stream_response(
                provider, messages, model, stream, request_params
            )

            async def store_after_stream() -> None:
                if _storage:
                    await _store(response=full_response)

            from fastapi import BackgroundTasks

            return StreamingResponse(
                stream_iter,
                media_type="text/event-stream",
                background=BackgroundTasks(store_after_stream),
            )
        else:
            response_chunks = []
            async for chunk in provider.chat_completions(
                messages, model=model, stream=False, **request_params
            ):
                response_chunks.append(chunk)
            response = response_chunks[0] if response_chunks else {}
            if _storage:
                await _store(response=response)
            return Response(content=json.dumps(response), media_type="application/json")
    except httpx.HTTPStatusError as e:
        logger.error(
            "provider_http_error",
            provider=provider_name,
            status=e.response.status_code,
            error=str(e),
        )
        if _storage:
            await _store(error_str=str(e))
        # Map HTTP errors per spec: auth failures -> 401, bad gateway -> 502
        if e.response.status_code == 401:
            return Response(
                content=f'{{"error": "Authentication failed", "detail": "{str(e)}"}}',
                status_code=401,
                media_type="application/json",
            )
        return Response(
            content=f'{{"error": "Provider error", "detail": "{str(e)}"}}',
            status_code=502,
            media_type="application/json",
        )
    except httpx.ConnectError as e:
        logger.error("provider_connection_error", provider=provider_name, error=str(e))
        if _storage:
            await _store(error_str=str(e))
        return Response(
            content='{"error": "Bad Gateway", "detail": "Could not connect to provider"}',
            status_code=502,
            media_type="application/json",
        )
    except Exception as e:
        logger.error("provider_error", provider=provider_name, error=str(e))
        if _storage:
            await _store(error_str=str(e))
        return Response(
            content=f'{{"error": "Internal error", "detail": "{str(e)}"}}',
            status_code=500,
            media_type="application/json",
        )


async def _stream_response(
    provider: Any,
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    request_params: dict[str, Any],
) -> tuple[AsyncIterator[str], dict[str, Any]]:
    """Stream provider response and return full response for storage.

    Returns:
        Tuple of (stream iterator, full response dict).
    """
    full_response = {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
    response_id = ""
    response_model = model
    first = True

    async def generate() -> AsyncIterator[str]:
        nonlocal full_response, first, response_id, response_model
        async for chunk in await provider.chat_completions(
            messages, model=model, stream=True, **request_params
        ):
            if first:
                response_id = chunk.get("id", "")
                response_model = chunk.get("model", model)
                full_response = {
                    "id": response_id,
                    "model": response_model,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
                }
                first = False

            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                # Accumulate delta content instead of replacing
                if "content" in delta:
                    full_response["choices"][0]["delta"]["content"] = (
                        full_response["choices"][0]["delta"].get("content", "") + delta["content"]
                    )
                if chunk["choices"][0].get("finish_reason"):
                    full_response["choices"][0]["finish_reason"] = chunk["choices"][0][
                        "finish_reason"
                    ]

            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return generate(), full_response


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    """Anthropic messages API endpoint."""
    start_time = time.monotonic()
    body = await request.json()
    # Provider is determined by the endpoint, not client input
    provider_name = "anthropic"

    provider = _providers.get(provider_name)
    if not provider:
        # Fall back to first available provider
        if _providers:
            provider_name, provider = next(iter(_providers.items()))
        else:
            return Response(content="Provider not configured", status_code=500)

    # Validate provider can handle this request format
    if provider and not provider.supports(body):
        # Fall back to first available provider that supports this request
        provider = None
        for name, p in _providers.items():
            if p.supports(body):
                provider = p
                provider_name = name
                break
        if not provider:
            return Response(content="No provider supports this request format", status_code=400)

    messages_data = body.get("messages", [])
    model = body.get("model", "claude-sonnet-4-20250514")
    stream = body.get("stream", False)

    request_params = {k: v for k, v in body.items() if k not in ("messages",)}
    request_params["model"] = model

    routed_to_distilled = False
    routing_confidence = 0.0
    routing_reasoning = ""

    def _store(response=None, error_str=None):
        latency = (time.monotonic() - start_time) * 1000
        prompt_tokens = completion_tokens = total_tokens = 0
        tool_calls = None
        tool_feedback = None
        if response:
            prompt_tokens, completion_tokens, total_tokens = _extract_usage_from_response(response)
            tool_calls = _extract_tool_calls(response)
        # Extract tool feedback from messages (looks for tool result messages)
        tool_feedback = _extract_tool_feedback(messages_data)
        return _storage.store(
            provider=provider_name,
            model=model,
            messages=messages_data,
            request_params=request_params,
            response=response,
            error=error_str,
            routed_to_distilled=routed_to_distilled,
            tool_calls=tool_calls,
            tool_feedback=tool_feedback,
            metadata={
                "routing_confidence": routing_confidence,
                "routing_reasoning": routing_reasoning,
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
        )

    can_handle, routing_confidence, routing_reasoning = await _should_route_to_distilled(
        messages_data
    )
    if can_handle:
        routed_to_distilled = True

        if stream:
            return StreamingResponse(
                _stream_from_distilled(messages_data, model, request_params),
                media_type="text/event-stream",
            )
        else:
            response = await _call_distilled_model(messages_data, model, stream, request_params)
            if response:
                if _storage:
                    await _store(response=response)
                transformed = provider.transform_response(response)
                return Response(content=json.dumps(transformed), media_type="application/json")

    if _storage:
        await _store()

    try:
        if stream:
            stream_iter, full_response = await _stream_anthropic_response(
                provider, messages_data, model, request_params
            )

            async def store_after_stream() -> None:
                if _storage:
                    await _store(response=full_response)

            from fastapi import BackgroundTasks

            return StreamingResponse(
                stream_iter,
                media_type="text/event-stream",
                background=BackgroundTasks(store_after_stream),
            )
        else:
            response_chunks = []
            async for chunk in provider.chat_completions(
                messages_data, model=model, stream=False, **request_params
            ):
                response_chunks.append(chunk)
            response = response_chunks[0] if response_chunks else {}
            if _storage:
                await _store(response=response)
            transformed = provider.transform_response(response)
            return Response(content=json.dumps(transformed), media_type="application/json")
    except httpx.HTTPStatusError as e:
        logger.error(
            "provider_http_error",
            provider=provider_name,
            status=e.response.status_code,
            error=str(e),
        )
        if _storage:
            await _store(error_str=str(e))
        if e.response.status_code == 401:
            return Response(
                content=f'{{"error": "Authentication failed", "detail": "{str(e)}"}}',
                status_code=401,
                media_type="application/json",
            )
        return Response(
            content=f'{{"error": "Provider error", "detail": "{str(e)}"}}',
            status_code=502,
            media_type="application/json",
        )
    except httpx.ConnectError as e:
        logger.error("provider_connection_error", provider=provider_name, error=str(e))
        if _storage:
            await _store(error_str=str(e))
        return Response(
            content='{"error": "Bad Gateway", "detail": "Could not connect to provider"}',
            status_code=502,
            media_type="application/json",
        )
    except Exception as e:
        logger.error("provider_error", provider=provider_name, error=str(e))
        if _storage:
            await _store(error_str=str(e))
        return Response(
            content=f'{{"error": "Internal error", "detail": "{str(e)}"}}',
            status_code=500,
            media_type="application/json",
        )


async def _stream_anthropic_response(
    provider: Any,
    messages: list[dict[str, Any]],
    model: str,
    request_params: dict[str, Any],
) -> tuple[AsyncIterator[str], dict[str, Any]]:
    """Stream Anthropic-style response and return full response for storage.

    Returns:
        Tuple of (stream iterator, full response dict).
    """
    full_response: dict[str, Any] = {"content": [], "stop_reason": None}
    # Track accumulated text for text deltas and tool_use deltas
    accumulated_text = ""
    accumulated_tool_input = ""
    current_tool_use_id = None

    async def generate() -> AsyncIterator[str]:
        nonlocal full_response, accumulated_text, accumulated_tool_input, current_tool_use_id
        async for chunk in await provider.chat_completions(
            messages, model=model, stream=True, **request_params
        ):
            # Handle different Anthropic streaming event types
            if chunk.get("type") == "content_block_start":
                block = chunk.get("content_block", {})
                if block.get("type") == "text":
                    full_response["content"].append({"type": "text", "text": ""})
                    accumulated_text = ""
                elif block.get("type") == "tool_use":
                    # Start of a tool_use block
                    tool_use = {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": "",
                    }
                    full_response["content"].append(tool_use)
                    current_tool_use_id = block.get("id", "")
                    accumulated_tool_input = ""
            elif chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    accumulated_text += text
                    # Update the last text block with accumulated content
                    if (
                        full_response["content"]
                        and full_response["content"][-1].get("type") == "text"
                    ):
                        full_response["content"][-1]["text"] = accumulated_text
                elif delta.get("type") == "tool_use_delta":
                    # Accumulate tool input
                    partial_json = delta.get("partial_json", "")
                    accumulated_tool_input += partial_json
                    # Update the last tool_use block
                    if (
                        full_response["content"]
                        and full_response["content"][-1].get("type") == "tool_use"
                    ):
                        full_response["content"][-1]["input"] = accumulated_tool_input
            elif chunk.get("type") == "content_block_stop":
                # Reset accumulation for next block if multiple blocks
                accumulated_text = ""
                accumulated_tool_input = ""
                current_tool_use_id = None
            if "stop_reason" in chunk:
                full_response["stop_reason"] = chunk.get("stop_reason")
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return generate(), full_response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/trajectories")
async def list_trajectories(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not _storage:
        return {"error": "Storage not initialized"}
    trajectories = await _storage.list(limit=limit, offset=offset)
    return {"trajectories": [t.to_dict() for t in trajectories]}


@app.get("/trajectories/{trajectory_id}")
async def get_trajectory(trajectory_id: str) -> dict[str, Any]:
    if not _storage:
        return {"error": "Storage not initialized"}
    trajectory = await _storage.get(trajectory_id)
    if not trajectory:
        return {"error": "Trajectory not found"}
    return trajectory.to_dict()


def create_app() -> FastAPI:
    return app
