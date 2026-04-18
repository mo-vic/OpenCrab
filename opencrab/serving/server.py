"""Serving layer — serves distilled model with routing."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .routers import Classifier, get_classifier

logger = structlog.get_logger()

# Annealing configuration
_ANNEALING_STATE_FILE = Path("./.opencrab_annealing_state.json")
_initial_confidence_threshold: float = 0.95  # Start very conservative (cold start)
_final_confidence_threshold: float = 0.6  # Allow more distilled handling as model improves
_annealing_training_steps: int = 1000  # Steps over which to anneal

_distilled_model_url: str | None = None
_distilled_api_key: str | None = None
_model_path: str | None = None
_classifier: Classifier | None = None
_model_loaded: bool = False
_llm: Any = None
_sampling_params: Any = None
_sglang_engine: Any = None
_serving_backend: str = "none"  # "sglang", "vllm", or "none"
_start_time: float = time.time()
_last_model_path: str | None = None


def _load_annealing_state() -> dict[str, Any]:
    """Load annealing state from disk."""
    if _ANNEALING_STATE_FILE.exists():
        try:
            with open(_ANNEALING_STATE_FILE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return _default_annealing_state()


def _default_annealing_state() -> dict[str, Any]:
    """Default annealing state for cold start."""
    return {
        "training_steps": 0,
        "last_updated": time.time(),
    }


def _save_annealing_state(state: dict[str, Any]) -> None:
    """Persist annealing state to disk."""
    state["last_updated"] = time.time()
    with open(_ANNEALING_STATE_FILE, "w") as f:
        json.dump(state, f)


def _get_confidence_threshold(state: dict[str, Any]) -> float:
    """Calculate current confidence threshold based on annealing schedule.

    Linear interpolation from initial to final threshold over training steps.
    Cold start: high threshold (conservative) -> Final: lower threshold (trust distilled more)
    """
    global _initial_confidence_threshold, _final_confidence_threshold, _annealing_training_steps
    steps = state.get("training_steps", 0)
    progress = min(steps / _annealing_training_steps, 1.0)
    threshold = _initial_confidence_threshold - (
        (_initial_confidence_threshold - _final_confidence_threshold) * progress
    )
    return threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        _distilled_model_url, \
        _distilled_api_key, \
        _model_path, \
        _classifier, \
        _model_loaded, \
        _llm, \
        _sampling_params, \
        _sglang_engine, \
        _serving_backend, \
        _last_model_path, \
        _initial_confidence_threshold, \
        _final_confidence_threshold, \
        _annealing_training_steps

    from opencrab.config import get

    _distilled_model_url = get(
        "serving",
        "distilled_url",
        default=os.environ.get("DISTILLED_MODEL_URL", "http://localhost:8000/v1/chat/completions"),
    )
    _distilled_api_key = get("serving", "api_key", default=os.environ.get("DISTILLED_API_KEY"))
    _model_path = get(
        "serving", "model_path", default=os.environ.get("MODEL_PATH", "./model_output")
    )

    classifier_name = get(
        "serving", "router", "mode", default=os.environ.get("ROUTER_CLASSIFIER", "self_classifier")
    )
    _classifier = get_classifier(classifier_name)

    # Annealing configuration (overridden by OPENCRAB_SERVING_ANNEALING_* env vars)
    _initial_confidence_threshold = get("serving", "annealing", "initial_threshold", default=0.95)
    _final_confidence_threshold = get("serving", "annealing", "final_threshold", default=0.6)
    _annealing_training_steps = get("serving", "annealing", "training_steps", default=1000)

    _llm, _sampling_params, _sglang_engine, _model_loaded, _serving_backend = await _load_model()
    _last_model_path = _model_path

    if not _model_loaded:
        logger.error(
            "model_loading_failed",
            model_path=_model_path,
            message="No model loaded. SGLANG and vLLM are not available or model path is invalid. "
            "Install SGLANG or vLLM, or set DISTILLED_MODEL_URL to proxy to an external model.",
        )
    else:
        logger.info(
            "serving_server_started",
            model_url=_distilled_model_url,
            model_path=_model_path,
            backend=_serving_backend,
        )
    yield


app = FastAPI(title="OpenCrab Serving", lifespan=lifespan)


async def _load_model() -> tuple[Any, Any, Any, bool, str]:
    """Load the distilled model using SGLANG (primary) or vLLM (fallback).

    Returns:
        Tuple of (llm, sampling_params, sglang_engine, loaded, backend_name)
    """
    global _llm, _sampling_params, _sglang_engine

    if not _model_path:
        return None, None, None, False, "none"

    # Try SGLANG first (primary per spec)
    try:
        from sglang import SamplingParams, SGLangEngine

        logger.info("loading_model_with_sglang", model_path=_model_path)
        _sglang_engine = SGLangEngine(_model_path, tp_size=1)
        _sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
        return None, _sampling_params, _sglang_engine, True, "sglang"
    except ImportError:
        logger.warning("sglang_not_available_trying_vllm")

    # Fall back to vLLM
    try:
        from vllm import LLM, SamplingParams

        logger.info("loading_model_with_vllm", model_path=_model_path)
        _llm = LLM(model=_model_path)
        _sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
        return _llm, _sampling_params, None, True, "vllm"
    except ImportError:
        logger.warning("vllm_not_available")
        return None, None, None, False, "none"


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "distilled")
    stream = body.get("stream", False)

    if _model_loaded and _model_path:
        return await _serve_locally(messages, model, stream, body)
    elif _distilled_model_url:
        return await _proxy_to_external(messages, model, stream, body)
    else:
        return JSONResponse(content={"error": "No model configured"})


async def _serve_locally(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    body: dict[str, Any],
) -> StreamingResponse | JSONResponse:
    """Serve request using local model (SGLANG or vLLM)."""
    global _llm, _sampling_params, _sglang_engine, _serving_backend

    try:
        prompt = _build_prompt_from_messages(messages)

        if _serving_backend == "sglang" and _sglang_engine:
            return await _serve_with_sglang(prompt, model, stream)
        elif _serving_backend == "vllm" and _llm:
            return await _serve_with_vllm(prompt, model, stream)
        else:
            return JSONResponse(content={"error": "Model not loaded"})
    except Exception as e:
        logger.error("local_model_error", backend=_serving_backend, error=str(e))
        return JSONResponse(content={"error": str(e)})


async def _serve_with_sglang(
    prompt: str,
    model: str,
    stream: bool,
) -> StreamingResponse | JSONResponse:
    """Serve request using SGLANG engine."""
    global _sglang_engine, _sampling_params

    if stream:
        # Run blocking generate() in thread pool to avoid blocking the event loop
        output = await asyncio.to_thread(_sglang_engine.generate, prompt, _sampling_params)
        return StreamingResponse(
            _stream_sglang_output(output.text),
            media_type="text/event-stream",
        )
    else:
        output = await asyncio.to_thread(_sglang_engine.generate, prompt, _sampling_params)
        return JSONResponse(content=_format_sglang_response(output))


async def _serve_with_vllm(
    prompt: str,
    model: str,
    stream: bool,
) -> StreamingResponse | JSONResponse:
    """Serve request using vLLM engine."""
    global _llm, _sampling_params

    if stream:
        # Run blocking generate() in thread pool to avoid blocking the event loop
        outputs = await asyncio.to_thread(_llm.generate, [prompt], _sampling_params)
        return StreamingResponse(
            _stream_vllm_output(outputs[0].outputs[0].text),
            media_type="text/event-stream",
        )
    else:
        outputs = await asyncio.to_thread(_llm.generate, [prompt], _sampling_params)
        response = _format_vllm_response(outputs[0])
        return JSONResponse(content=response)


async def _proxy_to_external(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    body: dict[str, Any],
) -> StreamingResponse | JSONResponse:
    """Proxy request to external distilled model URL."""
    headers = {"Authorization": f"Bearer {_distilled_api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": stream}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:

                async def stream_from_model() -> AsyncIterator[str]:
                    async with client.stream(
                        "POST", _distilled_model_url, json=payload, headers=headers
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                yield f"{line}\n\n"
                            elif line == "":
                                continue
                            else:
                                yield f"{line}\n\n"

                return StreamingResponse(stream_from_model(), media_type="text/event-stream")
            else:
                async with client.post(
                    _distilled_model_url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    return await response.json()
    except Exception as e:
        logger.error("distilled_model_error", error=str(e))
        return JSONResponse(content={"error": str(e)})


def _build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Build prompt string from messages using ChatML format.

    Handles tool_calls by formatting them alongside content in assistant messages.
    """
    import json

    # ChatML format as used by Qwen and other models
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            if tool_calls:
                # Format tool calls as JSON alongside content
                tc_json = json.dumps(tool_calls)
                if content:
                    parts.append(f"<|im_start|>assistant\n{content}\n[{tc_json}]<|im_end|>")
                else:
                    parts.append(f"<|im_start|>assistant\n[{tc_json}]<|im_end|>")
            elif content:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                parts.append("<|im_start|>assistant")
        elif role == "tool":
            # Tool result messages
            output = msg.get("content", "") or msg.get("output", "")
            tool_call_id = msg.get("tool_call_id", "")
            parts.append(
                f"<|im_start|>tool\n<|tool_call_id|>{tool_call_id}</|tool_call_id|>\n{output}<|im_end|>"
            )
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


def _stream_vllm_output(text: str) -> AsyncIterator[str]:
    """Stream text output as SSE, chunking by words for efficiency."""
    import json
    import re

    chunk_id = f"chatcmpl-{os.urandom(12).hex()}"
    # Split on whitespace but preserve the whitespace in chunks
    words = re.split(r"(\s+)", text)
    for i, chunk in enumerate(words):
        if not chunk:
            continue
        is_last = i == len(words) - 1
        delta = {"content": chunk}
        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": "distilled",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": None if not is_last else "stop"}
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_sglang_output(text: str) -> AsyncIterator[str]:
    """Stream SGLANG output as SSE (same format as vLLM)."""
    return _stream_vllm_output(text)


def _format_sglang_response(output: Any) -> dict[str, Any]:
    """Format SGLANG output as OpenAI-style response."""
    import uuid

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": "distilled",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output.text},
                "finish_reason": "stop",
            }
        ],
    }


def _format_vllm_response(output: Any) -> dict[str, Any]:
    """Format vLLM output as OpenAI-style response."""
    import uuid

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": "distilled",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output.outputs[0].text},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/router/classify")
async def router_classify(request: Request) -> JSONResponse:
    """Classify whether a request should be handled by distilled model or general AI.

    Applies confidence annealing: during cold start, threshold is high (conservative),
    and decreases over time as the distilled model improves.
    """
    body = await request.json()
    messages = body.get("messages", [])
    context = body.get("context")

    if not _classifier:
        return JSONResponse(
            content={
                "can_handle_locally": False,
                "confidence": 0.0,
                "reasoning": "Classifier not configured",
            }
        )

    try:
        decision, raw_confidence = await _classifier.classify(messages, context)

        # Apply annealing threshold
        annealing_state = _load_annealing_state()
        threshold = _get_confidence_threshold(annealing_state)

        # Apply threshold: only route to distilled if classifier confidence exceeds threshold
        can_handle = (decision == "distilled") and (raw_confidence >= threshold)
        final_confidence = raw_confidence if can_handle else (1.0 - raw_confidence)

        reasoning = _get_reasoning_for_decision(decision, messages, raw_confidence, threshold)
        return JSONResponse(
            content={
                "can_handle_locally": can_handle,
                "confidence": final_confidence,
                "reasoning": reasoning,
            }
        )
    except Exception as e:
        logger.error("classification_error", error=str(e))
        return JSONResponse(
            content={
                "can_handle_locally": False,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
            }
        )


def _get_reasoning_for_decision(
    decision: str,
    messages: list[dict[str, Any]],
    raw_confidence: float = 0.0,
    threshold: float = 0.95,
) -> str:
    """Generate reasoning text for routing decision."""
    last_msg = messages[-1].get("content", "")[:50] if messages else ""
    if decision == "distilled":
        if raw_confidence < threshold:
            return f"Distilled model suggested but confidence {raw_confidence:.2f} below threshold {threshold:.2f}."
        return f"Distilled model confident ({raw_confidence:.2f}) and exceeds threshold ({threshold:.2f})."
    elif decision == "general":
        return "This query requires general AI capabilities beyond the distilled model's training."
    else:
        return f"Fallback routing due to uncertainty for: {last_msg}..."


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "opencrab-distilled-model",
                "object": "model",
                "created": int(_start_time),
                "owned_by": "opencrab",
                "model": "opencrab-distilled-model",
            }
        ],
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy" if _model_loaded or _distilled_model_url else "degraded",
        "models": {
            "distilled_model": {
                "loaded": _model_loaded,
                "model_id": _model_path or "none",
                "backend": _serving_backend,
            },
            "router": {
                "loaded": _classifier is not None,
                "mode": _classifier.mode if _classifier else "none",
            },
        },
        "uptime_seconds": int(time.time() - _start_time),
    }


@app.post("/admin/reload")
async def reload_model() -> dict[str, Any]:
    """Reload the distilled model gracefully if the model path has changed."""
    global \
        _model_loaded, \
        _llm, \
        _sampling_params, \
        _sglang_engine, \
        _serving_backend, \
        _last_model_path, \
        _model_path

    from opencrab.config import get

    current_model_path = get(
        "serving", "model_path", default=os.environ.get("MODEL_PATH", "./model_output")
    )

    # Skip reload if model path hasn't changed and model is already loaded
    if current_model_path == _last_model_path and _model_loaded:
        logger.info("model_reload_skipped_no_change", model_path=current_model_path)
        return {"status": "unchanged", "model_loaded": _model_loaded, "backend": _serving_backend}

    logger.info("model_reload_triggered", old_path=_last_model_path, new_path=current_model_path)

    # Load new model first, keep old model running until new one is ready
    (
        new_llm,
        new_sampling_params,
        new_sglang_engine,
        new_model_loaded,
        new_serving_backend,
    ) = await _load_model()

    if new_model_loaded:
        # Only unload old model after new one is fully loaded
        _llm = new_llm
        _sampling_params = new_sampling_params
        _sglang_engine = new_sglang_engine
        _model_loaded = True
        _serving_backend = new_serving_backend
        _last_model_path = current_model_path
        logger.info("model_reload_completed", backend=_serving_backend)
        return {"status": "reloaded", "model_loaded": _model_loaded, "backend": _serving_backend}
    else:
        # New model failed to load - keep old model running
        logger.error("model_reload_failed_keeping_old_model", backend=_serving_backend)
        return {
            "status": "reload_failed",
            "model_loaded": _model_loaded,
            "backend": _serving_backend,
        }


@app.post("/admin/annealing/advance")
async def advance_annealing(steps: int = 1) -> dict[str, Any]:
    """Advance the annealing schedule by the given number of training steps.

    Called automatically after training completes to lower the routing threshold,
    allowing more queries to be handled by the distilled model as it improves.
    """
    state = _load_annealing_state()
    state["training_steps"] = state.get("training_steps", 0) + steps
    _save_annealing_state(state)
    threshold = _get_confidence_threshold(state)
    logger.info("annealing_advanced", training_steps=state["training_steps"], threshold=threshold)
    return {
        "status": "advanced",
        "training_steps": state["training_steps"],
        "current_threshold": threshold,
    }


@app.post("/admin/annealing/reset")
async def reset_annealing() -> dict[str, Any]:
    """Reset annealing state back to cold start.

    Should be called when deploying a new model to start fresh with
    conservative routing thresholds.
    """
    global _initial_confidence_threshold
    state = _default_annealing_state()
    _save_annealing_state(state)
    logger.info("annealing_reset")
    return {
        "status": "reset",
        "training_steps": 0,
        "current_threshold": _initial_confidence_threshold,
    }


@app.get("/admin/annealing")
async def get_annealing_state() -> dict[str, Any]:
    """Get current annealing state and threshold."""
    global _initial_confidence_threshold, _final_confidence_threshold, _annealing_training_steps
    state = _load_annealing_state()
    threshold = _get_confidence_threshold(state)
    return {
        "training_steps": state.get("training_steps", 0),
        "current_threshold": threshold,
        "initial_threshold": _initial_confidence_threshold,
        "final_threshold": _final_confidence_threshold,
        "annealing_steps_total": _annealing_training_steps,
    }


def create_app() -> FastAPI:
    return app
