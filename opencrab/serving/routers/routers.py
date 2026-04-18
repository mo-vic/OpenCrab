"""Router classifiers and inference helpers for routing decisions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# Router model path for head_classifier and standalone_classifier
ROUTER_MODEL_PATH = os.environ.get("ROUTER_MODEL_PATH", "./router_model")
STANDALONE_MODEL_URL = os.environ.get("STANDALONE_MODEL_URL", "http://localhost:8002")

DISTILLED_MODEL_URL = os.environ.get(
    "DISTILLED_MODEL_URL", "http://localhost:8000/v1/chat/completions"
)
DISTILLED_API_KEY = os.environ.get("DISTILLED_API_KEY", "")


def _format_conversation_context(messages: list[dict[str, Any]]) -> str:
    """Format full conversation context for classification prompts.

    Args:
        messages: Full list of conversation messages.

    Returns:
        Formatted string representation of the conversation.
    """
    if not messages:
        return "(empty conversation)"

    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts) if parts else "(no content)"


async def run_self_classification(
    messages: list[dict[str, Any]], context: dict[str, Any] | None = None
) -> str:
    """Run self-classification via distilled model.

    The distilled model itself determines whether it has learned to handle
    this type of query based on past training.

    Args:
        messages: Conversation messages (full context).
        context: Optional context dict with user preferences, trajectory_id, etc.

    Returns:
        Routing decision: 'distilled', 'general', or 'fallback'.
    """
    conversation_context = _format_conversation_context(messages)

    # Include context info in prompt if available
    context_info = ""
    if context:
        if context.get("user_preferences"):
            context_info = f"\nUser preferences: {context['user_preferences']}"
        if context.get("trajectory_id"):
            context_info += f"\nTrajectory ID: {context['trajectory_id']}"

    prompt = f"""Based on your training, can you handle this query well? Answer only 'yes' or 'no'.

Conversation:
{conversation_context}{context_info}

Query: {messages[-1]["content"] if messages else ""}"""

    headers = {"Authorization": f"Bearer {DISTILLED_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "distilled",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.post(DISTILLED_MODEL_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return _parse_self_routing_decision(content)
    except Exception:
        return "general"


async def run_cot_classification(
    messages: list[dict[str, Any]], context: dict[str, Any] | None = None
) -> str:
    """Run chain-of-thought classification via distilled model.

    The distilled model explicitly reasons about whether it has seen
    similar examples in training before making a decision.

    Args:
        messages: Conversation messages (full context).
        context: Optional context dict with user preferences, trajectory_id, etc.

    Returns:
        Routing decision: 'distilled', 'general', or 'fallback'.
    """
    conversation_context = _format_conversation_context(messages)

    # Include context info in prompt if available
    context_info = ""
    if context:
        if context.get("user_preferences"):
            context_info = f"\nUser preferences: {context['user_preferences']}"
        if context.get("trajectory_id"):
            context_info += f"\nTrajectory ID: {context['trajectory_id']}"

    prompt = f"""Think step by step: Have you seen similar queries in your training that you learned to handle correctly?
Consider the full conversation context and what the query is asking for.
Answer with 'distilled' if confident you can handle it, 'general' otherwise.

Conversation:
{conversation_context}{context_info}

Query: {messages[-1]["content"] if messages else ""}"""

    headers = {"Authorization": f"Bearer {DISTILLED_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "distilled",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 256,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.post(DISTILLED_MODEL_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return _parse_cot_routing_decision(content)
    except Exception:
        return "general"


def _parse_self_routing_decision(content: str) -> str:
    """Parse routing decision from self-classifier response."""
    content_lower = content.lower().strip()
    if "yes" in content_lower:
        return "distilled"
    elif "no" in content_lower:
        return "general"
    return "general"


def _parse_cot_routing_decision(content: str) -> str:
    """Parse routing decision from chain-of-thought classifier response.

    Looks for the final answer in the content, handling reasoning patterns.
    """
    content_lower = content.lower().strip()

    if "final answer: distilled" in content_lower or "answer: distilled" in content_lower:
        return "distilled"
    if "final answer: general" in content_lower or "answer: general" in content_lower:
        return "general"

    if "[can_handle]" in content_lower or "can handle" in content_lower:
        return "distilled"
    if "[needs_general]" in content_lower or "needs general" in content_lower:
        return "general"

    if "distilled" in content_lower and "general" not in content_lower:
        return "distilled"
    if "general" in content_lower:
        return "general"

    return "general"


class Router(ABC):
    """Base class for routing classifiers (per spec Router interface)."""

    def __init__(self):
        self._head_model: Any = None
        self._head_tokenizer: Any = None
        self._head_loaded: bool = False
        self._base_model: Any = None
        self._base_model_loaded: bool = False

    @property
    @abstractmethod
    def mode(self) -> str:
        """Router mode name."""

    @abstractmethod
    async def classify(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> tuple[str, float]:
        """Classify a request.

        Args:
            messages: Conversation messages.
            context: Optional context (user preferences, etc.).

        Returns:
            Tuple of (routing_decision, confidence) where:
            - routing_decision: 'distilled', 'general', or 'fallback'
            - confidence: float between 0.0 and 1.0
        """

    def _load_head_classifier(self) -> tuple[Any, Any, bool]:
        """Load head classifier model lazily.

        Loads the custom head model saved by HeadClassifierTrainingPipeline.
        The head model is a torch.nn.Sequential that takes hidden states as input.
        Also loads the base model (frozen) to extract features for the head.
        """
        if self._head_loaded:
            return self._head_model, self._head_tokenizer, self._head_loaded

        try:
            import json

            import torch
            from transformers import AutoModel, AutoTokenizer

            head_model_path = Path(ROUTER_MODEL_PATH)

            if not head_model_path.exists():
                self._head_loaded = False
                return None, None, False

            if not self._base_model_loaded:
                base_model_name = os.environ.get(
                    "DISTILLED_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"
                )
                self._base_model = AutoModel.from_pretrained(base_model_name)
                self._base_model.eval()
                for param in self._base_model.parameters():
                    param.requires_grad = False
                self._base_model_loaded = True
                logger.info("head_classifier_base_model_loaded", base_model=base_model_name)

            head_state_dict = torch.load(head_model_path / "head_model.pt", weights_only=True)

            with open(head_model_path / "head_config.json") as f:
                head_config = json.load(f)

            if head_config.get("head_type") == "head_classifier":
                hidden_size = head_config["hidden_size"]
                self._head_model = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden_size // 2, 2),
                )
            else:
                self._head_model = torch.nn.Linear(head_config["hidden_size"], 2)

            self._head_model.load_state_dict(head_state_dict)
            self._head_model.eval()

            self._head_tokenizer = AutoTokenizer.from_pretrained(head_model_path)
            self._head_loaded = True
            logger.info("head_classifier_loaded", path=ROUTER_MODEL_PATH)
        except Exception as e:
            logger.warning("head_classifier_load_failed", error=str(e))
            self._head_loaded = False

        return self._head_model, self._head_tokenizer, self._head_loaded


class SelfClassifier(Router):
    """Self-classifier: distilled model decides if it can handle the query."""

    @property
    def mode(self) -> str:
        return "self_classifier"

    async def classify(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> tuple[str, float]:
        decision = await run_self_classification(messages, context)
        if decision in ("distilled", "general"):
            confidence = 0.75
        else:
            confidence = 0.3
        return decision, confidence


class CoTClassifier(Router):
    """Chain-of-Thought classifier: uses reasoning to decide routing."""

    @property
    def mode(self) -> str:
        return "cot_classifier"

    async def classify(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> tuple[str, float]:
        decision = await run_cot_classification(messages, context)
        if decision in ("distilled", "general"):
            confidence = 0.80
        else:
            confidence = 0.25
        return decision, confidence


class HeadClassifier(Router):
    """Head classifier: lightweight trained head for fast routing decisions."""

    @property
    def mode(self) -> str:
        return "head_classifier"

    async def classify(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> tuple[str, float]:
        head_model, tokenizer, loaded = self._load_head_classifier()
        if not loaded or self._base_model is None:
            logger.warning("head_classifier_not_available_falling_back")
            decision = await run_self_classification(messages)
            return decision, 0.4

        try:
            import torch

            # Use full conversation context, not just the last message
            conversation_context = _format_conversation_context(messages)
            inputs = tokenizer(
                conversation_context, return_tensors="pt", truncation=True, max_length=512
            )

            with torch.no_grad():
                base_outputs = self._base_model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                if hasattr(base_outputs, "last_hidden_state"):
                    features = base_outputs.last_hidden_state[:, 0, :]
                else:
                    features = base_outputs[0][:, 0, :]

                logits = head_model(features)
                probs = torch.softmax(logits, dim=-1)
                confidence, prediction = torch.max(probs, dim=-1)
                confidence = confidence.item()
                prediction = prediction.item()

            decision = "distilled" if prediction == 1 else "general"
            return decision, confidence
        except Exception as e:
            logger.error("head_classification_error", error=str(e))
            return "general", 0.5


class StandaloneClassifier(Router):
    """Standalone classifier: independent model for routing only."""

    @property
    def mode(self) -> str:
        return "standalone_classifier"

    async def classify(
        self, messages: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> tuple[str, float]:
        # Use full conversation context, not just the last message
        conversation_context = _format_conversation_context(messages)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{STANDALONE_MODEL_URL}/classify",
                    json={"messages": messages, "context": conversation_context},
                )
                response.raise_for_status()
                result = response.json()
                decision = result.get("decision", "general")
                confidence = result.get("confidence", 0.6)
                return decision, confidence
        except Exception as e:
            logger.warning("standalone_classifier_error", error=str(e))
            return "general", 0.25


ROUTERS: dict[str, type[Router]] = {
    "self_classifier": SelfClassifier,
    "cot_classifier": CoTClassifier,
    "head_classifier": HeadClassifier,
    "standalone_classifier": StandaloneClassifier,
}


def get_router(name: str) -> Router:
    """Get a router by name."""
    cls = ROUTERS.get(name)
    if not cls:
        raise ValueError(f"Unknown router: {name}")
    return cls()


# Backwards compatibility aliases
Classifier = Router
CLASSIFIERS = ROUTERS
get_classifier = get_router
