"""Serving layer routers (distilled model + classifier)."""

from .routers import (
    CLASSIFIERS,
    ROUTERS,
    Classifier,
    CoTClassifier,
    HeadClassifier,
    Router,
    SelfClassifier,
    StandaloneClassifier,
    # Expose inference helpers for testing
    _format_conversation_context,
    _parse_cot_routing_decision,
    _parse_self_routing_decision,
    get_classifier,
    get_router,
)

__all__ = [
    "Router",
    "ROUTERS",
    "get_router",
    "Classifier",
    "CLASSIFIERS",
    "get_classifier",
    "SelfClassifier",
    "CoTClassifier",
    "HeadClassifier",
    "StandaloneClassifier",
    # Expose inference helpers for testing
    "_format_conversation_context",
    "_parse_self_routing_decision",
    "_parse_cot_routing_decision",
]
