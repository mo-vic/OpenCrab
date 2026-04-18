"""Base provider interface for intercept layer."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class Provider(ABC):
    """Abstract base class for API providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""

    @abstractmethod
    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a chat completions request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model identifier.
            stream: Whether to stream response.
            **kwargs: Additional provider-specific params.

        Returns:
            Async iterator of response dicts (single item for non-streaming).
        """

    @abstractmethod
    def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Transform incoming request to provider format.

        Args:
            request: Request in OpenCrab's internal format.

        Returns:
            Request dict in provider's native format.
        """

    @abstractmethod
    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Transform provider response to OpenCrab's internal format.

        Args:
            response: Response in provider's native format.

        Returns:
            Response dict in OpenCrab's internal format.
        """

    def supports(self, request: dict[str, Any]) -> bool:
        """Check if this provider can handle the request.

        Args:
            request: Request dict.

        Returns:
            True if this provider supports the request.
        """
        return True
