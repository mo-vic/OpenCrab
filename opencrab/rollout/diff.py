"""Diff logic to generate training samples from original vs corrected trajectories."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .trajectory_analyzer import CorrectedTrajectory


class Differ:
    """Pure diffing logic — compares original and corrected trajectories."""

    def generate_training_samples(
        self,
        original_messages: list[dict[str, Any]],
        corrected_messages: list[dict[str, Any]],
        trajectory_id: str,
        mistake_type: str | None = None,
        mistake_description: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training samples from diff of original vs corrected.

        Args:
            original_messages: Original trajectory messages.
            corrected_messages: Corrected trajectory messages.
            trajectory_id: ID of the trajectory for reference.
            mistake_type: Type of mistake detected (from analyzer).
            mistake_description: Description of the mistake.

        Returns:
            List of training samples (one per changed message).
        """
        samples = []
        timestamp = datetime.now(UTC).isoformat()

        for i, (orig, corr) in enumerate(zip(original_messages, corrected_messages, strict=True)):
            if orig.get("role") != corr.get("role"):
                # Role changed - this is a significant difference
                sample = self._create_sample(
                    original_messages[: i + 1],
                    corrected_messages[: i + 1],
                    trajectory_id,
                    timestamp,
                    mistake_type,
                    mistake_description,
                )
                samples.append(sample)
            elif orig.get("role") == "assistant":
                orig_content = orig.get("content", "") or ""
                corr_content = corr.get("content", "") or ""
                orig_tool_calls = orig.get("tool_calls")
                corr_tool_calls = corr.get("tool_calls")

                if orig_content != corr_content or orig_tool_calls != corr_tool_calls:
                    sample = self._create_sample(
                        original_messages[: i + 1],
                        corrected_messages[: i + 1],
                        trajectory_id,
                        timestamp,
                        mistake_type,
                        mistake_description,
                    )
                    samples.append(sample)
            elif orig != corr:
                # Other message types (e.g., user with different content)
                sample = self._create_sample(
                    original_messages[: i + 1],
                    corrected_messages[: i + 1],
                    trajectory_id,
                    timestamp,
                    mistake_type,
                    mistake_description,
                )
                samples.append(sample)

        return samples

    def _create_sample(
        self,
        original: list[dict[str, Any]],
        corrected: list[dict[str, Any]],
        trajectory_id: str,
        timestamp: str,
        mistake_type: str | None = None,
        mistake_description: str | None = None,
    ) -> dict[str, Any]:
        """Create a single training sample from corrected conversation up to this point."""
        system_message = None
        corrected_copy = list(corrected)
        if corrected_copy and corrected_copy[0].get("role") == "system":
            system_message = corrected_copy[0].get("content")
            corrected_copy = corrected_copy[1:]

        training_messages = []
        if system_message:
            training_messages.append({"role": "system", "content": system_message})
        training_messages.extend(corrected_copy)

        # Get original response for comparison
        original_response = ""
        for msg in reversed(original):
            if msg.get("role") == "assistant":
                original_response = msg.get("content", "") or ""
                break

        # Get corrected response
        corrected_response = ""
        for msg in reversed(corrected_copy):
            if msg.get("role") == "assistant":
                corrected_response = msg.get("content", "") or ""
                break

        return {
            "messages": training_messages,
            "system_message": system_message,
            "conversation": [m for m in training_messages if m.get("role") != "system"],
            "original_response": original_response,
            "corrected_response": corrected_response,
            "mistake_type": mistake_type or "correction",
            "mistake_description": mistake_description or "AI response was corrected",
            "metadata": {
                "trajectory_id": trajectory_id,
                "timestamp": timestamp,
            },
        }

    def generate_router_samples(
        self,
        samples: list[dict[str, Any]],
        trajectory_id: str,
        original_messages: list[dict[str, Any]] | None = None,
        corrected_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Generate router training sample from a trajectory.

        Args:
            samples: Training samples generated from this trajectory.
            trajectory_id: ID of the trajectory.
            original_messages: Original trajectory messages (for negative samples).
            corrected_messages: Corrected trajectory messages (for negative samples).

        Returns:
            Router training sample dict with should_handle_locally field.
            Returns None only if no messages available at all.
        """
        last_messages = None
        last_user_msg = None

        if samples:
            # Positive sample: use training sample messages
            # should_handle_locally=True only if original and corrected differ
            orig_response = samples[0].get("original_response", "")
            corr_response = samples[0].get("corrected_response", "")
            should_handle = orig_response != corr_response

            last_messages = samples[-1].get("messages", [])
            for msg in reversed(last_messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            return {
                "query": last_user_msg or "",
                "should_handle_locally": should_handle,
                "context": {
                    "trajectory_id": trajectory_id,
                    "mistake_found": should_handle,
                },
            }
        elif original_messages and corrected_messages:
            # Negative sample: no correction needed (original was correct)
            # Detect if original and corrected are actually different
            orig_assistant = self._get_last_assistant_response(original_messages)
            corr_assistant = self._get_last_assistant_response(corrected_messages)

            if orig_assistant != corr_assistant:
                # The diff found a difference but no training sample was generated.
                # This can happen when the message structure changed (e.g., role changed,
                # tool_calls changed). Treat as positive: distilled model should handle
                # this type of query since it required correction.
                return {
                    "query": self._get_last_user_message(original_messages) or "",
                    "should_handle_locally": True,
                    "context": {
                        "trajectory_id": trajectory_id,
                        "mistake_found": True,
                    },
                }
            else:
                # No actual correction - original response was correct
                return {
                    "query": self._get_last_user_message(original_messages) or "",
                    "should_handle_locally": False,
                    "context": {
                        "trajectory_id": trajectory_id,
                        "mistake_found": False,
                    },
                }
        else:
            # No messages available - can't generate sample
            return None

    def _get_last_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Get the last user message content from a message list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "") or ""
        return ""

    def _get_last_assistant_response(self, messages: list[dict[str, Any]]) -> str:
        """Get the last assistant response content from a message list."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "") or ""
        return ""

    def process_trajectory_pair(
        self,
        original: dict[str, Any],
        corrected: CorrectedTrajectory,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Process a pair of original and corrected trajectories.

        Args:
            original: Original trajectory dict.
            corrected: CorrectedTrajectory object.

        Returns:
            Tuple of (training_sample_dicts, router_sample_dict).
        """
        original_messages = original.get("messages", [])
        corrected_messages = corrected.corrected_messages
        trajectory_id = corrected.trajectory_id

        training_samples = self.generate_training_samples(
            original_messages,
            corrected_messages,
            trajectory_id,
            corrected.mistake_type,
            corrected.mistake_description,
        )

        router_sample = self.generate_router_samples(
            training_samples,
            trajectory_id,
            original_messages=original_messages,
            corrected_messages=corrected_messages,
        )

        return training_samples, router_sample


def diff_trajectories(
    original: dict[str, Any],
    corrected: CorrectedTrajectory,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Generate training samples and router sample from original vs corrected.

    Args:
        original: Original trajectory dict.
        corrected: CorrectedTrajectory object.

    Returns:
        Tuple of (training_sample_dicts, router_sample_dict).
    """
    differ = Differ()
    return differ.process_trajectory_pair(original, corrected)
