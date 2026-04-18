"""Training sample extractor — constructs TrainingSample/RoutingSample from trajectories."""

from __future__ import annotations

from typing import Any

from .diff import Differ
from .trajectory_analyzer import CorrectedTrajectory


class TrainingSampleExtractor:
    """Extracts training samples from analyzed trajectories using diff."""

    def __init__(self):
        self._differ = Differ()

    def extract(
        self, original_trajectory: dict[str, Any], corrected_trajectory: CorrectedTrajectory
    ) -> list[TrainingSample]:
        """Extract training samples by diffing original vs corrected.

        Args:
            original_trajectory: Original trajectory dict with messages.
            corrected_trajectory: CorrectedTrajectory from analyzer.

        Returns:
            List of TrainingSample objects ready for training.
        """
        sample_dicts, _ = self._differ.process_trajectory_pair(
            original_trajectory, corrected_trajectory
        )

        return [TrainingSample.from_dict(s) for s in sample_dicts]

    def extract_for_routing(
        self, original_trajectory: dict[str, Any], corrected_trajectory: CorrectedTrajectory
    ) -> RoutingSample | None:
        """Extract routing training sample.

        Args:
            original_trajectory: Original trajectory dict.
            corrected_trajectory: CorrectedTrajectory from analyzer.

        Returns:
            RoutingSample for training the router.
        """
        _, router_dict = self._differ.process_trajectory_pair(
            original_trajectory, corrected_trajectory
        )

        if not router_dict:
            return None

        return RoutingSample.from_dict(router_dict)


class TrainingSample:
    """A training sample for fine-tuning the distilled model."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        system_message: str | None,
        conversation: list[dict[str, Any]],
        original_response: str,
        corrected_response: str,
        mistake_type: str,
        mistake_description: str,
    ):
        self.messages = messages
        self.system_message = system_message
        self.conversation = conversation
        self.original_response = original_response
        self.corrected_response = corrected_response
        self.mistake_type = mistake_type
        self.mistake_description = mistake_description

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingSample:
        return cls(
            messages=data["messages"],
            system_message=data["system_message"],
            conversation=data["conversation"],
            original_response=data["original_response"],
            corrected_response=data["corrected_response"],
            mistake_type=data["mistake_type"],
            mistake_description=data["mistake_description"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "system_message": self.system_message,
            "conversation": self.conversation,
            "original_response": self.original_response,
            "corrected_response": self.corrected_response,
            "mistake_type": self.mistake_type,
            "mistake_description": self.mistake_description,
        }

    def to_training_format(self) -> dict[str, Any]:
        """Convert to model training format.

        Includes metadata with trajectory_id and timestamp per spec.
        """
        training_messages = []
        if self.system_message:
            training_messages.append({"role": "system", "content": self.system_message})
        training_messages.extend(self.conversation)
        # Replace last assistant message with corrected_response
        # (conversation already contains assistant messages; we need to update them with corrected content)
        replaced = False
        for i in range(len(training_messages) - 1, -1, -1):
            if training_messages[i].get("role") == "assistant":
                training_messages[i] = {"role": "assistant", "content": self.corrected_response}
                replaced = True
                break
        # If no assistant found (edge case), append corrected response
        if not replaced:
            training_messages.append({"role": "assistant", "content": self.corrected_response})

        # Build metadata from stored metadata dict
        meta = self.to_dict()
        metadata = {
            "trajectory_id": meta.get("metadata", {}).get("trajectory_id", ""),
            "timestamp": meta.get("metadata", {}).get("timestamp", ""),
        }
        return {"messages": training_messages, "metadata": metadata}


class RoutingSample:
    """A training sample for training the router."""

    def __init__(
        self, query: str, should_handle_locally: bool, context: dict[str, Any] | None = None
    ):
        self.query = query
        self.should_handle_locally = should_handle_locally
        self.context = context or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingSample:
        return cls(
            query=data["query"],
            should_handle_locally=data["should_handle_locally"],
            context=data.get("context"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "should_handle_locally": self.should_handle_locally,
            "context": self.context,
        }
