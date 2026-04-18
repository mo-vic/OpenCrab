"""HuggingFace datasets transform for rollout layer."""

from __future__ import annotations

from pathlib import Path

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    Dataset = None
    DatasetDict = None

from ..extractor import RoutingSample, TrainingSample


class HuggingFaceTransform:
    """Transforms training samples to HuggingFace datasets format."""

    def to_huggingface(
        self, samples: list[TrainingSample], routing_samples: list[RoutingSample] | None = None
    ) -> DatasetDict | None:
        """Convert samples to HuggingFace DatasetDict.

        Args:
            samples: List of TrainingSample objects.
            routing_samples: Optional list of RoutingSample objects.

        Returns:
            DatasetDict with 'train' and optionally 'routing' splits.
        """
        if Dataset is None:
            raise ImportError("datasets package not installed. Run: pip install datasets")

        data = {"messages": [], "system_message": [], "corrected_response": [], "mistake_type": []}

        for sample in samples:
            data["messages"].append(sample.conversation)
            data["system_message"].append(sample.system_message)
            data["corrected_response"].append(sample.corrected_response)
            data["mistake_type"].append(sample.mistake_type)

        dataset = Dataset.from_dict(data)

        if routing_samples:
            routing_data = {
                "query": [s.query for s in routing_samples],
                "should_handle_locally": [s.should_handle_locally for s in routing_samples],
            }
            routing_dataset = Dataset.from_dict(routing_data)
            return DatasetDict({"train": dataset, "routing": routing_dataset})

        return DatasetDict({"train": dataset})

    def save(self, dataset: DatasetDict, output_dir: str | Path) -> None:
        """Save dataset to disk.

        Args:
            dataset: HuggingFace DatasetDict.
            output_dir: Output directory path.
        """
        dataset.save_to_disk(output_dir)
