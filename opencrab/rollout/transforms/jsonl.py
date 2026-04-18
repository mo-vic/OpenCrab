"""JSONL transform for rollout layer."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiofiles

from ..extractor import RoutingSample, TrainingSample


class JSONLTransform:
    """Transforms training samples to JSONL format."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    async def write_samples(self, samples: list[TrainingSample]) -> None:
        """Write training samples to JSONL file.

        Args:
            samples: List of TrainingSample objects.
        """
        async with aiofiles.open(self.output_path, "a") as f:
            for sample in samples:
                await f.write(json.dumps(sample.to_training_format()) + "\n")

    async def write_routing_samples(self, samples: list[RoutingSample]) -> None:
        """Write routing samples to JSONL file.

        Args:
            samples: List of RoutingSample objects.
        """
        async with aiofiles.open(self.output_path, "a") as f:
            for sample in samples:
                await f.write(json.dumps(sample.to_dict()) + "\n")

    async def read_samples(self) -> AsyncIterator[dict[str, Any]]:
        """Read training samples from JSONL file.

        Yields:
            Training sample dicts.
        """
        if not self.output_path.exists():
            return

        async with aiofiles.open(self.output_path) as f:
            async for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
