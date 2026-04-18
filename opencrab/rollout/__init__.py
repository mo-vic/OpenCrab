"""Rollout layer — extracts training examples from trajectories."""

from .diff import Differ, diff_trajectories
from .extractor import RoutingSample, TrainingSample, TrainingSampleExtractor
from .trajectory_analyzer import AnalysisResult, CorrectedTrajectory, Mistake, TrajectoryAnalyzer
from .transforms.huggingface import HuggingFaceTransform
from .transforms.jsonl import JSONLTransform

__all__ = [
    "TrajectoryAnalyzer",
    "AnalysisResult",
    "Mistake",
    "CorrectedTrajectory",
    "TrainingSampleExtractor",
    "TrainingSample",
    "RoutingSample",
    "JSONLTransform",
    "HuggingFaceTransform",
    "Differ",
    "diff_trajectories",
]
