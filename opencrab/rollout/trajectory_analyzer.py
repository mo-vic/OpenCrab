"""Trajectory analyzer — uses AI to identify mistakes in trajectories."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class TrajectoryAnalyzer:
    """Analyzes trajectories to find mistakes and generate corrections."""

    def __init__(
        self,
        analyzer_api_key: str | None = None,
        analyzer_base_url: str | None = None,
        analyzer_model: str | None = None,
        max_parallel: int = 5,
    ):
        self.api_key = analyzer_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = (analyzer_base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.model = analyzer_model or os.environ.get("ANALYZER_MODEL", "claude-sonnet-4-20250514")
        self.max_parallel = max_parallel

    async def analyze(self, trajectory: dict[str, Any]) -> CorrectedTrajectory:
        """Analyze a trajectory for mistakes.

        Args:
            trajectory: Trajectory dict with messages and response.

        Returns:
            CorrectedTrajectory with corrected messages.

        Per spec: The analyzer outputs a corrected trajectory — same structure as
        the original but with all AI responses corrected. A local diff script
        then compares original vs corrected to generate training samples.
        """
        messages = trajectory.get("messages", [])
        trajectory_id = trajectory.get("id", str(uuid.uuid4()))

        # Per spec: analyzer outputs corrected_messages directly
        structured = await self._call_analyzer(messages)

        # Use corrected_messages from analyzer response directly
        # If analysis failed, keep original messages so trajectory isn't silently dropped
        if structured.get("analysis_failed"):
            corrected_messages = messages
        else:
            corrected_messages = structured.get("corrected_messages", messages)

        # Extract mistake type and description from analysis
        mistake_type = structured.get("mistake_type")
        if mistake_type and mistake_type != "null":
            mistake_type = self._normalize_mistake_type(mistake_type)
        else:
            mistake_type = None

        summary = structured.get("summary", "AI response was corrected")
        if summary == "null" or not summary:
            summary = "AI response was corrected"

        from datetime import datetime

        return CorrectedTrajectory(
            trajectory_id=trajectory_id,
            corrected_messages=corrected_messages,
            analyzed_at=datetime.now(UTC).isoformat(),
            model_used=self.model,
            mistake_type=mistake_type,
            mistake_description=summary,
        )

    async def analyze_batch(self, trajectories: list[dict[str, Any]]) -> list[CorrectedTrajectory]:
        """Analyze multiple trajectories with controlled parallelism.

        Args:
            trajectories: List of trajectory dicts.

        Returns:
            List of CorrectedTrajectory objects.

        Per spec 04-rollout.md: Batch analysis with parallelism control (max_parallel).
        """
        import asyncio

        semaphore = asyncio.Semaphore(self.max_parallel)

        async def analyze_with_semaphore(trajectory: dict[str, Any]) -> CorrectedTrajectory:
            async with semaphore:
                return await self.analyze(trajectory)

        tasks = [analyze_with_semaphore(t) for t in trajectories]
        return await asyncio.gather(*tasks)

    async def _call_analyzer(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call the analyzer AI with structured output request.

        Per spec, the analyzer outputs a corrected trajectory — same structure as
        the original but with all AI responses corrected.
        """
        prompt = self._build_analysis_prompt(messages)

        # Load system prompt from external file
        from .prompts import get_analyzer_system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": get_analyzer_system_prompt(),
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4096,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.post(
                    f"{self.base_url}/messages", json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    text = result["content"][0]["text"]
                    return self._parse_json_response(text)
        except Exception as e:
            logger.error("analyzer_error", error=str(e))
            return {"corrected_messages": messages, "mistake_type": None, "summary": f"Error: {e}"}

    def _build_analysis_prompt(self, messages: list[dict[str, Any]]) -> str:
        return f"""You are an expert at analyzing AI conversation trajectories to correct mistakes.
Given a conversation trajectory, your task is to output the corrected version of the conversation based on feedback.

Mistake types to look for:
1. Factual Error: User directly or indirectly corrects the AI's factual statement.
2. Self-Correction: AI figures out the correct answer after multiple turns of probing.
3. Tool Construction Error: Tool call fails, subsequent retry succeeds.
4. Domain Knowledge Gap: Consistent wrong answers on a specific topic.
5. Reasoning Error: Flawed logical step in multi-step reasoning.

Output the corrected conversation. Remove all intermediate negotiation turns (mistakes, corrections, retries) — keep only the user's question and the corrected AI response. Keep the same structure as the original trajectory.

Conversation Trajectory:
{self._format_messages(messages)}

Corrected Conversation:"""

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON response from analyzer AI.

        Returns corrected_messages format per spec.
        """
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("analyzer_json_parse_failed", text=text[:200])
            return {
                "corrected_messages": [],
                "analysis_failed": True,
                "mistake_type": None,
                "summary": "Parse error - analysis failed",
            }

    # Mapping from old types to spec-compliant types
    _MISTAKE_TYPE_MAP = {
        "factual": "factual",
        "self_correction": "self_correction",
        "tool_error": "tool_construction",
        "domain_knowledge": "domain_knowledge",
        "reasoning": "reasoning_error",
    }

    def _normalize_mistake_type(self, mistake_type: str) -> str:
        """Normalize mistake type to spec-compliant format."""
        return self._MISTAKE_TYPE_MAP.get(mistake_type, mistake_type)

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for the analyzer prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Handle tool calls in messages
            if msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    func_name = tc.get("name", "unknown")
                    args = tc.get("arguments", {})
                    formatted.append(f"{role.upper()} (tool_call): {func_name}({args})")
            elif msg.get("role") == "tool":
                # Tool result messages
                output = msg.get("content", "") or msg.get("output", "")
                formatted.append(f"{role.upper()}: {output}")
            else:
                formatted.append(f"{role.upper()}: {content}")
        return "\n".join(formatted)


class AnalysisResult:
    """Result of trajectory analysis."""

    def __init__(
        self,
        original_messages: list[dict[str, Any]],
        original_response: str,
        analysis: str,
        mistakes: list[Mistake],
        corrected_response: str | None = None,
    ):
        self.original_messages = original_messages
        self.original_response = original_response
        self.analysis = analysis
        self.mistakes = mistakes
        self.corrected_response = corrected_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_messages": self.original_messages,
            "original_response": self.original_response,
            "analysis": self.analysis,
            "mistakes": [m.to_dict() for m in self.mistakes],
            "corrected_response": self.corrected_response,
        }


class Mistake:
    """Represents a identified mistake in a trajectory."""

    def __init__(self, type: str, description: str, correction: str | None = None):
        self.type = type
        self.description = description
        self.correction = correction

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "description": self.description, "correction": self.correction}


class CorrectedTrajectory:
    """Represents a corrected trajectory with all AI responses fixed."""

    def __init__(
        self,
        trajectory_id: str,
        corrected_messages: list[dict[str, Any]],
        analyzed_at: str,
        model_used: str,
        mistake_type: str | None = None,
        mistake_description: str | None = None,
    ):
        self.trajectory_id = trajectory_id
        self.corrected_messages = corrected_messages
        self.analyzed_at = analyzed_at
        self.model_used = model_used
        self.mistake_type = mistake_type
        self.mistake_description = mistake_description

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "corrected_messages": self.corrected_messages,
            "analyzed_at": self.analyzed_at,
            "model_used": self.model_used,
            "mistake_type": self.mistake_type,
            "mistake_description": self.mistake_description,
        }
