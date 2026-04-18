"""Trajectory storage for intercept layer."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Trajectory(Base):
    __tablename__ = "trajectories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    messages = Column(JSON, nullable=False)
    request_params = Column(JSON, nullable=False)
    response = Column(JSON, nullable=True)
    tool_calls = Column(JSON, nullable=True)
    tool_feedback = Column(JSON, nullable=True)
    trajectory_metadata = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    routed_to_distilled = Column(Boolean, nullable=True)
    # Usage statistics per spec schema
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    # Latency tracking
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "request_params": self.request_params,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "tool_feedback": self.tool_feedback,
            "metadata": self.trajectory_metadata,
            "error": self.error,
            "routed_to_distilled": self.routed_to_distilled,
            "usage": {
                "prompt_tokens": self.prompt_tokens or 0,
                "completion_tokens": self.completion_tokens or 0,
                "total_tokens": self.total_tokens or 0,
            },
            "latency_ms": self.latency_ms or 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrajectoryStore:
    """Storage for captured trajectories."""

    def __init__(self, database_url: str = "sqlite+aiosqlite:///./trajectories.db"):
        self.engine = create_async_engine(database_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession)

    async def init(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def store(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        request_params: dict[str, Any],
        response: dict[str, Any] | None = None,
        error: str | None = None,
        routed_to_distilled: bool | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_feedback: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> Trajectory:
        """Store a trajectory."""
        trajectory = Trajectory(
            provider=provider,
            model=model,
            messages=messages,
            request_params=request_params,
            response=response,
            error=error,
            routed_to_distilled=routed_to_distilled,
            tool_calls=tool_calls,
            tool_feedback=tool_feedback,
            trajectory_metadata=metadata,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )
        async with self.session_factory() as session:
            session.add(trajectory)
            await session.commit()
            await session.refresh(trajectory)
        return trajectory

    async def get(self, trajectory_id: str) -> Trajectory | None:
        """Retrieve a trajectory by ID."""
        async with self.session_factory() as session:
            result = await session.get(Trajectory, trajectory_id)
            return result

    async def list(self, limit: int = 100, offset: int = 0) -> list[Trajectory]:
        """List trajectories with pagination."""
        async with self.session_factory() as session:
            from sqlalchemy import select

            stmt = (
                select(Trajectory)
                .order_by(Trajectory.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def query(
        self,
        provider: str | None = None,
        model: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        routed_to_distilled: bool | None = None,
        search_text: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Trajectory]:
        """Query trajectories with filters.

        Args:
            provider: Filter by provider (e.g., 'openai', 'anthropic').
            model: Filter by model name.
            start_time: Filter trajectories created after this time.
            end_time: Filter trajectories created before this time.
            routed_to_distilled: Filter by whether routed to distilled model.
            search_text: Full-text search in messages content.
            limit: Maximum number of results to return.
            offset: Number of results to skip for pagination.

        Returns:
            List of matching trajectories.
        """
        async with self.session_factory() as session:
            from sqlalchemy import and_, select

            conditions = []

            if provider:
                conditions.append(Trajectory.provider == provider)
            if model:
                conditions.append(Trajectory.model == model)
            if start_time:
                conditions.append(Trajectory.created_at >= start_time)
            if end_time:
                conditions.append(Trajectory.created_at <= end_time)
            if routed_to_distilled is not None:
                conditions.append(Trajectory.routed_to_distilled == routed_to_distilled)
            if search_text:
                # Search in messages JSON using JSON_each for SQLite
                # This searches for the text in the serialized JSON
                from sqlalchemy import String, cast

                conditions.append(cast(Trajectory.messages, String).like(f"%{search_text}%"))

            stmt = select(Trajectory)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.order_by(Trajectory.created_at.desc()).limit(limit).offset(offset)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()
