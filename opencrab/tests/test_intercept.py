"""Tests for intercept layer."""

import pytest
from fastapi.testclient import TestClient


class TestInterceptProviders:
    """Tests for provider adapters."""

    def test_openai_provider_supports_valid_request(self):
        """OpenAI provider should support valid chat completions request."""
        from opencrab.intercept.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        request = {"messages": [{"role": "user", "content": "hello"}]}

        assert provider.supports(request) is True

    def test_openai_provider_rejects_empty_messages(self):
        """OpenAI provider should reject request with empty messages."""
        from opencrab.intercept.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        request = {"messages": []}

        assert provider.supports(request) is False

    def test_openai_provider_rejects_non_dict_request(self):
        """OpenAI provider should reject non-dict request."""
        from opencrab.intercept.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")

        assert provider.supports("not a dict") is False
        assert provider.supports(None) is False

    def test_openai_provider_transform_request(self):
        """OpenAI provider should transform request correctly."""
        from opencrab.intercept.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        }

        result = provider.transform_request(request)

        assert result["model"] == "gpt-4"
        assert result["messages"] == request["messages"]
        assert result["temperature"] == 0.7
        assert result["stream"] is False

    def test_anthropic_provider_supports_valid_request(self):
        """Anthropic provider should support valid messages API request."""
        from opencrab.intercept.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        request = {"messages": [{"role": "user", "content": "hello"}]}

        assert provider.supports(request) is True

    def test_anthropic_provider_transform_request(self):
        """Anthropic provider should transform request correctly."""
        from opencrab.intercept.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hello"}],
        }

        result = provider.transform_request(request)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["stream"] is False


class TestTrajectoryStorage:
    """Tests for trajectory storage."""

    @pytest.fixture
    async def storage(self):
        """Create in-memory SQLite storage for testing."""
        from opencrab.intercept.storage import TrajectoryStore

        store = TrajectoryStore(database_url="sqlite+aiosqlite:///:memory:")
        await store.init()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_trajectory(self, storage):
        """Should store and retrieve a trajectory."""
        trajectory = await storage.store(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            request_params={"model": "gpt-4"},
            response={"id": "test-id", "choices": []},
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency_ms=100.0,
        )

        assert trajectory.id is not None
        assert trajectory.provider == "openai"
        assert trajectory.model == "gpt-4"

        retrieved = await storage.get(trajectory.id)
        assert retrieved is not None
        assert retrieved.id == trajectory.id
        assert retrieved.prompt_tokens == 10
        assert retrieved.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_list_trajectories(self, storage):
        """Should list trajectories with pagination."""
        for i in range(5):
            await storage.store(
                provider="openai",
                model="gpt-4",
                messages=[{"role": "user", "content": f"msg {i}"}],
                request_params={},
            )

        results = await storage.list(limit=3, offset=0)
        assert len(results) == 3

        results = await storage.list(limit=3, offset=3)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_provider(self, storage):
        """Should filter trajectories by provider."""
        await storage.store(provider="openai", model="gpt-4", messages=[], request_params={})
        await storage.store(provider="anthropic", model="claude", messages=[], request_params={})
        await storage.store(provider="openai", model="gpt-3.5", messages=[], request_params={})

        results = await storage.query(provider="openai")
        assert len(results) == 2
        assert all(t.provider == "openai" for t in results)

    @pytest.mark.asyncio
    async def test_query_by_routed_to_distilled(self, storage):
        """Should filter trajectories by routing decision."""
        await storage.store(
            provider="openai",
            model="gpt-4",
            messages=[],
            request_params={},
            routed_to_distilled=True,
        )
        await storage.store(
            provider="openai",
            model="gpt-4",
            messages=[],
            request_params={},
            routed_to_distilled=False,
        )

        results = await storage.query(routed_to_distilled=True)
        assert len(results) == 1
        assert results[0].routed_to_distilled is True

    @pytest.mark.asyncio
    async def test_trajectory_to_dict(self, storage):
        """Should convert trajectory to dict with correct structure."""
        trajectory = await storage.store(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            request_params={"model": "gpt-4"},
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency_ms=150.5,
        )

        d = trajectory.to_dict()

        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4"
        assert d["messages"] == [{"role": "user", "content": "hello"}]
        assert d["usage"]["prompt_tokens"] == 10
        assert d["usage"]["completion_tokens"] == 20
        assert d["usage"]["total_tokens"] == 30
        assert d["latency_ms"] == 150.5
        assert "id" in d
        assert "created_at" in d


class TestInterceptServer:
    """Tests for intercept server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for intercept server."""
        import asyncio

        from opencrab.intercept.server import app

        # Create in-memory storage for testing
        async def init_storage():
            global _storage
            from opencrab.intercept.storage import TrajectoryStore

            _storage = TrajectoryStore(database_url="sqlite+aiosqlite:///:memory:")
            await _storage.init()

        # Run the async init
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_storage())
        loop.close()

        with TestClient(app) as client:
            yield client

    def test_health_endpoint(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_list_trajectories_empty(self, client):
        """List trajectories should return empty list when no trajectories."""
        response = client.get("/trajectories")

        assert response.status_code == 200
        assert "trajectories" in response.json()

    def test_get_trajectory_not_found(self, client):
        """Get trajectory should return error for non-existent ID."""
        response = client.get("/trajectories/nonexistent-id")

        assert response.status_code == 200
        assert "error" in response.json()
