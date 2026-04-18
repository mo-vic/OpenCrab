"""Tests for rollout layer."""

import pytest


class TestTrajectoryAnalyzer:
    """Tests for trajectory analyzer."""

    def test_normalize_mistake_type_factual(self):
        """Should normalize factual mistake type."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        assert analyzer._normalize_mistake_type("factual") == "factual"

    def test_normalize_mistake_type_tool_error(self):
        """Should normalize tool_error to tool_construction."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        assert analyzer._normalize_mistake_type("tool_error") == "tool_construction"

    def test_normalize_mistake_type_reasoning(self):
        """Should normalize reasoning to reasoning_error."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        assert analyzer._normalize_mistake_type("reasoning") == "reasoning_error"

    def test_normalize_mistake_type_unknown(self):
        """Should pass through unknown mistake types."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        assert analyzer._normalize_mistake_type("unknown_type") == "unknown_type"

    def test_format_messages_with_tool_calls(self):
        """Should format tool call messages correctly."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "tool_calls": [{"name": "run_command", "arguments": {"cmd": "ls"}}],
            },
        ]

        result = analyzer._format_messages(messages)
        assert "USER: Run ls" in result
        assert "ASSISTANT (tool_call): run_command" in result

    def test_format_messages_with_tool_result(self):
        """Should format tool result messages correctly."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        messages = [{"role": "tool", "content": "file1.txt\nfile2.txt"}]

        result = analyzer._format_messages(messages)
        assert "TOOL: file1.txt" in result

    def test_parse_json_response_clean(self):
        """Should parse clean JSON response."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        text = '{"corrected_messages": [], "mistake_type": null, "summary": "test"}'

        result = analyzer._parse_json_response(text)
        assert result["corrected_messages"] == []
        assert result["summary"] == "test"

    def test_parse_json_response_with_json_prefix(self):
        """Should parse JSON with markdown code fence."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        text = '```json\n{"corrected_messages": [], "mistake_type": null, "summary": "test"}\n```'

        result = analyzer._parse_json_response(text)
        assert result["corrected_messages"] == []

    def test_parse_json_response_fallback(self):
        """Should fall back to finding JSON in text."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        text = 'Here is the response: {"corrected_messages": [], "mistake_type": null}'

        result = analyzer._parse_json_response(text)
        assert "corrected_messages" in result

    def test_parse_json_response_invalid(self):
        """Should return default on parse failure."""
        from opencrab.rollout.trajectory_analyzer import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        text = "not valid json at all"

        result = analyzer._parse_json_response(text)
        assert result["corrected_messages"] == []
        assert result["mistake_type"] is None


class TestCorrectedTrajectory:
    """Tests for CorrectedTrajectory model."""

    def test_corrected_trajectory_creation(self):
        """Should create CorrectedTrajectory with all fields."""
        from opencrab.rollout.trajectory_analyzer import CorrectedTrajectory

        ct = CorrectedTrajectory(
            trajectory_id="traj_123",
            corrected_messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            analyzed_at="2026-04-19T10:00:00Z",
            model_used="claude-sonnet-4-20250514",
            mistake_type="factual",
            mistake_description="User corrected AI",
        )

        assert ct.trajectory_id == "traj_123"
        assert len(ct.corrected_messages) == 2
        assert ct.model_used == "claude-sonnet-4-20250514"
        assert ct.mistake_type == "factual"

    def test_corrected_trajectory_to_dict(self):
        """Should convert to dict correctly."""
        from opencrab.rollout.trajectory_analyzer import CorrectedTrajectory

        ct = CorrectedTrajectory(
            trajectory_id="traj_123",
            corrected_messages=[{"role": "user", "content": "hi"}],
            analyzed_at="2026-04-19T10:00:00Z",
            model_used="claude",
        )

        d = ct.to_dict()
        assert d["trajectory_id"] == "traj_123"
        assert d["corrected_messages"] == [{"role": "user", "content": "hi"}]
        assert d["analyzed_at"] == "2026-04-19T10:00:00Z"
        assert d["model_used"] == "claude"


class TestDiff:
    """Tests for diff logic."""

    def test_generate_training_samples_no_changes(self):
        """Should return empty list when no changes."""
        from opencrab.rollout.diff import Differ

        differ = Differ()
        original = [{"role": "user", "content": "hello"}]
        corrected = [{"role": "user", "content": "hello"}]

        samples = differ.generate_training_samples(original, corrected, "traj_123")

        assert len(samples) == 0

    def test_generate_training_samples_assistant_change(self):
        """Should generate sample when assistant response changes."""
        from opencrab.rollout.diff import Differ

        differ = Differ()
        original = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "wrong"}]
        corrected = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "correct"}]

        samples = differ.generate_training_samples(original, corrected, "traj_123")

        assert len(samples) == 1
        assert samples[0]["original_response"] == "wrong"
        assert samples[0]["corrected_response"] == "correct"
        assert samples[0]["mistake_type"] == "correction"

    def test_generate_training_samples_preserves_system(self):
        """Should preserve system message in training sample."""
        from opencrab.rollout.diff import Differ

        differ = Differ()
        original = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "wrong"},
        ]
        corrected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "correct"},
        ]

        samples = differ.generate_training_samples(original, corrected, "traj_123")

        assert len(samples) == 1
        assert samples[0]["system_message"] == "You are helpful"
        assert samples[0]["messages"][0]["role"] == "system"

    def test_generate_router_samples(self):
        """Should generate router sample from training samples."""
        from opencrab.rollout.diff import Differ

        differ = Differ()
        training_samples = [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "original_response": "wrong",
                "corrected_response": "correct",
            }
        ]

        router_sample = differ.generate_router_samples(training_samples, "traj_123")

        assert router_sample is not None
        assert router_sample["should_handle_locally"] is True
        assert router_sample["query"] == "hi"
        assert router_sample["context"]["trajectory_id"] == "traj_123"

    def test_generate_router_samples_empty(self):
        """Should return None when no training samples."""
        from opencrab.rollout.diff import Differ

        differ = Differ()
        router_sample = differ.generate_router_samples([], "traj_123")

        assert router_sample is None

    def test_process_trajectory_pair(self):
        """Should process original and corrected trajectory pair."""
        from opencrab.rollout.diff import Differ
        from opencrab.rollout.trajectory_analyzer import CorrectedTrajectory

        differ = Differ()
        original = {
            "id": "traj_123",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "wrong"},
            ],
        }
        corrected = CorrectedTrajectory(
            trajectory_id="traj_123",
            corrected_messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "correct"},
            ],
            analyzed_at="2026-04-19T10:00:00Z",
            model_used="claude",
        )

        training_samples, router_sample = differ.process_trajectory_pair(original, corrected)

        assert len(training_samples) == 1
        assert training_samples[0]["corrected_response"] == "correct"
        assert router_sample["should_handle_locally"] is True


class TestExtractor:
    """Tests for training sample extractor."""

    def test_training_sample_from_dict(self):
        """Should create TrainingSample from dict."""
        from opencrab.rollout.extractor import TrainingSample

        data = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "correct"},
            ],
            "system_message": None,
            "conversation": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "wrong"},
            ],
            "original_response": "wrong",
            "corrected_response": "correct",
            "mistake_type": "factual",
            "mistake_description": "User corrected",
        }

        sample = TrainingSample.from_dict(data)

        assert sample.conversation == data["conversation"]
        assert sample.original_response == "wrong"
        assert sample.corrected_response == "correct"

    def test_training_sample_to_dict(self):
        """Should convert TrainingSample to dict."""
        from opencrab.rollout.extractor import TrainingSample

        sample = TrainingSample(
            messages=[{"role": "user", "content": "hi"}],
            system_message=None,
            conversation=[{"role": "user", "content": "hi"}],
            original_response="wrong",
            corrected_response="correct",
            mistake_type="factual",
            mistake_description="User corrected",
        )

        d = sample.to_dict()

        assert d["original_response"] == "wrong"
        assert d["corrected_response"] == "correct"
        assert d["mistake_type"] == "factual"

    def test_training_sample_to_training_format(self):
        """Should convert to model training format correctly."""
        from opencrab.rollout.extractor import TrainingSample

        sample = TrainingSample(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "5"},
            ],
            system_message="You are helpful",
            conversation=[
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "5"},
            ],
            original_response="5",
            corrected_response="4",
            mistake_type="factual",
            mistake_description="User corrected",
        )

        result = sample.to_training_format()

        # Should contain system + conversation, with assistant message replaced with corrected_response
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"
        # Last message should have corrected_response, not original
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "assistant"
        assert last_msg["content"] == "4"

    def test_routing_sample_from_dict(self):
        """Should create RoutingSample from dict."""
        from opencrab.rollout.extractor import RoutingSample

        data = {
            "query": "What's 2+2?",
            "should_handle_locally": True,
            "context": {"trajectory_id": "traj_123"},
        }

        sample = RoutingSample.from_dict(data)

        assert sample.query == "What's 2+2?"
        assert sample.should_handle_locally is True
        assert sample.context["trajectory_id"] == "traj_123"

    def test_routing_sample_to_dict(self):
        """Should convert RoutingSample to dict."""
        from opencrab.rollout.extractor import RoutingSample

        sample = RoutingSample(
            query="What's 2+2?", should_handle_locally=True, context={"trajectory_id": "traj_123"}
        )

        d = sample.to_dict()

        assert d["query"] == "What's 2+2?"
        assert d["should_handle_locally"] is True


class TestJSONLTransform:
    """Tests for JSONL transform."""

    @pytest.mark.asyncio
    async def test_write_and_read_samples(self, tmp_path):
        """Should write and read samples correctly."""
        from opencrab.rollout.extractor import TrainingSample
        from opencrab.rollout.transforms.jsonl import JSONLTransform

        output_path = tmp_path / "test.jsonl"
        transform = JSONLTransform(output_path)

        samples = [
            TrainingSample(
                messages=[{"role": "user", "content": "hi"}],
                system_message=None,
                conversation=[{"role": "user", "content": "hi"}],
                original_response="wrong",
                corrected_response="correct",
                mistake_type="factual",
                mistake_description="User corrected",
            )
        ]

        await transform.write_samples(samples)

        # Read back
        read_samples = [s async for s in transform.read_samples()]
        assert len(read_samples) == 1

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        """Should handle nonexistent file gracefully."""
        from opencrab.rollout.transforms.jsonl import JSONLTransform

        output_path = tmp_path / "nonexistent.jsonl"
        transform = JSONLTransform(output_path)

        read_samples = [s async for s in transform.read_samples()]
        assert len(read_samples) == 0
