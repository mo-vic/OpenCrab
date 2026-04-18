"""Tests for serving layer."""

import pytest
from fastapi.testclient import TestClient


class TestServingServer:
    """Tests for serving server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for serving server."""
        from opencrab.serving.server import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data

    def test_list_models_endpoint(self, client):
        """List models should return model information."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "opencrab-distilled-model"

    def test_annealing_state_endpoint(self, client):
        """Annealing state endpoint should return current state."""
        response = client.get("/admin/annealing")

        assert response.status_code == 200
        data = response.json()
        assert "training_steps" in data
        assert "current_threshold" in data
        assert "initial_threshold" in data
        assert "final_threshold" in data


class TestClassifiers:
    """Tests for routing classifiers."""

    def test_self_classifier_name(self):
        """Self classifier should have correct mode."""
        from opencrab.serving.routers import SelfClassifier

        classifier = SelfClassifier()
        assert classifier.mode == "self_classifier"

    def test_cot_classifier_name(self):
        """CoT classifier should have correct mode."""
        from opencrab.serving.routers import CoTClassifier

        classifier = CoTClassifier()
        assert classifier.mode == "cot_classifier"

    def test_head_classifier_name(self):
        """Head classifier should have correct mode."""
        from opencrab.serving.routers import HeadClassifier

        classifier = HeadClassifier()
        assert classifier.mode == "head_classifier"

    def test_standalone_classifier_name(self):
        """Standalone classifier should have correct mode."""
        from opencrab.serving.routers import StandaloneClassifier

        classifier = StandaloneClassifier()
        assert classifier.mode == "standalone_classifier"

    def test_get_classifier_invalid_name(self):
        """Get classifier should raise error for invalid name."""
        from opencrab.serving.routers import get_classifier

        with pytest.raises(ValueError, match="Unknown router"):
            get_classifier("invalid_classifier")

    def test_get_classifier_valid_names(self):
        """Get classifier should return correct classifier for valid names."""
        from opencrab.serving.routers import get_classifier

        classifiers = [
            "self_classifier",
            "cot_classifier",
            "head_classifier",
            "standalone_classifier",
        ]
        for name in classifiers:
            cls = get_classifier(name)
            assert cls is not None
            assert cls.mode == name


class TestInference:
    """Tests for inference helpers."""

    def test_format_conversation_context_empty(self):
        """Should handle empty conversation."""
        from opencrab.serving.routers import _format_conversation_context

        result = _format_conversation_context([])
        assert result == "(empty conversation)"

    def test_format_conversation_context_single_message(self):
        """Should format single message correctly."""
        from opencrab.serving.routers import _format_conversation_context

        messages = [{"role": "user", "content": "hello"}]
        result = _format_conversation_context(messages)
        assert "USER: hello" in result

    def test_format_conversation_context_multiple_messages(self):
        """Should format multiple messages correctly."""
        from opencrab.serving.routers import _format_conversation_context

        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _format_conversation_context(messages)
        assert "SYSTEM: you are helpful" in result
        assert "USER: hello" in result
        assert "ASSISTANT: hi there" in result

    def test_parse_self_routing_decision_yes(self):
        """Should parse 'yes' as distilled."""
        from opencrab.serving.routers import _parse_self_routing_decision

        assert _parse_self_routing_decision("yes") == "distilled"
        assert _parse_self_routing_decision("YES") == "distilled"
        assert _parse_self_routing_decision("Yes, I can handle it") == "distilled"

    def test_parse_self_routing_decision_no(self):
        """Should parse 'no' as general."""
        from opencrab.serving.routers import _parse_self_routing_decision

        assert _parse_self_routing_decision("no") == "general"
        assert _parse_self_routing_decision("NO") == "general"
        assert _parse_self_routing_decision("No, I cannot handle this") == "general"

    def test_parse_self_routing_decision_fallback(self):
        """Should fall back to general for ambiguous input."""
        from opencrab.serving.routers import _parse_self_routing_decision

        assert _parse_self_routing_decision("maybe") == "general"
        assert _parse_self_routing_decision("") == "general"

    def test_parse_cot_routing_decision_explicit(self):
        """Should parse explicit markers in CoT response."""
        from opencrab.serving.routers import _parse_cot_routing_decision

        assert _parse_cot_routing_decision("final answer: distilled") == "distilled"
        assert _parse_cot_routing_decision("answer: general") == "general"

    def test_parse_cot_routing_decision_markers(self):
        """Should parse marker keywords in CoT response."""
        from opencrab.serving.routers import _parse_cot_routing_decision

        assert _parse_cot_routing_decision("I [can_handle] this") == "distilled"
        assert _parse_cot_routing_decision("I [needs_general] this") == "general"


class TestAnnealingLogic:
    """Tests for annealing logic in serving."""

    def test_annealing_threshold_calculation(self):
        """Should calculate threshold based on training steps."""
        from opencrab.serving.server import _default_annealing_state, _get_confidence_threshold

        # Cold start (0 steps)
        state = _default_annealing_state()
        threshold = _get_confidence_threshold(state)
        assert threshold == 0.95  # Initial high threshold

        # Mid training (500 steps out of 1000)
        state = {"training_steps": 500, "last_updated": 0}
        threshold = _get_confidence_threshold(state)
        assert abs(threshold - 0.775) < 0.001  # Linear interpolation (use abs for float comparison)

        # Fully trained (1000 steps)
        state = {"training_steps": 1000, "last_updated": 0}
        threshold = _get_confidence_threshold(state)
        assert threshold == 0.6  # Final low threshold

    def test_annealing_threshold_clamped(self):
        """Should clamp threshold at final value after full training."""
        from opencrab.serving.server import _get_confidence_threshold

        state = {"training_steps": 2000, "last_updated": 0}  # Beyond training
        threshold = _get_confidence_threshold(state)
        assert threshold == 0.6  # Should not go below final


class TestPromptBuilding:
    """Tests for ChatML prompt building."""

    def test_build_prompt_from_messages(self):
        """Should build properly formatted ChatML prompt."""
        from opencrab.serving.server import _build_prompt_from_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        prompt = _build_prompt_from_messages(messages)

        assert "<|im_start|>system" in prompt
        assert "You are helpful." in prompt
        assert "<|im_start|>user" in prompt
        assert "What is 2+2?" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "4" in prompt
        assert prompt.endswith("<|im_start|>assistant")
