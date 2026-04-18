"""Tests for training layer."""

import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch", reason="torch not installed")  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_training_config_defaults(self):
        """Should have correct default values."""
        from opencrab.training.pipeline import TrainingConfig

        config = TrainingConfig()

        assert config.base_model == os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        assert config.method == "lora"
        assert config.num_epochs == 3
        assert config.batch_size == 1
        assert config.learning_rate == 1e-4

    def test_training_config_custom(self):
        """Should accept custom values."""
        from opencrab.training.pipeline import TrainingConfig

        config = TrainingConfig(
            base_model="Qwen/Qwen3-7B",
            method="qlora",
            num_epochs=5,
            batch_size=4,
        )

        assert config.base_model == "Qwen/Qwen3-7B"
        assert config.method == "qlora"
        assert config.num_epochs == 5
        assert config.batch_size == 4


class TestTrainingCallbacks:
    """Tests for training callbacks."""

    def test_callbacks_initial_state(self):
        """Should have correct initial state."""
        from opencrab.training.pipeline import TrainingCallbacks

        cb = TrainingCallbacks()

        assert cb.final_loss == 0.0
        assert cb.training_duration_seconds == 0.0

    def test_callbacks_on_step(self):
        """Should track loss on step."""
        from opencrab.training.pipeline import TrainingCallbacks

        cb = TrainingCallbacks()
        cb.on_step(1, 0.5)

        assert cb.final_loss == 0.5

    def test_callbacks_training_duration(self):
        """Should calculate training duration."""
        import time

        from opencrab.training.pipeline import TrainingCallbacks

        cb = TrainingCallbacks()
        cb.set_start_time()
        time.sleep(0.01)

        duration = cb.training_duration_seconds
        assert duration >= 0.01


class TestTrainingPipelines:
    """Tests for training pipeline selection."""

    def test_get_pipeline_lora(self):
        """Should get LoRA pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("lora")
        assert pipeline.method == "lora"

    def test_get_pipeline_qlora(self):
        """Should get QLoRA pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("qlora")
        assert pipeline.method == "qlora"

    def test_get_pipeline_full(self):
        """Should get full fine-tuning pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("full")
        assert pipeline.method == "full"

    def test_get_pipeline_last_layer(self):
        """Should get last layer fine-tuning pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("last_layer")
        assert pipeline.method == "last_layer"

    def test_get_pipeline_slime(self):
        """Should get slime pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("slime")
        assert pipeline.method == "slime"

    def test_get_pipeline_router(self):
        """Should get router training pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("router")
        assert pipeline.method == "router"

    def test_get_pipeline_head_classifier(self):
        """Should get head classifier pipeline."""
        from opencrab.training.pipeline import get_pipeline

        pipeline = get_pipeline("head_classifier")
        assert pipeline.method == "head_classifier"

    def test_get_pipeline_invalid(self):
        """Should raise error for invalid method."""
        from opencrab.training.pipeline import get_pipeline

        with pytest.raises(ValueError, match="Unknown training method"):
            get_pipeline("invalid_method")


class TestLoRATrainingPipeline:
    """Tests for LoRA training pipeline."""

    def test_lora_pipeline_properties(self):
        """Should have correct method name."""
        from opencrab.training.pipeline import LoRATrainingPipeline

        pipeline = LoRATrainingPipeline()
        assert pipeline.method == "lora"
        assert pipeline.sample_count == 0

    def test_load_jsonl_empty(self, tmp_path):
        """Should handle empty JSONL file."""
        from opencrab.training.pipeline import LoRATrainingPipeline

        pipeline = LoRATrainingPipeline()
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        data = list(pipeline._load_jsonl(empty_file))
        assert len(data) == 0

    def test_load_jsonl_with_data(self, tmp_path):
        """Should load valid JSONL data."""
        from opencrab.training.pipeline import LoRATrainingPipeline

        pipeline = LoRATrainingPipeline()
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"messages": [{"role": "user", "content": "hi"}]}\n{"messages": [{"role": "user", "content": "hello"}]}'
        )

        data = list(pipeline._load_jsonl(data_file))
        assert len(data) == 2
        assert data[0]["messages"][0]["content"] == "hi"

    def test_prepare_dataset(self, tmp_path):
        """Should prepare dataset correctly."""
        from transformers import AutoTokenizer

        from opencrab.training.pipeline import LoRATrainingPipeline

        pipeline = LoRATrainingPipeline()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token

        data = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "assistant", "content": "hello"}]},
        ]

        dataset = pipeline._prepare_dataset(data, tokenizer, max_length=128)

        assert len(dataset) == 2


class TestGenerateModelCard:
    """Tests for model card generation."""

    def test_generate_model_card(self, tmp_path):
        """Should generate valid model card."""
        from opencrab.training.pipeline import TrainingConfig, generate_model_card

        config = TrainingConfig(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            method="lora",
            output_dir=str(tmp_path),
            num_epochs=3,
        )

        card = generate_model_card(
            config, tmp_path, sample_count=100, final_loss=0.15, training_duration_minutes=30.0
        )

        assert card["model_id"].startswith("opencrab-distilled-model-")
        assert card["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert card["training_method"] == "lora"
        assert card["sample_count"] == 100
        assert card["final_loss"] == 0.15
        assert card["training_duration_minutes"] == 30.0
        assert "training_config" in card

    def test_model_card_saved_to_file(self, tmp_path):
        """Should save model card to file."""
        from opencrab.training.pipeline import TrainingConfig, generate_model_card

        config = TrainingConfig()
        generate_model_card(config, tmp_path, sample_count=50)

        card_file = tmp_path / "model_card.json"
        assert card_file.exists()

        with open(card_file) as f:
            card = json.load(f)
        assert card["sample_count"] == 50


class TestHeadClassifierPipeline:
    """Tests for head classifier training pipeline."""

    def test_head_classifier_pipeline_properties(self):
        """Should have correct method name."""
        from opencrab.training.pipeline import HeadClassifierTrainingPipeline

        pipeline = HeadClassifierTrainingPipeline()
        assert pipeline.method == "head_classifier"

    def test_head_classifier_save(self, tmp_path):
        """Should save head classifier model correctly."""
        import torch

        from opencrab.training.pipeline import HeadClassifierTrainingPipeline

        pipeline = HeadClassifierTrainingPipeline()

        # Create a mock head model
        pipeline._head_model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

        # Mock tokenizer
        pipeline._tokenizer = MagicMock()
        pipeline._tokenizer.save_pretrained = MagicMock()

        pipeline._final_loss = 0.1

        # Set env to use temp path
        with patch.dict(os.environ, {"ROUTER_MODEL_PATH": str(tmp_path)}):
            import asyncio

            asyncio.run(pipeline.save(tmp_path))

        # Check files were saved
        assert (tmp_path / "head_model.pt").exists()
        assert (tmp_path / "head_config.json").exists()
        assert (tmp_path / "head_config.json").read_text()

        # Verify config structure
        config = json.loads((tmp_path / "head_config.json").read_text())
        assert config["head_type"] == "head_classifier"
        assert config["hidden_size"] == 128
        assert config["num_labels"] == 2
