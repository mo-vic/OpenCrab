"""Training pipeline for fine-tuning distilled model."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import structlog
import torch

logger = structlog.get_logger()


class TrainingCallbacks:
    """Callbacks for training progress monitoring."""

    _last_loss: float = 0.0
    _training_start_time: float | None = None

    def on_step(self, step: int, loss: float) -> None:
        """Called after each training step."""
        self._last_loss = loss

    def on_epoch(self, epoch: int, val_loss: float) -> None:
        """Called after each epoch."""

    def on_complete(self, model_id: str, metrics: dict[str, Any]) -> None:
        """Called when training completes."""

    def on_error(self, error: Exception) -> None:
        """Called when training fails."""

    @property
    def final_loss(self) -> float:
        """Return the final loss from training."""
        return self._last_loss

    @property
    def training_duration_seconds(self) -> float:
        """Return training duration in seconds."""
        if self._training_start_time is None:
            return 0.0
        import time

        return time.time() - self._training_start_time

    def set_start_time(self) -> None:
        """Set the training start time."""
        import time

        self._training_start_time = time.time()


@dataclass
class TrainingConfig:
    """Configuration for training.

    base_model defaults to BASE_MODEL env var, or Qwen/Qwen2.5-0.5B-Instruct if not set.
    """

    base_model: str = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    method: str = "lora"
    output_dir: str = "./model_output"
    num_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500


class TrainingPipeline(ABC):
    """Base class for training pipelines."""

    _model: Any = None
    _tokenizer: Any = None

    @property
    @abstractmethod
    def method(self) -> str:
        """Training method name."""

    @abstractmethod
    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        """Run training.

        Args:
            config: Training configuration.
            data_path: Path to training data JSONL.
        """

    @abstractmethod
    async def save(self, output_path: str | Path) -> None:
        """Save the trained model.

        Args:
            output_path: Path to save model.
        """

    def _load_jsonl(self, path: str | Path) -> Iterator[dict[str, Any]]:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _prepare_dataset(self, data: list[dict[str, Any]], tokenizer: Any, max_length: int) -> Any:
        """Prepare dataset for training."""
        from transformers import Dataset

        def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
            messages = sample.get("messages", [])
            if not messages:
                return {"text": ""}
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role.upper()}: {content}\n"
            return {"text": text.strip()}

        formatted_data = [format_sample(d) for d in data if format_sample(d)["text"]]
        return Dataset.from_list(formatted_data)

    def _tokenize_function(
        self, examples: dict[str, Any], tokenizer: Any, max_length: int
    ) -> dict[str, Any]:
        """Tokenize text for training."""
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result


class LoRATrainingPipeline(TrainingPipeline):
    """LoRA fine-tuning pipeline."""

    _sample_count: int = 0
    _final_loss: float | None = None
    _training_start_time: float | None = None

    @property
    def method(self) -> str:
        return "lora"

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def final_loss(self) -> float | None:
        return self._final_loss

    @property
    def training_duration_minutes(self) -> float | None:
        if self._training_start_time is None:
            return None
        import time

        return (time.time() - self._training_start_time) / 60.0

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        import time

        self._training_start_time = time.time()
        logger.info("lora_training_started", config=config, data_path=data_path)
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except ImportError:
            logger.error("training_dependencies_missing", required=["transformers", "peft"])
            raise ImportError("Run: pip install transformers peft") from None

        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(config.base_model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)

        data = list(self._load_jsonl(data_path))
        self._sample_count = len(data)
        train_data = self._prepare_dataset(data, self._tokenizer, config.max_seq_length)

        def tokenize(examples):
            return self._tokenize_function(examples, self._tokenizer, config.max_seq_length)

        train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])

        class LossCallback(TrainingCallbacks):
            def __init__(self):
                pass

            def on_step(self, step: int, loss: float):
                self._last_loss = loss

        loss_cb = LossCallback()

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            fp16=True,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=self._tokenizer,
        )

        trainer.add_callback(loss_cb)
        trainer.train()
        self._final_loss = loss_cb.final_loss
        logger.info("lora_training_completed")

    async def save(self, output_path: str | Path) -> None:
        if self._model is None:
            logger.warning("no_model_to_save")
            return
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("lora_model_save_started", path=output_path)
        self._model.save_pretrained(output_path)
        if self._tokenizer:
            self._tokenizer.save_pretrained(output_path)
        logger.info("lora_model_saved", path=str(output_path))


class QLoRATrainingPipeline(LoRATrainingPipeline):
    """QLoRA fine-tuning pipeline (4-bit quantized LoRA)."""

    @property
    def method(self) -> str:
        return "qlora"

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        logger.info("qlora_training_started", config=config, data_path=data_path)
        try:
            from bitsandbytes import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except ImportError:
            logger.error(
                "qlora_dependencies_missing", required=["transformers", "peft", "bitsandbytes"]
            )
            raise ImportError("Run: pip install transformers peft bitsandbytes") from None

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization_config,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)

        data = list(self._load_jsonl(data_path))
        self._sample_count = len(data)
        train_data = self._prepare_dataset(data, self._tokenizer, config.max_seq_length)

        def tokenize(examples):
            return self._tokenize_function(examples, self._tokenizer, config.max_seq_length)

        train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            fp16=True,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=self._tokenizer,
        )

        class LossCallback(TrainingCallbacks):
            def __init__(self):
                pass

            def on_step(self, step: int, loss: float):
                self._last_loss = loss

        loss_cb = LossCallback()
        trainer.add_callback(loss_cb)
        trainer.train()
        self._final_loss = loss_cb.final_loss
        logger.info("qlora_training_completed")


class FullFineTuningPipeline(TrainingPipeline):
    """Full fine-tuning pipeline (all parameters)."""

    _final_loss: float | None = None

    @property
    def method(self) -> str:
        return "full"

    @property
    def final_loss(self) -> float | None:
        return self._final_loss

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        logger.info("full_finetuning_started", config=config, data_path=data_path)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except ImportError:
            logger.error("training_dependencies_missing", required=["transformers"])
            raise ImportError("Run: pip install transformers") from None

        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(config.base_model)

        for param in self._model.parameters():
            param.requires_grad = True

        data = list(self._load_jsonl(data_path))
        train_data = self._prepare_dataset(data, self._tokenizer, config.max_seq_length)

        def tokenize(examples):
            return self._tokenize_function(examples, self._tokenizer, config.max_seq_length)

        train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])

        class LossCallback(TrainingCallbacks):
            def __init__(self):
                pass

            def on_step(self, step: int, loss: float):
                self._last_loss = loss

        loss_cb = LossCallback()

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            fp16=True,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=self._tokenizer,
        )

        trainer.add_callback(loss_cb)
        trainer.train()
        self._final_loss = loss_cb.final_loss
        logger.info("full_finetuning_completed")

    async def save(self, output_path: str | Path) -> None:
        if self._model is None:
            logger.warning("no_model_to_save")
            return
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(output_path)
        if self._tokenizer:
            self._tokenizer.save_pretrained(output_path)
        logger.info("full_model_saved", path=str(output_path))


class LastLayerFineTuningPipeline(TrainingPipeline):
    """Last layer fine-tuning using a lightweight trainable head on frozen base model embeddings.

    The base model remains frozen (no parameter updates). A trainable classification
    head is attached that takes base model embeddings as input and learns to predict
    correct responses. This is lighter than full fine-tuning while being more
    effective than naive last-layer freezing.
    """

    _head_layer: Any = None
    _sample_count: int = 0

    @property
    def method(self) -> str:
        return "last_layer"

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        logger.info("last_layer_finetuning_started", config=config, data_path=data_path)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("training_dependencies_missing", required=["transformers"])
            raise ImportError("Run: pip install transformers") from None

        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load base model (frozen) for feature extraction
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
        base_model.eval()

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Get hidden size from config
        hidden_size = base_model.config.hidden_size

        # Create a lightweight trainable head that takes last transformer block features
        self._head_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, 2),
        )

        # Prepare training data
        data = list(self._load_jsonl(data_path))
        self._sample_count = len(data)

        # Extract training samples
        queries = []
        labels = []
        for item in data:
            msgs = item.get("messages", [])
            # Get last user message as query
            query = ""
            for msg in reversed(msgs):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            # Label: 1 if there's a corrected response (distilled should handle it)
            has_correction = "corrected_response" in item or "original_response" in item
            labels.append(1 if has_correction else 0)
            queries.append(query)

        encodings = self._tokenizer(
            queries,
            truncation=True,
            padding=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )

        # Training loop
        optimizer = torch.optim.AdamW(self._head_layer.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        batch_size = config.batch_size
        num_epochs = config.num_epochs

        self._head_layer.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(encodings["input_ids"]), batch_size):
                batch_input_ids = encodings["input_ids"][i : i + batch_size]
                batch_attention_mask = encodings["attention_mask"][i : i + batch_size]
                batch_labels = torch.tensor(labels[i : i + batch_size])

                optimizer.zero_grad()

                with torch.no_grad():
                    outputs = base_model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        output_hidden_states=True,
                    )
                    # Use last hidden state, take [CLS] token (first token)
                    hidden_states = outputs.hidden_states[-1]
                    features = hidden_states[:, 0, :]

                logits = self._head_layer(features)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info("last_layer_epoch", epoch=epoch + 1, avg_loss=avg_loss)

        self._final_loss = avg_loss if num_batches > 0 else 0
        logger.info("last_layer_finetuning_completed")

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def final_loss(self) -> float | None:
        return getattr(self, "_final_loss", None)

    async def save(self, output_path: str | Path) -> None:
        if self._head_layer is None:
            logger.warning("no_model_to_save")
            return
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save only the trainable head weights
        torch.save(self._head_layer.state_dict(), output_path / "head_layer.pt")
        if self._tokenizer:
            self._tokenizer.save_pretrained(output_path)
        logger.info("last_layer_model_saved", path=str(output_path))


class SlimeRemotePipeline(TrainingPipeline):
    """Slime remote training pipeline."""

    @property
    def method(self) -> str:
        return "slime"

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        logger.info("slime_remote_training_started", config=config, data_path=data_path)
        slime_server = os.environ.get("SLIME_SERVER_URL", "http://localhost:5000")

        data = list(self._load_jsonl(data_path))
        payload = {
            "model": config.base_model,
            "method": "slime",
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "data": data,
        }

        import httpx

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(f"{slime_server}/train", json=payload)
                response.raise_for_status()
                result = response.json()
                logger.info("slime_training_completed", result=result)
        except Exception as e:
            logger.error("slime_training_failed", error=str(e))
            raise

    async def save(self, output_path: str | Path) -> None:
        slime_server = os.environ.get("SLIME_SERVER_URL", "http://localhost:5000")
        import httpx

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{slime_server}/model/{output_path}")
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info("slime_model_saved", path=str(output_path))
        except Exception as e:
            logger.error("slime_save_failed", error=str(e))


class RouterTrainingPipeline(TrainingPipeline):
    """Training pipeline for the router classifier."""

    _final_loss: float | None = None

    @property
    def method(self) -> str:
        return "router"

    @property
    def final_loss(self) -> float | None:
        return self._final_loss

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        logger.info("router_training_started", config=config, data_path=data_path)
        try:
            from datasets import Dataset
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError:
            logger.error("router_dependencies_missing", required=["transformers", "datasets"])
            raise ImportError("Run: pip install transformers datasets") from None

        data = list(self._load_jsonl(data_path))
        queries = [d.get("query", "") for d in data]
        labels = [1 if d.get("should_handle_locally", False) else 0 for d in data]

        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        encodings = self._tokenizer(queries, truncation=True, padding=True, max_length=512)

        train_dataset = Dataset.from_dict(encodings)
        train_dataset = train_dataset.add_column("labels", labels)

        def tokenize_fn(examples):
            return self._tokenize_router(examples, self._tokenizer)

        train_dataset = train_dataset.map(
            tokenize_fn, batched=True, remove_columns=["input_ids", "attention_mask"]
        )
        train_dataset.set_format("torch")

        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self._tokenizer,
        )

        class LossCallback(TrainingCallbacks):
            def __init__(self):
                pass

            def on_step(self, step: int, loss: float):
                self._last_loss = loss

        loss_cb = LossCallback()
        trainer.add_callback(loss_cb)
        trainer.train()
        self._final_loss = loss_cb.final_loss
        logger.info("router_training_completed")

    def _tokenize_router(self, examples: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
        return tokenizer(
            examples["input_ids"],
            attention_mask=examples["attention_mask"],
            truncation=True,
            max_length=512,
        )

    async def save(self, output_path: str | Path) -> None:
        if self._model is None:
            logger.warning("no_model_to_save")
            return
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(output_path)
        if self._tokenizer:
            self._tokenizer.save_pretrained(output_path)
        logger.info("router_model_saved", path=str(output_path))


class HeadClassifierTrainingPipeline(TrainingPipeline):
    """Training pipeline for the HeadClassifier.

    Trains a lightweight classification head on top of the base model's embeddings.
    The base model remains frozen; only the classification head is trained.
    This produces a model compatible with HeadClassifier in serving.

    Saves to ROUTER_MODEL_PATH so HeadClassifier can load it directly.
    """

    _head_model: Any = None
    _head_tokenizer: Any = None

    @property
    def method(self) -> str:
        return "head_classifier"

    async def train(self, config: TrainingConfig, data_path: str | Path) -> None:
        logger.info("head_classifier_training_started", config=config, data_path=data_path)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.error("head_classifier_dependencies_missing", required=["transformers"])
            raise ImportError("Run: pip install transformers") from None

        # Load base model (frozen) to extract embeddings
        base_model = AutoModel.from_pretrained(config.base_model)
        base_model.eval()

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size

        # Create lightweight classification head
        self._head_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, 2),
        )

        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        data = list(self._load_jsonl(data_path))
        self._sample_count = len(data)

        # Prepare data
        queries = []
        labels = []
        for item in data:
            query = item.get("query", "")
            should_handle = item.get("should_handle_locally", False)
            queries.append(query)
            labels.append(1 if should_handle else 0)

        encodings = self._tokenizer(
            queries, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )

        # Training loop
        optimizer = torch.optim.AdamW(self._head_model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        batch_size = config.batch_size
        num_epochs = config.num_epochs

        self._head_model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(encodings["input_ids"]), batch_size):
                batch_input_ids = encodings["input_ids"][i : i + batch_size]
                batch_attention_mask = encodings["attention_mask"][i : i + batch_size]
                batch_labels = torch.tensor(labels[i : i + batch_size])

                optimizer.zero_grad()

                with torch.no_grad():
                    outputs = base_model(
                        input_ids=batch_input_ids, attention_mask=batch_attention_mask
                    )
                    # Use [CLS] token embedding or mean pooling
                    if hasattr(outputs, "last_hidden_state"):
                        features = outputs.last_hidden_state[:, 0, :]
                    else:
                        features = outputs[0][:, 0, :]

                logits = self._head_model(features)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info("head_classifier_epoch", epoch=epoch + 1, avg_loss=avg_loss)

        self._final_loss = avg_loss if num_batches > 0 else 0
        self._training_start_time = None
        logger.info("head_classifier_training_completed")

    @property
    def final_loss(self) -> float | None:
        return getattr(self, "_final_loss", None)

    async def save(self, output_path: str | Path) -> None:
        if self._head_model is None:
            logger.warning("no_head_model_to_save")
            return

        router_model_path = os.environ.get("ROUTER_MODEL_PATH", "./router_model")
        save_path = Path(output_path) if output_path else Path(router_model_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save head model state
        torch.save(self._head_model.state_dict(), save_path / "head_model.pt")

        # Save a minimal config for AutoModelForSequenceClassification compatibility
        import json

        config_data = {
            "model_type": "sequence_classification",
            "hidden_size": self._head_model[0].in_features,
            "num_labels": 2,
            "head_type": "head_classifier",
        }
        with open(save_path / "head_config.json", "w") as f:
            json.dump(config_data, f)

        if self._tokenizer:
            self._tokenizer.save_pretrained(save_path)

        logger.info("head_classifier_model_saved", path=str(save_path))


TRAINING_PIPELINES: dict[str, type[TrainingPipeline]] = {
    "lora": LoRATrainingPipeline,
    "qlora": QLoRATrainingPipeline,
    "full": FullFineTuningPipeline,
    "last_layer": LastLayerFineTuningPipeline,
    "slime": SlimeRemotePipeline,
    "router": RouterTrainingPipeline,
    "head_classifier": HeadClassifierTrainingPipeline,
}


def get_pipeline(method: str) -> TrainingPipeline:
    """Get a training pipeline by method."""
    cls = TRAINING_PIPELINES.get(method)
    if not cls:
        raise ValueError(f"Unknown training method: {method}")
    return cls()


def generate_model_card(
    config: TrainingConfig,
    output_path: str | Path,
    sample_count: int = 0,
    final_loss: float | None = None,
    training_duration_minutes: float | None = None,
) -> dict[str, Any]:
    """Generate a model card for the trained model in JSON format."""
    import json
    from datetime import datetime

    card = {
        "model_id": f"opencrab-distilled-model-{datetime.now(UTC).strftime('%Y-%m-%d')}",
        "base_model": config.base_model,
        "training_method": config.method,
        "training_date": datetime.now(UTC).isoformat(),
        "training_duration_minutes": training_duration_minutes,
        "sample_count": sample_count,
        "epochs": config.num_epochs,
        "final_loss": final_loss,
        "training_config": {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "max_seq_length": config.max_seq_length,
        },
    }

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    return card
