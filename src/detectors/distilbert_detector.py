import os

# Do not force offline mode here; allow environment to control HF connectivity.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from src.data.preprocessor import clean_text


DEFAULT_DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        inputs["labels"] = self.labels[idx]
        return inputs


def default_collate(batch, tokenizer):
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    batch_data = {
        "input_ids": [item["input_ids"] for item in batch],
        "attention_mask": [item["attention_mask"] for item in batch],
    }
    batch_data = tokenizer.pad(batch_data, padding=True, return_tensors="pt")
    batch_data["labels"] = labels
    return batch_data


class HFTransformerDetector:
    """Generic HF text classifier wrapper for transformer-based models."""

    def __init__(
        self,
        model_name: str,
        detector_label: str,
        threshold: float = 0.5,
        max_length: int = 512,
        batch_size: int = 16,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.detector_label = detector_label
        self.threshold = threshold
        self.max_length = max_length
        self.batch_size = batch_size
        self.local_files_only = local_files_only

        print(f"[{self.detector_label}] Loading model: {model_name} ...")
        auth_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        auth_kwargs = {"token": auth_token} if auth_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
            use_fast=True,
            **auth_kwargs,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
            use_safetensors=False,
            **auth_kwargs,
        )

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            local_files_only=self.local_files_only,
        )
        self.label_map = self._build_label_map(self.model.config)
        print(f"[{self.detector_label}] Model loaded.")

    @staticmethod
    def _build_label_map(config):
        id2label = getattr(config, "id2label", {}) or {}
        return {int(k): str(v).lower() for k, v in id2label.items()}

    @staticmethod
    def _clean_text(text: str) -> str:
        return clean_text(text)

    def _get_phishing_probability(self, output) -> float:
        if isinstance(output, dict):
            output = [output]

        best_score = 0.0
        for item in output:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            best_score = max(best_score, score)

            if label in {"phishing", "spam", "malicious", "negative", "contradiction"}:
                return min(max(score, 0.0), 1.0)
            if label in {"benign", "legitimate", "positive", "entailment"}:
                return min(max(1.0 - score, 0.0), 1.0)
            if label == "neutral":
                return 0.5

        return min(max(best_score, 0.0), 1.0)

    def predict_proba(self, text: str) -> float:
        cleaned = self._clean_text(text)
        raw = self.classifier(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        return self._get_phishing_probability(raw)

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= self.threshold)

    def predict_batch(self, texts, batch_size: int | None = None) -> list[int]:
        probs = self.predict_proba_batch(texts, batch_size=batch_size)
        return [int(p >= self.threshold) for p in probs]

    def predict_proba_batch(self, texts, batch_size: int | None = None) -> list[float]:
        cleaned_texts = [self._clean_text(text) for text in texts]
        raw_outputs = self.classifier(
            cleaned_texts,
            truncation=True,
            max_length=self.max_length,
            batch_size=batch_size or self.batch_size,
        )
        return [self._get_phishing_probability(output) for output in raw_outputs]

    def fit(
        self,
        texts,
        labels,
        epochs: int = 1,
        batch_size: int = 16,
        lr: float = 2e-5,
        device: str | None = None,
        sample_size: int | None = None,
    ):
        device_name = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        device_obj = torch.device(device_name)

        if sample_size is not None and sample_size < len(texts):
            texts = texts[:sample_size]
            labels = labels[:sample_size]

        dataset = TextClassificationDataset(texts, labels, self.tokenizer, self.max_length)
        print(f"[{self.detector_label}] Training on {len(dataset)} samples (batch_size={batch_size}) using {device_obj}.")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: default_collate(batch, self.tokenizer),
            num_workers=0,
        )

        self.model.to(device_obj)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(loader, start=1):
                batch = {k: v.to(device_obj) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss += loss.item()
                if batch_idx % 20 == 0:
                    print(f"[{self.detector_label}] Epoch {epoch + 1}/{epochs} batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")
            print(f"[{self.detector_label}] Epoch {epoch + 1}/{epochs} avg loss: {epoch_loss / len(loader):.4f}")

        self.model.eval()


class DistilBERTDetector(HFTransformerDetector):
    """DistilBERT-based text classifier using HuggingFace pipeline."""

    def __init__(
        self,
        model_name: str = DEFAULT_DISTILBERT_MODEL,
        threshold: float = 0.5,
        max_length: int = 512,
        batch_size: int = 16,
        local_files_only: bool = False,
    ):
        super().__init__(model_name, "DistilBERT", threshold, max_length, batch_size, local_files_only)


class DebertaDetector(HFTransformerDetector):
    """DeBERTa-based detector using a pretrained HuggingFace model."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        threshold: float = 0.5,
        max_length: int = 512,
        batch_size: int = 16,
        local_files_only: bool = False,
    ):
        super().__init__(model_name, "DeBERTa", threshold, max_length, batch_size, local_files_only)


class RobertaDetector(HFTransformerDetector):
    """RoBERTa-based detector using a pretrained HuggingFace model."""

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        threshold: float = 0.5,
        max_length: int = 512,
        batch_size: int = 16,
        local_files_only: bool = False,
    ):
        super().__init__(model_name, "RoBERTa", threshold, max_length, batch_size, local_files_only)
