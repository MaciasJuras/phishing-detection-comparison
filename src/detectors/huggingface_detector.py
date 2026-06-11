import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from src.data.preprocessor import clean_text


MODEL_NAME = "ealvaradob/bert-finetuned-phishing"
MAX_LENGTH = 512


class HuggingFaceDetector:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = 0.5,
        max_length: int = MAX_LENGTH,
        batch_size: int = 16,
        local_files_only: bool = False,
    ):
        self.threshold = threshold
        self.max_length = max_length
        self.batch_size = batch_size
        self.local_files_only = local_files_only

        print(f"[HuggingFace] Loading model: {model_name} ...")
        auth_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        auth_kwargs = {"token": auth_token} if auth_token else {}
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
            use_fast=True,
            **auth_kwargs,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
            use_safetensors=False,
            **auth_kwargs,
        )
        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            local_files_only=self.local_files_only,
        )
        self.id2label = {
            int(k): str(v).lower()
            for k, v in getattr(self.classifier.model.config, "id2label", {}).items()
        }
        self.label2id = {
            str(k).lower(): int(v)
            for k, v in getattr(self.classifier.model.config, "label2id", {}).items()
        }
        self.phishing_label = "phishing"
        self.benign_label = "benign"
        print("[HuggingFace] Model loaded.")

    @staticmethod
    def _collect_scores(output) -> list[dict]:
        if isinstance(output, dict):
            return [output]

        if isinstance(output, list):
            scores = []
            for item in output:
                scores.extend(HuggingFaceDetector._collect_scores(item))
            return scores

        return []

    def _phishing_probability(self, output) -> float:
        scores = self._collect_scores(output)
        if not scores:
            return 0.0

        phishing_id = self.label2id.get(self.phishing_label)
        benign_id = self.label2id.get(self.benign_label)

        for item in scores:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))

            if label == self.phishing_label:
                return min(max(score, 0.0), 1.0)

            if label == self.benign_label:
                return min(max(1.0 - score, 0.0), 1.0)

            if phishing_id is not None and label == f"label_{phishing_id}":
                return min(max(score, 0.0), 1.0)

            if benign_id is not None and label == f"label_{benign_id}":
                return min(max(1.0 - score, 0.0), 1.0)

        best = max(scores, key=lambda item: float(item.get("score", 0.0)))
        return min(max(float(best.get("score", 0.0)), 0.0), 1.0)

    def predict_proba(self, text: str) -> float:
        cleaned = clean_text(text)
        raw = self.classifier(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        return self._phishing_probability(raw)

    def predict_proba_batch(self, texts, batch_size: int | None = None) -> list[float]:
        cleaned_texts = [clean_text(text) for text in texts]
        raw_outputs = self.classifier(
            cleaned_texts,
            truncation=True,
            max_length=self.max_length,
            batch_size=batch_size or self.batch_size,
        )
        return [self._phishing_probability(output) for output in raw_outputs]

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= self.threshold)

    def predict_batch(self, texts, batch_size: int = 16) -> list[int]:
        probs = self.predict_proba_batch(texts, batch_size=batch_size)
        return [int(prob >= self.threshold) for prob in probs]