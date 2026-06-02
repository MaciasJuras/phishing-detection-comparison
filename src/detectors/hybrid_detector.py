class HybridDetector:
    """Combine rule-based, HuggingFace, and URL detectors into one score."""

    def __init__(
        self,
        rule_detector,
        hf_detector,
        url_detector,
        weights: tuple[float, float, float] = (0.25, 0.50, 0.25),
        threshold: float = 0.5,
        strategy: str = "weighted",
    ):
        self.detectors = [rule_detector, hf_detector, url_detector]
        self.weights = weights
        self.threshold = threshold
        self.strategy = strategy.lower().strip()

        if len(self.weights) != len(self.detectors):
            raise ValueError("weights must contain one value per detector")

        if self.strategy not in {"weighted", "majority"}:
            raise ValueError("strategy must be 'weighted' or 'majority'")

        total_weight = sum(self.weights)
        self.normalized_weights = (
            tuple(w / total_weight for w in self.weights)
            if total_weight else tuple(1.0 / len(self.detectors) for _ in self.detectors)
        )

    def _predict_detector_proba(self, detector, text: str) -> float:
        if hasattr(detector, "predict_proba"):
            return float(detector.predict_proba(text))
        return float(detector.predict(text))

    def _detector_probabilities(self, text: str) -> list[float]:
        return [self._predict_detector_proba(detector, text) for detector in self.detectors]

    def predict_proba(self, text: str) -> float:
        if self.strategy == "majority":
            votes = [detector.predict(text) for detector in self.detectors]
            return sum(votes) / len(votes)

        scores = self._detector_probabilities(text)
        return sum(weight * score for weight, score in zip(self.normalized_weights, scores))

    def predict_proba_strict(self, text: str) -> float:
        scores = self._detector_probabilities(text)
        if not scores:
            return 0.0

        positive_votes = [score for score in scores if score >= self.threshold]
        if len(positive_votes) < 2:
            return 0.0

        positive_weights = [
            weight for weight, score in zip(self.normalized_weights, scores)
            if score >= self.threshold
        ]
        total_weight = sum(positive_weights)
        if total_weight == 0:
            return sum(positive_votes) / len(positive_votes)

        return sum(
            weight * score
            for weight, score in zip(self.normalized_weights, scores)
            if score >= self.threshold
        ) / total_weight

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= self.threshold)

    def predict_strict(self, text: str) -> int:
        return int(self.predict_proba_strict(text) >= self.threshold)

    def predict_batch(self, texts) -> list[int]:
        return [self.predict(text) for text in texts]

    def predict_strict_batch(self, texts) -> list[int]:
        return [self.predict_strict(text) for text in texts]