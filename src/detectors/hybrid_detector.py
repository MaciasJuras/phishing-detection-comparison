class HybridDetector:
    """
    Combines rule-based, HuggingFace, and URL detectors via weighted scoring.
    Each detector exposes predict_proba() → float in [0, 1].
    """

    def __init__(
            self,
            rule_detector,
            hf_detector,
            url_detector,
            weights: tuple[float, float, float] = (0.25, 0.50, 0.25),
            threshold: float = 0.5,
    ):
        self.detectors = [rule_detector, hf_detector, url_detector]
        self.weights = weights
        self.threshold = threshold
        #assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"