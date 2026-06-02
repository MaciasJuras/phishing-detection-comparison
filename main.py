from src.data.loader import load_dataset
from src.detectors.rule_based import RuleBasedDetector
from src.detectors.huggingface_detector import HuggingFaceDetector
from src.detectors.url_detector import URLDetector
from src.detectors.hybrid_detector import HybridDetector
from src.evaluation.evaluator import evaluate, save_results


def main():
    # ── Load data ──────────────────────────────────────────────────
    _, test_df = load_dataset()

    # Optional: use a smaller slice for fast iteration during dev
    test_df = test_df.sample(200, random_state=42).reset_index(drop=True)

    texts = test_df["body"].tolist()
    y_true = test_df["label"].tolist()

    # ── Instantiate detectors ──────────────────────────────────────
    rule_det = RuleBasedDetector()
    hf_det = HuggingFaceDetector()
    url_det = URLDetector()
    hybrid_det = HybridDetector(rule_det, hf_det, url_det, weights=(0.25, 0.50, 0.25))

    # ── Run predictions ────────────────────────────────────────────
    print("\n[Main] Running Rule-Based detector...")
    rule_preds = rule_det.predict_batch(texts)

    print("\n[Main] Running HuggingFace detector...")
    hf_preds = hf_det.predict_batch(texts)

    print("\n[Main] Running URL detector...")
    url_preds = url_det.predict_batch(texts)

    print("\n[Main] Running Hybrid detector...")
    hybrid_preds = hybrid_det.predict_batch(texts)

    print("\n[Main] Running Strict Hybrid detector...")
    strict_preds = hybrid_det.predict_strict_batch(texts)

    # ── Evaluate ───────────────────────────────────────────────────
    all_metrics = [
        evaluate("Rule-Based", y_true, rule_preds),
        evaluate("HuggingFace (BERT)", y_true, hf_preds),
        evaluate("URL Analyzer", y_true, url_preds),
        evaluate("Hybrid", y_true, hybrid_preds),
        evaluate("Strict Hybrid", y_true, strict_preds),
    ]

    # ── Save results ───────────────────────────────────────────────
    save_results(all_metrics)


if __name__ == "__main__":
    main()