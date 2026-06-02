from src.data.loader import load_dataset
from src.detectors.rule_based import RuleBasedDetector
from src.detectors.huggingface_detector import HuggingFaceDetector
from src.detectors.url_detector import URLDetector
from src.detectors.hybrid_detector import HybridDetector
from src.evaluation.evaluator import (
    evaluate,
    save_results,
)


def main():
    # ── Load data ──────────────────────────────────────────────────
    _, test_df = load_dataset()

    # Optional: use a smaller slice for fast iteration during dev
    #test_df = test_df.sample(200, random_state=42).reset_index(drop=True)

    texts = test_df["body"].tolist()
    y_true = test_df["label"].tolist()

    # ── Instantiate detectors ──────────────────────────────────────
    rule_det = RuleBasedDetector(threshold=1.4)
    hf_det = HuggingFaceDetector()
    url_det = URLDetector(
        url_score_threshold=0.45,
        multi_url_bonus=0.20,
        feature_weights={
            "http_not_https": 1.2,
            "suspicious_path_keyword": 1.4,
            "many_query_params": 1.2,
            "open_redirect_param": 1.8,
        },
    )
    hybrid_det = HybridDetector(
        rule_det,
        hf_det,
        url_det,
        weights=(0.20, 0.55, 0.25),
        threshold=0.45,
        strict_vote_threshold=0.35,
    )

    # ── Run predictions ────────────────────────────────────────────
    print("\n[Main] Running Rule-Based detector...")
    rule_preds = rule_det.predict_batch(texts)

    print("\n[Main] Running HuggingFace detector...")
    hf_preds = hf_det.predict_batch(texts, batch_size=32)

    print("\n[Main] Running URL detector...")
    url_preds = url_det.predict_batch(texts)

    print("\n[Main] Running Hybrid detector...")
    hybrid_preds = hybrid_det.predict_batch(texts)

    print("\n[Main] Running Strict Hybrid detector...")
    strict_preds = hybrid_det.predict_strict_batch(texts)

    print("\n[Main] Running Hybrid (Rule+URL) detector...")
    rule_url_probs = [
        0.60 * rule_det.predict_proba(text) + 0.40 * url_det.predict_proba(text)
        for text in texts
    ]
    rule_url_preds = [int(prob >= 0.5) for prob in rule_url_probs]

    # ── Evaluate ───────────────────────────────────────────────────
    all_metrics = [
        evaluate("Rule-Based", y_true, rule_preds),
        evaluate("HuggingFace (BERT)", y_true, hf_preds),
        evaluate("URL Analyzer", y_true, url_preds),
        evaluate("Hybrid", y_true, hybrid_preds),
        evaluate("Strict Hybrid", y_true, strict_preds),
        evaluate("Hybrid (Rule+URL)", y_true, rule_url_preds),
    ]

    # ── Save results ───────────────────────────────────────────────
    save_results(all_metrics)


if __name__ == "__main__":
    main()