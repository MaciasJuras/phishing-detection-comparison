import torch

from src.data.loader import load_dataset
from src.detectors.classical_ml import LogisticRegressionDetector, NaiveBayesDetector
from src.detectors.distilbert_detector import DebertaDetector, DistilBERTDetector, RobertaDetector
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
    train_df, test_df = load_dataset()

    # Optional: use a smaller slice for fast iteration during development
    # - Uncomment or adjust the lines below to speed up training/prediction locally.
    #test_df = test_df.sample(200, random_state=42).reset_index(drop=True)
    # DEV: subsample training set for faster iteration (uncomment to use)
    #train_df = train_df.sample(2000, random_state=42).reset_index(drop=True)

    train_texts = train_df["body"].tolist()
    train_labels = train_df["label"].tolist()
    texts = test_df["body"].tolist()
    y_true = test_df["label"].tolist()

    # ── Instantiate detectors ──────────────────────────────────────
    rule_det = RuleBasedDetector(threshold=1.4)
    bayes_det = NaiveBayesDetector(alpha=1.0)
    logistic_det = LogisticRegressionDetector(max_iter=1000, solver="liblinear")
    try:
        hf_det = HuggingFaceDetector()
    except Exception as e:
        print(f"[Main] Warning: HuggingFaceDetector failed to load: {e}")
        hf_det = None

    try:
        distilbert_det = DistilBERTDetector()
    except Exception as e:
        print(f"[Main] Warning: DistilBERTDetector failed to load: {e}")
        distilbert_det = None

    try:
        deberta_det = DebertaDetector()
    except Exception as e:
        print(f"[Main] Warning: DebertaDetector failed to load: {e}")
        deberta_det = None

    try:
        roberta_det = RobertaDetector()
    except Exception as e:
        print(f"[Main] Warning: RobertaDetector failed to load: {e}")
        roberta_det = None
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

    print("\n[Main] Training Naive Bayes detector...")
    bayes_det.fit(train_texts, train_labels)

    print("[Main] Training Logistic Regression detector...")
    logistic_det.fit(train_texts, train_labels)

    if distilbert_det is not None:
        try:
            print("[Main] Fine-tuning DistilBERT detector...")
            distilbert_det.fit(train_texts, train_labels, epochs=1, batch_size=16, lr=2e-5, sample_size=800)
        except Exception as e:
            print(f"[Main] DistilBERT fine-tuning failed: {e}")

    if deberta_det is not None:
        try:
            if not torch.cuda.is_available():
                print("[Main] CPU only detected; skipping DeBERTa fine-tuning because it is too slow on CPU.")
            else:
                print("[Main] Fine-tuning DeBERTa detector...")
                deberta_det.fit(train_texts, train_labels, epochs=1, batch_size=8, lr=2e-5, sample_size=800)
        except Exception as e:
            print(f"[Main] DeBERTa fine-tuning failed: {e}")

    # ── Run predictions ────────────────────────────────────────────
    print("\n[Main] Running Rule-Based detector...")
    rule_preds = rule_det.predict_batch(texts)

    print("\n[Main] Running Naive Bayes detector...")
    bayes_preds = bayes_det.predict_batch(texts)

    print("\n[Main] Running Logistic Regression detector...")
    logistic_preds = logistic_det.predict_batch(texts)

    if hf_det is not None:
        try:
            print("\n[Main] Running HuggingFace detector...")
            hf_preds = hf_det.predict_batch(texts, batch_size=32)
        except Exception as e:
            print(f"[Main] HuggingFace prediction failed: {e}")
            hf_preds = [0] * len(texts)
    else:
        hf_preds = None

    if distilbert_det is not None:
        try:
            print("\n[Main] Running DistilBERT detector...")
            distilbert_preds = distilbert_det.predict_batch(texts)
        except Exception as e:
            print(f"[Main] DistilBERT prediction failed: {e}")
            distilbert_preds = [0] * len(texts)
    else:
        distilbert_preds = None

    if deberta_det is not None:
        try:
            print("\n[Main] Running DeBERTa detector...")
            deberta_preds = deberta_det.predict_batch(texts)
        except Exception as e:
            print(f"[Main] DeBERTa prediction failed: {e}")
            deberta_preds = [0] * len(texts)
    else:
        deberta_preds = None

    if roberta_det is not None:
        try:
            print("\n[Main] Running RoBERTa detector...")
            roberta_preds = roberta_det.predict_batch(texts)
        except Exception as e:
            print(f"[Main] RoBERTa prediction failed: {e}")
            roberta_preds = [0] * len(texts)
    else:
        roberta_preds = None

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
    all_metrics = []
    all_metrics.append(evaluate("Rule-Based", y_true, rule_preds))
    all_metrics.append(evaluate("Naive Bayes", y_true, bayes_preds))
    all_metrics.append(evaluate("Logistic Regression", y_true, logistic_preds))

    if hf_preds is not None:
        all_metrics.append(evaluate("HuggingFace (BERT)", y_true, hf_preds))
    if distilbert_preds is not None:
        all_metrics.append(evaluate("DistilBERT", y_true, distilbert_preds))
    if deberta_preds is not None:
        all_metrics.append(evaluate("DeBERTa", y_true, deberta_preds))
    if roberta_preds is not None:
        all_metrics.append(evaluate("RoBERTa", y_true, roberta_preds))

    all_metrics.append(evaluate("URL Analyzer", y_true, url_preds))
    all_metrics.append(evaluate("Hybrid", y_true, hybrid_preds))
    all_metrics.append(evaluate("Strict Hybrid", y_true, strict_preds))
    all_metrics.append(evaluate("Hybrid (Rule+URL)", y_true, rule_url_preds))

    # ── Save results ───────────────────────────────────────────────
    save_results(all_metrics)


if __name__ == "__main__":
    main()