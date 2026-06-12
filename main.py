import os
from pathlib import Path

from src.data.loader import load_dataset
from src.detectors.classical_ml import LogisticRegressionDetector, NaiveBayesDetector
from src.detectors.distilbert_detector import DebertaDetector
from src.detectors.rule_based import RuleBasedDetector
from src.detectors.huggingface_detector import HuggingFaceDetector
from src.detectors.url_detector import URLDetector
from src.detectors.hybrid_detector import HybridDetector
from src.evaluation.evaluator import (
    evaluate,
    save_results,
)


BASE_DEBERTA_MODEL = "microsoft/deberta-v3-small"
DEBERTA_OUTPUT_DIR = Path(os.environ.get("DEBERTA_OUTPUT_DIR", "models/deberta-phishing"))
DEBERTA_MODEL_NAME = os.environ.get("DEBERTA_MODEL_NAME")
TRAIN_DEBERTA = os.environ.get("TRAIN_DEBERTA", "auto").lower()
DEBERTA_EPOCHS = int(os.environ.get("DEBERTA_EPOCHS", "1"))
DEBERTA_SAMPLE_SIZE = int(os.environ.get("DEBERTA_SAMPLE_SIZE", "800"))
DEBERTA_MAX_LENGTH = int(os.environ.get("DEBERTA_MAX_LENGTH", "256"))
DEBERTA_FREEZE_BASE = os.environ.get("DEBERTA_FREEZE_BASE", "auto").lower()


def resolve_deberta_model_name() -> str:
    if DEBERTA_MODEL_NAME:
        return DEBERTA_MODEL_NAME
    if (DEBERTA_OUTPUT_DIR / "config.json").exists():
        return str(DEBERTA_OUTPUT_DIR)
    return BASE_DEBERTA_MODEL


def should_train_deberta(device_name: str) -> bool:
    if TRAIN_DEBERTA in {"1", "true", "yes", "force"}:
        return True
    if TRAIN_DEBERTA in {"0", "false", "no", "skip"}:
        return False
    return device_name != "cpu"


def should_freeze_deberta_base(device_name: str) -> bool:
    if DEBERTA_FREEZE_BASE in {"1", "true", "yes"}:
        return True
    if DEBERTA_FREEZE_BASE in {"0", "false", "no"}:
        return False
    return device_name == "cpu"


def main():
    # ── Load data ──────────────────────────────────────────────────
    train_df, test_df = load_dataset()

    # Optional: use a smaller slice for fast iteration during development
    # - Uncomment or adjust the lines below to speed up training/prediction locally.
    # test_df = test_df.sample(200, random_state=42).reset_index(drop=True)
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
        deberta_det = DebertaDetector(
            model_name=resolve_deberta_model_name(),
            max_length=DEBERTA_MAX_LENGTH,
        )
    except Exception as e:
        print(f"[Main] Warning: DebertaDetector failed to load: {e}")
        deberta_det = None

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

    print("\n[Main] Training Naive Bayes detector...")
    bayes_det.fit(train_texts, train_labels)

    print("[Main] Training Logistic Regression detector...")
    logistic_det.fit(train_texts, train_labels)

    if deberta_det is not None:
        try:
            if not deberta_det._uses_base_checkpoint():
                print("[Main] Using fine-tuned DeBERTa checkpoint without additional local training.")
            else:
                deberta_device = deberta_det.best_training_device()
                if not should_train_deberta(deberta_device):
                    print(
                        "[Main] CPU only detected; skipping base DeBERTa training. "
                        "Set TRAIN_DEBERTA=force to train a small sample on CPU."
                    )
                    deberta_det = None
                else:
                    deberta_batch_size = 8 if deberta_device in {"cuda", "mps"} else 1
                    print(f"[Main] Fine-tuning DeBERTa detector on {deberta_device}...")
                    deberta_det.fit(
                        train_texts,
                        train_labels,
                        epochs=DEBERTA_EPOCHS,
                        batch_size=deberta_batch_size,
                        lr=2e-5,
                        device=deberta_device,
                        sample_size=DEBERTA_SAMPLE_SIZE,
                        freeze_base=should_freeze_deberta_base(deberta_device),
                    )
                    deberta_det.save(str(DEBERTA_OUTPUT_DIR))
        except Exception as e:
            print(f"[Main] DeBERTa fine-tuning failed: {e}")
            deberta_det = None

    if deberta_det is None and (DEBERTA_OUTPUT_DIR / "config.json").exists():
        try:
            print(f"[Main] Loading saved DeBERTa checkpoint from {DEBERTA_OUTPUT_DIR}...")
            deberta_det = DebertaDetector(
                model_name=str(DEBERTA_OUTPUT_DIR),
                max_length=DEBERTA_MAX_LENGTH,
            )
        except Exception as e:
            print(f"[Main] Saved DeBERTa checkpoint failed to load: {e}")
            deberta_det = None

    transformer_det = deberta_det if deberta_det is not None else hf_det
    if transformer_det is None:
        print("[Main] Warning: No transformer detector available for hybrid models.")

    hybrid_detectors = [rule_det, bayes_det, logistic_det]
    hybrid_weights = [0.15, 0.15, 0.30]
    if transformer_det is not None:
        hybrid_detectors.append(transformer_det)
        hybrid_weights.append(0.30)
    hybrid_detectors.append(url_det)
    hybrid_weights.append(0.10)

    hybrid_det = HybridDetector(
        *hybrid_detectors,
        weights=tuple(hybrid_weights),
        threshold=0.45,
        strict_vote_threshold=0.35,
    )

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

    if deberta_det is not None:
        try:
            print("\n[Main] Running DeBERTa detector...")
            deberta_preds = deberta_det.predict_batch(texts)
        except Exception as e:
            print(f"[Main] DeBERTa prediction failed: {e}")
            deberta_preds = [0] * len(texts)
    else:
        deberta_preds = None

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
    if deberta_preds is not None:
        all_metrics.append(evaluate("DeBERTa", y_true, deberta_preds))

    all_metrics.append(evaluate("URL Analyzer", y_true, url_preds))
    all_metrics.append(evaluate("Hybrid", y_true, hybrid_preds))
    all_metrics.append(evaluate("Strict Hybrid", y_true, strict_preds))
    all_metrics.append(evaluate("Hybrid (Rule+URL)", y_true, rule_url_preds))

    # ── Save results ───────────────────────────────────────────────
    save_results(all_metrics)


if __name__ == "__main__":
    main()
