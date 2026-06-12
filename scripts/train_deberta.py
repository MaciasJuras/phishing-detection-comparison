import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.detectors.distilbert_detector import BASE_DEBERTA_MODEL, DebertaDetector


OUTPUT_DIR = Path(os.environ.get("DEBERTA_OUTPUT_DIR", "models/deberta-phishing"))
MODEL_NAME = os.environ.get("DEBERTA_MODEL_NAME", BASE_DEBERTA_MODEL)
EPOCHS = int(os.environ.get("DEBERTA_EPOCHS", "1"))
MAX_LENGTH = int(os.environ.get("DEBERTA_MAX_LENGTH", "128"))
LR = float(os.environ.get("DEBERTA_LR", "2e-5"))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def main():
    device = DebertaDetector.best_training_device()
    sample_size = env_int("DEBERTA_SAMPLE_SIZE", 600 if device == "cpu" else 800)
    batch_size = env_int("DEBERTA_BATCH_SIZE", 1 if device == "cpu" else 8)
    freeze_base = env_bool("DEBERTA_FREEZE_BASE", device == "cpu")

    train_df, _ = load_dataset()
    train_texts = train_df["body"].tolist()
    train_labels = train_df["label"].tolist()

    print(f"[TrainDeBERTa] Device: {device}")
    print(f"[TrainDeBERTa] Model: {MODEL_NAME}")
    print(f"[TrainDeBERTa] Output: {OUTPUT_DIR}")
    print(f"[TrainDeBERTa] Samples: {sample_size}, max_length: {MAX_LENGTH}, batch_size: {batch_size}")

    detector = DebertaDetector(
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
        batch_size=batch_size,
    )
    detector.fit(
        train_texts,
        train_labels,
        epochs=EPOCHS,
        batch_size=batch_size,
        lr=LR,
        device=device,
        sample_size=sample_size,
        freeze_base=freeze_base,
    )
    detector.save(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
