import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def evaluate(name: str, y_true, y_pred) -> dict:
    metrics = {
        "detector": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["true_neg"] = int(cm[0][0])
    metrics["false_pos"] = int(cm[0][1])
    metrics["false_neg"] = int(cm[1][0])
    metrics["true_pos"] = int(cm[1][1])

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={metrics['true_neg']}  FP={metrics['false_pos']}")
    print(f"    FN={metrics['false_neg']}  TP={metrics['true_pos']}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
        )
    )
    return metrics


def save_results(all_metrics: list[dict], output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Evaluator] Results saved to {csv_path}")

    txt_path = os.path.join(output_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(df.to_string(index=False))
    print(f"[Evaluator] Summary saved to {txt_path}")