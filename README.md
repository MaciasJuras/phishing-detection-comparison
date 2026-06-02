# Phishing Email Detection

A cybersecurity project comparing four phishing email detection approaches on a shared labelled dataset. Each detector is evaluated independently, with results saved to `results/`.

## Detectors

| # | Detector | Approach | F1 Score |
|---|----------|----------|----------|
| 1 | **Rule-Based** | Weighted regex patterns (urgent language, credential requests, impersonation, spam) | 52.5% |
| 2 | **HuggingFace (BERT)** | Fine-tuned BERT model (`ealvaradob/bert-finetuned-phishing`) | 97.1% |
| 3 | **URL Analyzer** | Heuristic scoring of extracted URLs (25 features) | 36.9% |
| 4a | **Hybrid (Weighted)** | Weighted ensemble of all three (weights: 0.20 / 0.55 / 0.25, threshold 0.45) | 97.1% |
| 4b | **Strict Hybrid** | Consensus voting — phishing only if ≥2 detectors exceed 0.35 | 30.6% |
| 4c | **Hybrid (Rule+URL)** | Weighted ensemble without BERT (weights: 0.60 / 0.40, threshold 0.5) | 4.6% |

## Dataset

- **Source:** `data/raw/Phishing_Email.csv`
- **Size:** 18,650 emails — 11,322 safe (60.7%), 7,328 phishing (39.3%)
- **Split:** 80/20 stratified train/test (random_state=42)
- **Features used:** Email body text only (no subject line or headers)

## Project Structure

```
phishing-detection/
├── data/
│   └── raw/
│       └── Phishing_Email.csv      # Labelled dataset
├── src/
│   ├── data/
│   │   ├── loader.py               # CSV ingestion, label encoding, stratified split
│   │   └── preprocessor.py        # Text cleaning, URL extraction
│   ├── detectors/
│   │   ├── rule_based.py          # Weighted pattern-matching detector
│   │   ├── huggingface_detector.py # BERT-based classifier wrapper
│   │   ├── url_detector.py        # URL heuristic risk scorer
│   │   └── hybrid_detector.py     # Weighted ensemble + strict voting mode
│   └── evaluation/
│       └── evaluator.py           # Accuracy, precision, recall, F1, confusion matrix
├── results/
│   ├── comparison.csv              # Per-detector metrics (machine-readable)
│   └── summary.txt                 # Per-detector metrics (human-readable)
├── main.py                         # Pipeline entry point
└── requirements.txt
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the BERT model (first run only — requires internet access):
   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
   AutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing")
   ```
   Subsequent runs use the local cache (`HF_HUB_OFFLINE=1` is set automatically).

3. Place the dataset at `data/raw/Phishing_Email.csv`.

4. Run all detectors:
   ```bash
   python main.py
   ```

Results are saved to `results/comparison.csv` and `results/summary.txt`.

## Results Summary

| Detector | Accuracy | Precision | Recall | F1 |
|----------|----------|-----------|--------|----|
| Rule-Based | 72.4% | 81.0% | 38.9% | 52.5% |
| HuggingFace (BERT) | **97.8%** | **99.9%** | 94.5% | **97.1%** |
| URL Analyzer | 50.6% | 36.9% | 36.9% | 36.9% |
| Hybrid (Weighted) | **97.8%** | **99.9%** | 94.5% | **97.1%** |
| Strict Hybrid | 67.7% | 97.4% | 18.1% | 30.6% |
| Hybrid (Rule+URL) | 61.4% | 76.1% | 2.4% | 4.6% |

Evaluated on 3,727 held-out test emails (2,265 legitimate + 1,462 phishing).

### Hybrid Variants Explained

Three hybrid strategies were implemented to isolate the contribution of each component:

- **Hybrid (Weighted)** — combines all three detectors with weights 0.20 (rule) / 0.55 (BERT) / 0.25 (URL). Matches BERT's performance, showing BERT dominates the ensemble.
- **Strict Hybrid** — classifies as phishing only when at least 2 of 3 detectors independently exceed a 0.35 probability threshold. Maximises precision (97.4%) at the cost of very low recall (18.1%).
- **Hybrid (Rule+URL)** — intentionally excludes BERT to evaluate how much the rule-based and URL detectors contribute on their own (weights: 0.60 rule / 0.40 URL). The near-zero recall (2.4%) confirms that without BERT, text-only heuristics cannot reliably detect phishing in this dataset.
