# phishing-detection-comparison

A phishing detecting project across multiple detection approaches.

## Project Structure

```
phishing-detection-comparison/
├── data/
│   ├── raw/                        # Original downloaded CSV
│   └── processed/                  # Cleaned, split dataset
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Load & split dataset
│   │   └── preprocessor.py         # Clean text, extract fields
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── classical_ml.py         # Naive Bayes and Logistic Regression
│   │   ├── distilbert_detector.py   # DeBERTa
│   │   ├── huggingface_detector.py  # BERT-based transformer detector
│   │   ├── rule_based.py           # Rule-based detector
│   │   ├── url_detector.py         # URL analysis detector
│   │   └── hybrid_detector.py      # Hybrid ensemble detector
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py            # Metrics, confusion matrix, report
├── results/                        # CSV + TXT outputs saved here
├── main.py                         # Entry point — runs all detectors
├── requirements.txt
└── README.md
```

## Detectors

1. **Rule-Based** - Traditional pattern matching approach
2. **Naive Bayes** - Probabilistic text classification
3. **Logistic Regression** - Linear classification on text features
4. **HuggingFace (BERT)** - Pre-trained transformer model
5. **DeBERTa** - Enhanced transformer architecture
6. **URL Analyzer** - URL-specific phishing features
7. **Hybrid** - Ensemble of rule-based, transformer, and URL signals
8. **Strict Hybrid** - More conservative ensemble voting
9. **Hybrid (Rule+URL)** - Rule and URL detector combination

## Updated Results

The latest evaluation results are saved in `results/summary.txt`. The current comparison contains accuracy, precision, recall, F1 score, and confusion matrix counts for all detectors.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place raw data in `data/raw/`

3. Run the main script:
   ```bash
   python main.py
   ```

   To fine-tune DeBERTa separately and save it to `models/deberta-phishing`:
   ```bash
   python scripts/train_deberta.py
   ```

   Later runs of `python main.py` load `models/deberta-phishing` automatically.

4. Review the output in `results/summary.txt`.