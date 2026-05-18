<<<<<<< HEAD
# Phishing Detection

A machine learning project for detecting phishing emails using multiple detection approaches.

## Project Structure

```
phishing-detection/
├── data/
│   ├── raw/                        # Original downloaded CSV
│   └── processed/                  # Cleaned, split dataset
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Load & split dataset
│   │   └── preprocessor.py        # Clean text, extract fields
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── rule_based.py          # Detector 1
│   │   ├── huggingface_detector.py # Detector 2
│   │   ├── url_detector.py        # Detector 3
│   │   └── hybrid_detector.py     # Detector 4
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py           # Metrics, confusion matrix, report
├── results/                        # CSV + TXT outputs saved here
├── main.py                         # Entry point — runs all detectors
├── requirements.txt
├── .gitignore
└── README.md
```

## Detectors

1. **Rule-based Detector** - Traditional pattern matching approach
2. **HuggingFace Detector** - Pre-trained transformers model
3. **URL Detector** - Specialized URL analysis
4. **Hybrid Detector** - Combination of multiple approaches

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

Results will be saved in the `results/` directory.

=======
# phishing-detection-comparison
>>>>>>> e0283bf2cc042285d24c304a2a3b2565cffa0f1c
