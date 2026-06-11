# Phishing Email Detection Report
**Authors:** Marta Kotłowska, Maciej Jurczyk

## 1. Project Overview
This report summarizes the updated phishing email detection evaluation for the current project. The system compares multiple detector families across the same held-out test set, including rule-based patterns, classical machine learning, transformer models, URL analysis, and hybrid ensembles.

## 2. Dataset
The evaluation uses the dataset in `data/raw/PhishingEmail.csv`. The original dataset comprises 18,650 labeled email bodies with binary phishing labels.

## 3. System Architecture
| Module | Responsibility |
|---|---|
| `src/data/loader.py` | CSV ingestion, label encoding, stratified train/test split |
| `src/data/preprocessor.py` | Text cleaning and URL extraction |
| `src/detectors/rule_based.py` | Rule-based pattern matching |
| `src/detectors/classical_ml.py` | Naive Bayes and Logistic Regression models |
| `src/detectors/huggingface_detector.py` | BERT-based transformer phishing classifier |
| `src/detectors/distilbert_detector.py` | DistilBERT, DeBERTa, and RoBERTa detectors |
| `src/detectors/url_detector.py` | URL quality and phishing heuristics |
| `src/detectors/hybrid_detector.py` | Ensemble and voting strategies |
| `src/evaluation/evaluator.py` | Metrics, confusion matrix, and summary output |

## 4. Detector Configurations
The current evaluation includes the following detectors:

- **Rule-Based**
- **Naive Bayes**
- **Logistic Regression**
- **HuggingFace (BERT)**
- **DistilBERT**
- **DeBERTa**
- **RoBERTa**
- **URL Analyzer**
- **Hybrid**
- **Strict Hybrid**
- **Hybrid (Rule+URL)**

## 5. Experimental Results
The table below presents the updated metrics from `results/summary.txt`.

| Detector | Accuracy | Precision | Recall | F1 score | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| Rule-Based | 0.724443 | 0.810271 | 0.388509 | 0.525196 | 2132 | 133 | 894 | 568 |
| Naive Bayes | 0.931312 | 0.997525 | 0.826949 | 0.904263 | 2262 | 3 | 253 | 1209 |
| Logistic Regression | 0.968339 | 0.936364 | 0.986320 | 0.960693 | 2167 | 98 | 20 | 1442 |
| HuggingFace (BERT) | 0.977998 | 0.998555 | 0.945280 | 0.971188 | 2263 | 2 | 80 | 1382 |
| DistilBERT | 0.607727 | 0.000000 | 0.000000 | 0.000000 | 2265 | 0 | 1462 | 0 |
| DeBERTa | 0.607727 | 0.000000 | 0.000000 | 0.000000 | 2265 | 0 | 1462 | 0 |
| RoBERTa | 0.384223 | 0.376593 | 0.869357 | 0.525532 | 161 | 2104 | 191 | 1271 |
| URL Analyzer | 0.505500 | 0.369431 | 0.368673 | 0.369052 | 1345 | 920 | 923 | 539 |
| Hybrid | 0.977998 | 0.998555 | 0.945280 | 0.971188 | 2263 | 2 | 80 | 1382 |
| Strict Hybrid | 0.676952 | 0.974265 | 0.181259 | 0.305652 | 2258 | 7 | 1197 | 265 |
| Hybrid (Rule+URL) | 0.614167 | 0.760870 | 0.023940 | 0.046419 | 2254 | 11 | 1427 | 35 |

## 6. Key Observations
- **Best overall performance:** `HuggingFace (BERT)` and `Hybrid` both achieve the highest accuracy (97.80%) and F1 score (97.12%).
- **Strong classical baseline:** `Logistic Regression` performs well with 96.83% accuracy and 96.07% F1. `Naive Bayes` also achieves solid results at 93.13% accuracy.
- **Rule-Based has high precision but low recall**, indicating it is conservative and misses many phishing emails.
- **URL-only analysis remains weak** on this dataset, with near-random accuracy and low recall.
- **Strict Hybrid and Hybrid (Rule+URL)** show the cost of conservative ensemble designs: strong precision but poor phishing recall in the current test set.

## 7. Notes
- This report was generated from the latest `results/summary.txt` metrics and the current detector implementation in `main.py`.