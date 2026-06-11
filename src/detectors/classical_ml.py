from __future__ import annotations
from typing import Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from src.data.preprocessor import clean_text


class NaiveBayesDetector:
    """Naive Bayes text classifier for phishing detection."""

    def __init__(self, alpha: float = 1.0, ngram_range: tuple[int, int] = (1, 2)):
        self.pipeline = Pipeline([
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=ngram_range,
                    min_df=2,
                    stop_words="english",
                ),
            ),
            ("classifier", MultinomialNB(alpha=alpha)),
        ])
        self.is_fitted = False

    @staticmethod
    def _clean_texts(texts):
        return [clean_text(text) for text in texts]

    def fit(self, texts, labels):
        self.pipeline.fit(self._clean_texts(texts), labels)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> int:
        if not self.is_fitted:
            raise ValueError("NaiveBayesDetector must be fitted before calling predict()")
        return int(self.pipeline.predict([clean_text(text)])[0])

    def predict_batch(self, texts) -> list[int]:
        if not self.is_fitted:
            raise ValueError("NaiveBayesDetector must be fitted before calling predict_batch()")
        return [int(pred) for pred in self.pipeline.predict(self._clean_texts(texts))]

    def predict_proba(self, text: str) -> float:
        if not self.is_fitted:
            raise ValueError("NaiveBayesDetector must be fitted before calling predict_proba()")
        proba = self.pipeline.predict_proba([clean_text(text)])[0]
        return float(proba[1])

    def predict_proba_batch(self, texts) -> list[float]:
        if not self.is_fitted:
            raise ValueError("NaiveBayesDetector must be fitted before calling predict_proba_batch()")
        return [float(p[1]) for p in self.pipeline.predict_proba(self._clean_texts(texts))]


class LogisticRegressionDetector:
    """Classical logistic regression text classifier for phishing detection."""

    def __init__(
        self,
        max_iter: int = 1000,
        solver: Literal["liblinear"] = "liblinear",
        class_weight: str | None = "balanced",
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.pipeline = Pipeline([
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=ngram_range,
                    min_df=2,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    solver=solver,
                    class_weight=class_weight,
                ),
            ),
        ])
        self.is_fitted = False

    @staticmethod
    def _clean_texts(texts):
        return [clean_text(text) for text in texts]

    def fit(self, texts, labels):
        self.pipeline.fit(self._clean_texts(texts), labels)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> int:
        if not self.is_fitted:
            raise ValueError("LogisticRegressionDetector must be fitted before calling predict()")
        return int(self.pipeline.predict([clean_text(text)])[0])

    def predict_batch(self, texts) -> list[int]:
        if not self.is_fitted:
            raise ValueError("LogisticRegressionDetector must be fitted before calling predict_batch()")
        return [int(pred) for pred in self.pipeline.predict(self._clean_texts(texts))]

    def predict_proba(self, text: str) -> float:
        if not self.is_fitted:
            raise ValueError("LogisticRegressionDetector must be fitted before calling predict_proba()")
        proba = self.pipeline.predict_proba([clean_text(text)])[0]
        return float(proba[1])

    def predict_proba_batch(self, texts) -> list[float]:
        if not self.is_fitted:
            raise ValueError("LogisticRegressionDetector must be fitted before calling predict_proba_batch()")
        return [float(p[1]) for p in self.pipeline.predict_proba(self._clean_texts(texts))]
