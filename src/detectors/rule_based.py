import re
from src.data.preprocessor import clean_text


URGENT_PATTERNS = [
    r"\burgent\b", r"\bimmediately\b", r"\bact now\b", r"\bexpires?\b",
    r"\blimited time\b", r"\bdeadline\b", r"\bfinal notice\b", r"\blast chance\b",
    r"\baction required\b", r"\byour account.{0,20}suspend", r"\bwithin \d+ hours?\b",
]

CREDENTIAL_PATTERNS = [
    r"\bverify your (account|identity|email|password)\b",
    r"\bconfirm your (details|information|credentials)\b",
    r"\benter your (password|username|login|credentials)\b",
    r"\bupdate your (payment|billing|account) (info|information|details)\b",
    r"\bprovide your\b",
]

SUSPICIOUS_PHRASES = [
    r"\byou('ve| have) been selected\b",
    r"\bcongratulations\b",
    r"\byou('ve| have) won\b",
    r"\bfree (gift|offer|prize|money|iphone|ipad)\b",
    r"\bclaim your\b",
    r"\bnigerian\b", r"\blottery\b", r"\binheritance\b",
    r"\bclick (here|below|the link)\b",
    r"\bdo not (ignore|delete) this\b",
    r"\bsuspicious activity\b",
    r"\bunusual sign.?in\b",
]

IMPERSONATION_PATTERNS = [
    r"\b(paypal|amazon|apple|microsoft|google|netflix|bank of america|chase|wellsfargo)\b",
    r"\bdear (customer|user|member|account holder|valued)\b",
    r"\bIT (support|department|helpdesk|team)\b",
]


def _count_pattern_hits(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


class RuleBasedDetector:
    def __init__(
            self,
            urgent_weight: float = 2.0,
            credential_weight: float = 3.0,
            suspicious_weight: float = 1.5,
            impersonation_weight: float = 1.0,
            threshold: float = 3.0,
    ):
        self.weights = {
            "urgent": urgent_weight,
            "credential": credential_weight,
            "suspicious": suspicious_weight,
            "impersonation": impersonation_weight,
        }
        self.threshold = threshold

    def score(self, text: str) -> dict:
        text = clean_text(text)
        hits = {
            "urgent": _count_pattern_hits(text, URGENT_PATTERNS),
            "credential": _count_pattern_hits(text, CREDENTIAL_PATTERNS),
            "suspicious": _count_pattern_hits(text, SUSPICIOUS_PHRASES),
            "impersonation": _count_pattern_hits(text, IMPERSONATION_PATTERNS),
        }
        total_score = sum(hits[k] * self.weights[k] for k in hits)
        return {"score": total_score, "hits": hits}

    def predict(self, text: str) -> int:
        return int(self.score(text)["score"] >= self.threshold)

    def predict_proba(self, text: str) -> float:
        """Normalized score [0, 1] for hybrid detector."""
        raw = self.score(text)["score"]
        return min(raw / 10.0, 1.0)

    def predict_batch(self, texts) -> list[int]:
        return [self.predict(t) for t in texts]