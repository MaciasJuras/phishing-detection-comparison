import re
from src.data.preprocessor import clean_text


URGENT_PATTERNS = [
    r"\burgent\b", r"\bimmediately\b", r"\bact now\b", r"\bexpires?\b",
    r"\blimited time\b", r"\bdeadline\b", r"\bfinal notice\b", r"\blast chance\b",
    r"\baction required\b", r"\byour account.{0,20}suspend", r"\bwithin \d+ hours?\b",
    r"\bsecurity alert\b", r"\bunauthori[sz]ed\b", r"\bverify now\b",
    r"\brespond now\b", r"\bfailure to comply\b", r"\bimmediate action\b",
]

CREDENTIAL_PATTERNS = [
    r"\bverify your (account|identity|email|password)\b",
    r"\bconfirm your (details|information|credentials)\b",
    r"\benter your (password|username|login|credentials)\b",
    r"\bupdate your (payment|billing|account) (info|information|details)\b",
    r"\bprovide your\b",
    r"\breset your password\b", r"\bre-authenticate\b",
    r"\blog[ -]?in to (your )?account\b", r"\bvalidate your account\b",
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
    r"\bclick (the )?(secure )?link\b",
    r"\bopen the attachment\b", r"\battached invoice\b",
    r"\bpayment (failed|declined)\b", r"\bconfirm your payment\b",
]

IMPERSONATION_PATTERNS = [
    r"\b(paypal|amazon|apple|microsoft|google|netflix|bank of america|chase|wellsfargo)\b",
    r"\bdear (customer|user|member|account holder|valued)\b",
    r"\bIT (support|department|helpdesk|team)\b",
    r"\bsecurity team\b", r"\badmin team\b", r"\bservice desk\b",
]

SPAM_OFFER_PATTERNS = [
    r"\bguaranteed\b", r"\bdouble (the )?speed\b", r"\bfree trial\b",
    r"\bwork from home\b", r"\bopt[- ]?in\b", r"\bclick now\b",
    r"\bremove\b.+\bsubject\b", r"\bno prescription\b",
    r"\blowest prices?\b", r"\bmake \$?\d+\b", r"\bovernight shipping\b",
]

LEGIT_CONTEXT_PATTERNS = [
    r"\bmailing list\b", r"\bcall for papers\b", r"\bconference\b",
    r"\bsourceforge\b", r"\bupdates@\b", r"\bforwarded by\b",
    r"\benron\b", r"\brpm-list\b", r"\buniversity\b", r"\bacm\b",
]

OBFUSCATED_PHARMA_TERMS = [
    "viagra", "cialis", "xanax", "valium", "levitra", "hydrocodone", "ambien", "pharmacy",
]


def _count_pattern_hits(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _count_compact_hits(text: str, terms: list[str]) -> int:
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    return sum(1 for term in terms if term in compact)


class RuleBasedDetector:
    def __init__(
            self,
            urgent_weight: float = 2.0,
            credential_weight: float = 3.0,
            suspicious_weight: float = 1.5,
            impersonation_weight: float = 1.0,
            spam_offer_weight: float = 2.0,
            obfuscated_pharma_weight: float = 2.5,
            legit_context_weight: float = -0.8,
            threshold: float = 3.0,
    ):
        self.weights = {
            "urgent": urgent_weight,
            "credential": credential_weight,
            "suspicious": suspicious_weight,
            "impersonation": impersonation_weight,
            "spam_offer": spam_offer_weight,
            "obfuscated_pharma": obfuscated_pharma_weight,
            "legit_context": legit_context_weight,
        }
        self.threshold = threshold

    def score(self, text: str) -> dict:
        text = clean_text(text)
        hits = {
            "urgent": _count_pattern_hits(text, URGENT_PATTERNS),
            "credential": _count_pattern_hits(text, CREDENTIAL_PATTERNS),
            "suspicious": _count_pattern_hits(text, SUSPICIOUS_PHRASES),
            "impersonation": _count_pattern_hits(text, IMPERSONATION_PATTERNS),
            "spam_offer": _count_pattern_hits(text, SPAM_OFFER_PATTERNS),
            "obfuscated_pharma": _count_compact_hits(text, OBFUSCATED_PHARMA_TERMS),
            "legit_context": min(2, _count_pattern_hits(text, LEGIT_CONTEXT_PATTERNS)),
        }
        total_score = max(0.0, sum(hits[k] * self.weights[k] for k in hits))
        return {"score": total_score, "hits": hits}

    def predict(self, text: str) -> int:
        return int(self.score(text)["score"] >= self.threshold)

    def predict_proba(self, text: str) -> float:
        """Normalized score [0, 1] for hybrid detector."""
        raw = self.score(text)["score"]
        return min(raw / 10.0, 1.0)

    def predict_batch(self, texts) -> list[int]:
        return [self.predict(t) for t in texts]