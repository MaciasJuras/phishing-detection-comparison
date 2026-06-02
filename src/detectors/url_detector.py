
import ipaddress
import re
from urllib.parse import parse_qs, urlparse
import tldextract

from src.data.preprocessor import clean_text, extract_urls


SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "top", "gq", "tk", "ml", "cf", "ga",
    "work", "click", "link", "fit", "xyz", "pw", "cc", "su",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "shorturl.at", "cutt.ly", "rb.gy",
}

SUSPICIOUS_PATH_KEYWORDS = {
    "login", "verify", "secure", "account", "update", "password",
    "signin", "auth", "webscr", "confirm", "validate", "billing",
}

REDIRECT_PARAM_KEYS = {
    "url", "redirect", "next", "continue", "return", "dest", "destination", "goto",
}

BRAND_KEYWORDS = {
    "paypal", "amazon", "apple", "microsoft", "google", "netflix",
    "instagram", "facebook", "chase", "wellsfargo", "bankofamerica",
}

SUSPICIOUS_PORTS = {
    8080, 8443, 8888, 9090, 3000, 4000, 5000, 7777,
}


class URLDetector:
    """Heuristic URL risk detector for phishing signals."""

    def __init__(self, url_score_threshold: float = 1.0, no_url_score: float = 0.3):
        self.url_score_threshold = url_score_threshold
        self.no_url_score = no_url_score

    @staticmethod
    def _safe_hostname(parsed) -> str:
        return (parsed.hostname or "").lower().strip(".")

    @staticmethod
    def _is_ip_host(hostname: str) -> bool:
        """Return True if the hostname is a bare IP address (v4 or v6)."""
        if not hostname:
            return False
        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            return False

    def score_url(self, url: str) -> tuple[float, dict]:
        """Return URL score and fired feature flags."""
        features: dict[str, bool] = {}
        score = 0.0

        raw_lower = url.lower()
        if raw_lower.startswith("hxxps://"):
            normalised = "https://" + url[8:]
        elif raw_lower.startswith("hxxp://"):
            normalised = "http://" + url[7:]
        else:
            normalised = url if raw_lower.startswith("http") else "http://" + url
        parsed = urlparse(normalised)
        hostname = self._safe_hostname(parsed)
        path_lower = (parsed.path or "").lower()
        query = parsed.query or ""

        features["at_in_netloc"] = "@" in parsed.netloc
        if features["at_in_netloc"]:
            score += 2.0

        pct_count = len(re.findall(r"%[0-9a-fA-F]{2}", url))
        features["excessive_encoding"] = pct_count >= 3
        if features["excessive_encoding"]:
            score += 1.0

        features["double_slash_in_path"] = bool(re.search(r"/{2,}", parsed.path or ""))
        if features["double_slash_in_path"]:
            score += 1.0

        original_scheme = parsed.scheme.lower() if parsed.scheme else "http"
        features["http_not_https"] = original_scheme != "https"
        if features["http_not_https"]:
            score += 1.0

        port = parsed.port
        features["suspicious_port"] = port is not None and port in SUSPICIOUS_PORTS
        if features["suspicious_port"]:
            score += 1.5

        features["ip_as_hostname"] = self._is_ip_host(hostname)
        if features["ip_as_hostname"]:
            score += 2.5

        ext = tldextract.extract(hostname)
        full_domain = ".".join(p for p in [ext.subdomain, ext.domain, ext.suffix] if p)
        registered_domain = ".".join(p for p in [ext.domain, ext.suffix] if p)

        features["url_shortener"] = registered_domain in URL_SHORTENERS
        if features["url_shortener"]:
            score += 2.0

        features["suspicious_tld"] = ext.suffix.lower() in SUSPICIOUS_TLDS
        if features["suspicious_tld"]:
            score += 1.5

        subdomain_parts = [p for p in ext.subdomain.split(".") if p]
        features["deep_subdomain"] = len(subdomain_parts) >= 3
        if features["deep_subdomain"]:
            score += 1.0

        features["punycode_idn"] = "xn--" in full_domain
        if features["punycode_idn"]:
            score += 2.0

        features["hyphen_in_domain"] = "-" in ext.domain
        if features["hyphen_in_domain"]:
            score += 0.5

        digit_ratio = sum(c.isdigit() for c in full_domain) / max(len(full_domain), 1)
        features["high_digit_ratio"] = digit_ratio >= 0.25
        if features["high_digit_ratio"]:
            score += 1.0

        brand_in_domain = any(b in full_domain for b in BRAND_KEYWORDS)
        brand_is_owner = ext.domain.lower() in BRAND_KEYWORDS
        features["brand_spoofing"] = brand_in_domain and not brand_is_owner
        if features["brand_spoofing"]:
            score += 2.0

        url_len = len(url)
        features["very_long_url"] = url_len >= 120
        features["long_url"] = 75 <= url_len < 120
        if features["very_long_url"]:
            score += 2.0
        elif features["long_url"]:
            score += 1.0

        features["suspicious_path_keyword"] = any(
            kw in path_lower for kw in SUSPICIOUS_PATH_KEYWORDS
        )
        if features["suspicious_path_keyword"]:
            score += 1.0

        query_dict = parse_qs(query, keep_blank_values=True)
        features["many_query_params"] = len(query_dict) >= 5
        if features["many_query_params"]:
            score += 1.0

        features["open_redirect_param"] = any(
            k.lower() in REDIRECT_PARAM_KEYS for k in query_dict
        )
        if features["open_redirect_param"]:
            score += 1.5

        return score, features

    def score(self, text: str) -> dict:
        """Score all URLs in text and return aggregate risk details."""
        text = clean_text(text)
        urls = extract_urls(text)

        if not urls:
            return {
                "score": self.no_url_score,
                "max_url_score": 0.0,
                "url_count": 0,
                "top_url": None,
                "url_scores": [],
            }

        results = [self.score_url(u) for u in urls]
        scores = [r[0] for r in results]

        max_idx = max(range(len(scores)), key=scores.__getitem__)
        max_score = scores[max_idx]

        suspicious_count = sum(1 for s in scores if s >= self.url_score_threshold)
        aggregate_bonus = sum(
            s * 0.10
            for i, s in enumerate(scores)
            if s >= self.url_score_threshold and i != max_idx
        )
        final_score = max_score + aggregate_bonus

        return {
            "score": final_score,
            "max_url_score": max_score,
            "url_count": len(urls),
            "suspicious_url_count": suspicious_count,
            "top_url": urls[max_idx],
            "top_url_features": results[max_idx][1],
            "url_scores": [(urls[i], scores[i]) for i in range(len(urls))],
        }

    def predict(self, text: str) -> int:
        return int(self.score(text)["score"] >= self.url_score_threshold)

    def predict_proba(self, text: str) -> float:
        """Normalised score in [0, 1] for use in the HybridDetector."""
        raw = self.score(text)["score"]
        return min(raw / 10.0, 1.0)

    def predict_batch(self, texts) -> list[int]:
        return [self.predict(t) for t in texts]