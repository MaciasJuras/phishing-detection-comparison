
import os
import ipaddress
import re
from urllib.parse import parse_qs, urlparse
import tldextract
from src.data.preprocessor import clean_text, extract_urls


TLD_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "tldextract")
TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=TLD_CACHE_DIR, suffix_list_urls=())


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

SUSPICIOUS_URL_TOKENS = {
    "verify", "secure", "account", "update", "password", "signin",
    "login", "confirm", "validate", "billing", "bank", "payment",
    "invoice", "bonus", "winner", "prize", "offer", "loan", "mortgage",
    "casino", "adult", "porn", "sex", "viagra", "cialis", "xanax", "pharmacy",
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

TRAILING_URL_PUNCTUATION = ".,;:!?)]}\"'"

DEFAULT_FEATURE_WEIGHTS = {
    "at_in_netloc": 2.0,
    "excessive_encoding": 1.0,
    "double_slash_in_path": 1.0,
    "http_not_https": 0.8,
    "suspicious_port": 1.5,
    "malformed_port": 1.0,
    "ip_as_hostname": 2.5,
    "url_shortener": 2.0,
    "suspicious_tld": 1.5,
    "deep_subdomain": 1.0,
    "punycode_idn": 2.0,
    "hyphen_in_domain": 0.5,
    "high_digit_ratio": 1.0,
    "brand_spoofing": 2.0,
    "very_long_url": 2.0,
    "long_url": 1.0,
    "suspicious_path_keyword": 1.0,
    "many_query_params": 1.0,
    "open_redirect_param": 1.5,
    "suspicious_url_tokens": 1.8,
    "high_special_char_ratio": 1.2,
    "deep_path": 0.9,
    "benign_structure": -1.1,
}


class URLDetector:
    """Heuristic URL risk detector for phishing signals."""

    def __init__(
        self,
        url_score_threshold: float = 1.0,
        no_url_score: float = 0.3,
        feature_weights: dict[str, float] | None = None,
        multi_url_bonus: float = 0.10,
    ):
        self.url_score_threshold = url_score_threshold
        self.no_url_score = no_url_score
        self.feature_weights = dict(DEFAULT_FEATURE_WEIGHTS)
        if feature_weights:
            self.feature_weights.update(feature_weights)
        self.multi_url_bonus = multi_url_bonus

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

    @staticmethod
    def _safe_port(parsed):
        try:
            return parsed.port
        except ValueError:
            return None

    @staticmethod
    def _normalise_url(url: str) -> str:
        url = str(url or "").strip().strip(TRAILING_URL_PUNCTUATION)
        url = re.sub(r"\s+", "", url)

        raw_lower = url.lower()
        if raw_lower.startswith("hxxps://"):
            url = "https://" + url[8:]
        elif raw_lower.startswith("hxxp://"):
            url = "http://" + url[7:]
        elif raw_lower.startswith("www."):
            url = "http://" + url

        return url

    def _add_feature(
        self,
        features: dict[str, bool],
        feature_name: str,
        triggered: bool,
        score: float,
    ) -> float:
        features[feature_name] = bool(triggered)
        if triggered:
            score += self.feature_weights.get(feature_name, 0.0)
        return score

    def score_url(self, url: str) -> tuple[float, dict]:
        """Return URL score and fired feature flags."""
        features: dict[str, bool] = {}
        score = 0.0

        normalised = self._normalise_url(url)
        if not normalised:
            return 0.0, features
        if not normalised.lower().startswith(("http://", "https://")):
            normalised = "http://" + normalised

        parsed = urlparse(normalised)
        hostname = self._safe_hostname(parsed)
        path_lower = (parsed.path or "").lower()
        query = parsed.query or ""
        netloc_lower = (parsed.netloc or "").lower()

        score = self._add_feature(features, "at_in_netloc", "@" in parsed.netloc, score)

        pct_count = len(re.findall(r"%[0-9a-fA-F]{2}", normalised))
        score = self._add_feature(features, "excessive_encoding", pct_count >= 3, score)

        score = self._add_feature(
            features,
            "double_slash_in_path",
            bool(re.search(r"/{2,}", parsed.path or "")),
            score,
        )

        original_scheme = parsed.scheme.lower() if parsed.scheme else "http"
        score = self._add_feature(features, "http_not_https", original_scheme != "https", score)

        port = self._safe_port(parsed)
        score = self._add_feature(features, "suspicious_port", port is not None and port in SUSPICIOUS_PORTS, score)
        has_port_hint = ":" in parsed.netloc and port is None
        score = self._add_feature(features, "malformed_port", has_port_hint, score)

        score = self._add_feature(features, "ip_as_hostname", self._is_ip_host(hostname), score)

        ext = TLD_EXTRACTOR(hostname)
        full_domain = ".".join(p for p in [ext.subdomain, ext.domain, ext.suffix] if p)
        registered_domain = ".".join(p for p in [ext.domain, ext.suffix] if p)

        score = self._add_feature(features, "url_shortener", registered_domain in URL_SHORTENERS, score)

        score = self._add_feature(features, "suspicious_tld", ext.suffix.lower() in SUSPICIOUS_TLDS, score)

        subdomain_parts = [p for p in ext.subdomain.split(".") if p]
        score = self._add_feature(features, "deep_subdomain", len(subdomain_parts) >= 3, score)

        score = self._add_feature(features, "punycode_idn", "xn--" in full_domain, score)

        score = self._add_feature(features, "hyphen_in_domain", "-" in ext.domain, score)

        digit_ratio = sum(c.isdigit() for c in full_domain) / max(len(full_domain), 1)
        score = self._add_feature(features, "high_digit_ratio", digit_ratio >= 0.25, score)

        brand_in_domain = any(b in full_domain for b in BRAND_KEYWORDS)
        brand_is_owner = ext.domain.lower() in BRAND_KEYWORDS
        score = self._add_feature(features, "brand_spoofing", brand_in_domain and not brand_is_owner, score)

        url_len = len(normalised)
        score = self._add_feature(features, "very_long_url", url_len >= 120, score)
        score = self._add_feature(features, "long_url", 75 <= url_len < 120, score)

        score = self._add_feature(
            features,
            "suspicious_path_keyword",
            any(kw in path_lower for kw in SUSPICIOUS_PATH_KEYWORDS),
            score,
        )

        query_dict = parse_qs(query, keep_blank_values=True)
        score = self._add_feature(features, "many_query_params", len(query_dict) >= 5, score)

        score = self._add_feature(
            features,
            "open_redirect_param",
            any(k.lower() in REDIRECT_PARAM_KEYS for k in query_dict),
            score,
        )

        path_segments = [segment for segment in (parsed.path or "").split("/") if segment]
        score = self._add_feature(features, "deep_path", len(path_segments) >= 4, score)

        lexical_source = f"{netloc_lower} {path_lower} {(query or '').lower()}"
        lexical_tokens = re.findall(r"[a-z]{3,}", lexical_source)
        suspicious_token_hits = sum(1 for token in lexical_tokens if token in SUSPICIOUS_URL_TOKENS)
        score = self._add_feature(features, "suspicious_url_tokens", suspicious_token_hits >= 2, score)

        special_char_count = len(re.findall(r"[^a-zA-Z0-9]", (parsed.path or "") + query))
        base_len = len((parsed.path or "") + query)
        special_char_ratio = special_char_count / max(base_len, 1)
        score = self._add_feature(features, "high_special_char_ratio", special_char_ratio >= 0.35, score)

        # Dampen risk for clean, simple HTTPS URLs without phishing indicators.
        strong_flags = (
            features.get("ip_as_hostname", False)
            or features.get("at_in_netloc", False)
            or features.get("url_shortener", False)
            or features.get("suspicious_tld", False)
            or features.get("open_redirect_param", False)
            or features.get("brand_spoofing", False)
            or features.get("punycode_idn", False)
            or features.get("suspicious_port", False)
            or features.get("malformed_port", False)
        )
        benign_structure = (
            original_scheme == "https"
            and not strong_flags
            and not features.get("suspicious_path_keyword", False)
            and not features.get("suspicious_url_tokens", False)
            and len(path_segments) <= 2
            and len(query_dict) <= 1
            and special_char_ratio < 0.20
        )
        score = self._add_feature(features, "benign_structure", benign_structure, score)

        return score, features

    def score(self, text: str) -> dict:
        """Score all URLs in text and return aggregate risk details."""
        text = clean_text(text)
        urls = [self._normalise_url(url) for url in extract_urls(text)]
        urls = [url for url in urls if url]

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
            s * self.multi_url_bonus
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
