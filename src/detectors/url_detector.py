class URLDetector:
    def __init__(self, url_score_threshold: float = 3.0, no_url_score: float = 0.3):
        self.url_score_threshold = url_score_threshold
        self.no_url_score = no_url_score