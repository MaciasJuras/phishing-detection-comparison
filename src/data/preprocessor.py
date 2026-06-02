import re


TRAILING_URL_PUNCTUATION = ".,;:!?)]}\"'"


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_urls(text: str) -> list[str]:
    text = str(text)
    patterns = [
        r'(?:https?|hxxps?)://[^\s<>"{}|\\^`\[\]]+',
        r'(?:https?|hxxps?)\s*:\s*/\s*/\s*[^\s<>"{}|\\^`\[\]]+',
        r'\bwww\.[^\s<>"{}|\\^`\[\]]+',
        r'\bwww\s*\.\s*[^\s<>"{}|\\^`\[\]]+',
    ]
    urls = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            url = re.sub(r"\s+", "", str(match).strip()).strip(TRAILING_URL_PUNCTUATION)
            if not url:
                continue
            if url.lower().startswith("hxxps://"):
                url = "https://" + url[8:]
            elif url.lower().startswith("hxxp://"):
                url = "http://" + url[7:]
            key = url.lower()
            if key not in seen:
                seen.add(key)
                urls.append(url)
    return urls