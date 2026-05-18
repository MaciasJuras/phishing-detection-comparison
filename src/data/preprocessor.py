import re


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_urls(text: str) -> list[str]:
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)