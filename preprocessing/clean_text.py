import re


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_review_text(text: str) -> str:
    """Clean raw review text while preserving Vietnamese characters."""
    if not text:
        return ""
    value = str(text).strip()
    value = URL_RE.sub(" ", value)
    value = value.replace("\u200b", " ")
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()
