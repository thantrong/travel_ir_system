import re


TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹ0-9_]+")

try:
    from underthesea import word_tokenize  # type: ignore
    HAS_UNDERTHESEA = True
except Exception:
    HAS_UNDERTHESEA = False


def tokenize_vi(text: str) -> list[str]:
    if not text:
        return []
    if HAS_UNDERTHESEA:
        return [tok for tok in word_tokenize(text, format="text").split() if tok]
    return TOKEN_RE.findall(text.lower())
