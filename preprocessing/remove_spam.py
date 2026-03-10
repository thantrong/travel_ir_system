import re


REPEATED_CHAR_RE = re.compile(r"(.)\1{7,}")


def is_spam_review(text: str) -> bool:
    """Simple spam heuristics for noisy travel reviews."""
    if not text:
        return True
    value = text.strip()
    if len(value) < 8:
        return True
    if REPEATED_CHAR_RE.search(value):
        return True
    lowered = value.lower()
    spam_phrases = (
        "liên hệ",
        "hotline",
        "promo code",
        "mã giảm giá",
    )
    return any(phrase in lowered for phrase in spam_phrases)
