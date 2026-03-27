from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def summarize_reviews_tfidf(reviews: Iterable[str], top_n: int = 3) -> list[str]:
    """Lightweight fallback extractive summarizer.

    The original project expects this symbol when importing the search engine.
    For the current refactor, we keep a minimal dependency-free version so the
    Streamlit app and retrieval pipeline can import cleanly even if the original
    summarizer implementation is absent.
    """
    texts = [str(r).strip() for r in reviews if str(r).strip()]
    if not texts:
        return []

    candidates: list[str] = []
    for text in texts:
        parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
        candidates.extend(parts)

    if not candidates:
        return texts[:top_n]

    freq = Counter()
    for sent in candidates:
        for tok in re.findall(r"[a-zA-ZÀ-ỹđ_]+", sent.lower()):
            if len(tok) >= 3:
                freq[tok] += 1

    scored = []
    for sent in candidates:
        score = 0
        for tok in re.findall(r"[a-zA-ZÀ-ỹđ_]+", sent.lower()):
            score += freq.get(tok, 0)
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = []
    seen = set()
    for _, sent in scored:
        if sent not in seen:
            seen.add(sent)
            result.append(sent)
        if len(result) >= top_n:
            break
    return result
