from pathlib import Path

from nlp.normalization import normalize_text
from nlp.stopwords import load_stopwords, remove_stopwords
from nlp.tokenizer import tokenize_vi


def process_query(query: str, stopwords_path: Path) -> list[str]:
    text = normalize_text(query or "")
    if not text:
        return []
    stopwords = load_stopwords(stopwords_path)
    tokens = tokenize_vi(text)
    return remove_stopwords(tokens, stopwords)
