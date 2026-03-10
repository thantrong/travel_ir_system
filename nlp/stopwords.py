from pathlib import Path


def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if token and not token.startswith("#"):
            values.add(token)
    return values


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    if not stopwords:
        return tokens
    return [tok for tok in tokens if tok.lower() not in stopwords]
