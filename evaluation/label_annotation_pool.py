from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi
from retrieval.query_understanding import understand_query, location_matched


POOL_PATH = PROJECT_ROOT / "evaluation" / "annotation_pool.tsv"
STOPWORDS_PATH = PROJECT_ROOT / "config" / "stopwords.txt"


GENERIC_TERMS = {
    "khách_sạn",
    "hotel",
    "resort",
    "homestay",
    "nhà_nghỉ",
    "chỗ_nghỉ",
    "chỗ_ở",
    "phòng",
    "view",
}


def _forms(token: str) -> set[str]:
    t = (token or "").strip().lower()
    if not t:
        return set()
    out = {t}
    if "_" in t:
        out.update(p for p in t.split("_") if p)
    return out


def _token_set(text: str) -> set[str]:
    norm = normalize_text(text or "")
    toks = tokenize_vi(norm)
    out: set[str] = set()
    for tok in toks:
        out.update(_forms(tok))
    return out


def _label_row(query: str, hotel_name: str, location: str, reviews: list[str]) -> tuple[str, str]:
    qu = understand_query(query, stopwords_path=STOPWORDS_PATH)
    blob = " ".join([hotel_name, location] + reviews)
    text_tokens = _token_set(blob)

    # 1) Location gate: query có location mà sai location thì loại.
    if qu.detected_location and not location_matched(qu.detected_location, location):
        return "0", "location_mismatch"

    descriptors = []
    for d in qu.descriptor_tokens:
        low = d.lower().strip()
        if low and low not in GENERIC_TERMS:
            descriptors.append(low)

    # 2) Nếu query không có descriptor rõ, giữ relevant theo location.
    if not descriptors:
        return "1", "location_or_generic_match"

    # 3) Match descriptor theo token forms.
    hits = 0
    for d in descriptors:
        if _forms(d).intersection(text_tokens):
            hits += 1

    # 4) Với query nhiều intent, yêu cầu hit mạnh hơn.
    need = 1
    if len(descriptors) >= 4:
        need = 2
    if len(descriptors) >= 7:
        need = 3

    if hits >= need:
        return "1", f"descriptor_hit_{hits}/{len(descriptors)}"
    return "0", f"descriptor_miss_{hits}/{len(descriptors)}"


def main() -> None:
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Missing pool file: {POOL_PATH}")

    with POOL_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        headers = reader.fieldnames or []

    pos = 0
    neg = 0
    for row in rows:
        query = row.get("query", "")
        hotel_name = row.get("hotel_name", "")
        location = row.get("location", "")
        reviews = [
            row.get("top_review_1", ""),
            row.get("top_review_2", ""),
            row.get("top_review_3", ""),
        ]
        rel, note = _label_row(query, hotel_name, location, reviews)
        row["relevance"] = rel
        row["note"] = note
        if rel == "1":
            pos += 1
        else:
            neg += 1

    with POOL_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Labeled rows: {len(rows)}")
    print(f"Relevant=1: {pos}")
    print(f"Irrelevant=0: {neg}")


if __name__ == "__main__":
    main()
