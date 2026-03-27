"""
Build BM25 Index (Review-level)
PHẦN 2: DESIGN DATA REPRESENTATION (REVIEW-LEVEL)
PHẦN 3: XÂY DỰNG INDEXING PIPELINE (Weight: hotel_name=3, location=2, text=1)
"""

import argparse
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rank_bm25 import BM25Okapi

from database.mongo_connection import get_database


def _tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    import re
    return [tok for tok in re.split(r'\W+', text.lower()) if tok]


def _normalize_tag_token(tag: str) -> list[str]:
    value = str(tag or "").strip().lower().replace(" ", "_")
    return [value] if value else []


def _extend_tags(doc_tokens: list[str], tags: list[str], weight: int = 1) -> None:
    for tag in tags or []:
        token = str(tag or "").strip().lower().replace(" ", "_")
        if not token:
            continue
        doc_tokens.extend([token] * weight)


def fetch_reviews_for_indexing() -> list[dict]:
    """Kéo dữ liệu review và append thông tin hotel."""
    db = get_database()
    places_col = db["places"]
    reviews_col = db["reviews"]

    place_map = {}
    for doc in places_col.find():
        sid = doc.get("source_hotel_id", "")
        if sid:
            place_map[sid] = {
                "hotel_name": doc.get("name", ""),
                "location": doc.get("location", ""),
            }

    docs = []
    cursor = reviews_col.find({"tokens": {"$exists": True, "$type": "array"}})
    for row in cursor:
        sid = str(row.get("source_hotel_id", "")).strip()
        if sid not in place_map:
            continue
            
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list):
            continue
            
        valid_tokens = [str(tok).strip() for tok in tokens if str(tok).strip()]
        if not valid_tokens:
            continue
            
        docs.append({
            "review_id": str(row.get("_id", row.get("review_id", ""))).strip(),
            "source": row.get("source", ""),
            "source_hotel_id": sid,
            "hotel_name": place_map[sid]["hotel_name"],
            "location": place_map[sid]["location"],
            "rating": place_map[sid].get("rating", row.get("review_rating", "")),
            "review_rating": row.get("review_rating", ""),
            "review_text": row.get("review_text", ""),
            "tokens": valid_tokens,
            "category_tags": list(row.get("category_tags", []) or []),
            "descriptor_tags": list(row.get("descriptor_tags", []) or []),
            "hotel_type_tags": list(row.get("hotel_type_tags", []) or []),
        })
        
    return docs


def build_index_payload(reviews: list[dict]) -> dict:
    tokenized_corpus = []
    documents = []

    for r in reviews:
        doc_tokens = []
        
        # PHẦN 3: Field Weighting (hotel_name: 3, location: 2, text: 1)
        hname_tokens = _tokenize_text(r["hotel_name"])
        doc_tokens.extend(hname_tokens * 3)
        
        loc_tokens = _tokenize_text(r["location"])
        doc_tokens.extend(loc_tokens * 2)
        
        doc_tokens.extend(r["tokens"])

        # Tags mới được nhúng trực tiếp vào corpus để tăng khả năng khớp intent.
        _extend_tags(doc_tokens, r.get("category_tags", []), weight=2)
        _extend_tags(doc_tokens, r.get("descriptor_tags", []), weight=2)
        _extend_tags(doc_tokens, r.get("hotel_type_tags", []), weight=1)

        tokenized_corpus.append(doc_tokens)
        
        # Schema Document Review-level (PHẦN 2)
        documents.append({
            "_id": r["review_id"],
            "review_id": r["review_id"],
            "source_hotel_id": r["source_hotel_id"],
            "hotel_name": r["hotel_name"],
            "location": r["location"],
            "review_text": r["review_text"],
            "review_rating": r["review_rating"],
            "source": r["source"],
            "category_tags": list(r.get("category_tags", []) or []),
            "descriptor_tags": list(r.get("descriptor_tags", []) or []),
            "hotel_type_tags": list(r.get("hotel_type_tags", []) or []),
        })

    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    review_ids = [doc.get("review_id", "") for doc in documents]
    review_id_to_idx = {rid: idx for idx, rid in enumerate(review_ids) if rid}

    return {
        "bm25": bm25,
        "documents": documents,
        "review_ids": review_ids,
        "review_id_to_idx": review_id_to_idx,
        "corpus_size": len(documents),
    }


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index (Review-level)")
    parser.add_argument("--output", default="data/index/bm25_index.pkl", help="Output path")
    args = parser.parse_args()

    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviews = fetch_reviews_for_indexing()
    print(f"Bắt đầu index {len(reviews)} reviews...")
    payload = build_index_payload(reviews)
    
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Indexed reviews: {payload['corpus_size']}")
    print(f"Saved BM25 review-level index: {output_path}")


if __name__ == "__main__":
    main()
