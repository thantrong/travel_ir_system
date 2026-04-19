"""
Build BM25 Index (Review-level)
PHẦN 2: DESIGN DATA REPRESENTATION (REVIEW-LEVEL)
PHẦN 3: XÂY DỰNG INDEXING PIPELINE (Weight: hotel_name=3, location=2, text=1)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rank_bm25 import BM25Okapi
from pymongo.errors import PyMongoError

from database.mongo_connection import get_collection_names, get_database
from nlp.tokenizer import tokenize_vi


def _load_processed_records() -> list[dict]:
    processed_jsonl = project_root / "data" / "processed" / "reviews_processed.jsonl"
    processed_json = project_root / "data" / "processed" / "reviews_processed.json"

    rows: list[dict] = []
    if processed_jsonl.exists():
        for line_no, line in enumerate(processed_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip malformed JSONL line {line_no} in {processed_jsonl}: {exc}")
                continue
            if isinstance(item, dict):
                rows.append(item)
        if rows:
            return rows

    if processed_json.exists():
        try:
            payload = json.loads(processed_json.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = [x for x in payload if isinstance(x, dict)]
        except json.JSONDecodeError as exc:
            print(f"[WARN] Invalid JSON in {processed_json}: {exc}")
            rows = []
    return rows


def _fetch_reviews_for_indexing_from_processed_files() -> list[dict]:
    rows = _load_processed_records()
    docs = []
    for row in rows:
        sid = str(row.get("source_hotel_id", "")).strip()
        if not sid:
            continue

        tokens_field = row.get("tokens")
        tokens = []
        if isinstance(tokens_field, list) and tokens_field:
            tokens = [str(tok).strip() for tok in tokens_field if str(tok).strip()]
        else:
            clean = str(row.get("clean_text", "")).strip()
            if clean:
                try:
                    tokens = tokenize_vi(clean)
                except Exception:
                    tokens = _tokenize_text(clean)
        if not tokens:
            continue

        review_id = str(row.get("review_id", row.get("_id", ""))).strip()
        if not review_id:
            continue

        docs.append({
            "review_id": review_id,
            "source": row.get("source", ""),
            "source_hotel_id": sid,
            "hotel_name": row.get("hotel_name", row.get("name", "")),
            "types": list(row.get("types", row.get("place_types", [])) or []),
            "location": row.get("location", ""),
            "rating": row.get("rating", row.get("review_rating", "")),
            "review_rating": row.get("review_rating", ""),
            "review_text": row.get("review_text", ""),
            "tokens": tokens,
            "category_tags": list(row.get("category_tags", []) or []),
            "descriptor_tags": list(row.get("descriptor_tags", []) or []),
        })
    return docs


def _tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    import re
    return [tok for tok in re.split(r'\W+', text.lower()) if tok]


def _normalize_tag_token(tag: str) -> list[str]:
    value = str(tag or "").strip().lower().replace(" ", "_")
    return [value] if value else []


# Danh sách các tiền tố tag tiêu cực cần loại bỏ khi indexing
_NEGATIVE_TAG_PREFIXES = ("!", "not_", "no_", "non_", "bad_", "poor_", "worst_")


def _is_negative_tag(tag: str) -> bool:
    """Kiểm tra xem tag có mang ý nghĩa tiêu cực/cảm xúc xấu không."""
    value = str(tag or "").strip().lower()
    if not value:
        return True
    # Tag bắt đầu bằng tiền tố phủ định
    if any(value.startswith(prefix) for prefix in _NEGATIVE_TAG_PREFIXES):
        return True
    # Tag có từ tiêu cực mạnh trong danh sách
    strong_negative_words = {
        "tệ", "tồi_tệ", "dở", "bẩn", "dơ", "ồn", "hôi",
        "thất_vọng", "không_đáng_tiền", "không_hài_lòng",
        "tệ_hại", "kinh_khủng", "thảm_họa", "rác_rưởi",
        "vỡi", "lừa_đảo", "phẫn_nộ", "gay_phẫn"
    }
    if value in strong_negative_words:
        return True
    return False


def _extend_tags(doc_tokens: list[str], tags: list[str], weight: int = 1) -> None:
    for tag in tags or []:
        token = str(tag or "").strip().lower().replace(" ", "_")
        if not token or _is_negative_tag(token):
            continue
        doc_tokens.extend([token] * weight)


def fetch_reviews_for_indexing() -> list[dict]:
    """Kéo dữ liệu review và append thông tin hotel."""
    try:
        db = get_database()
        collections = get_collection_names()
        places_col = db[collections["places"]]
        reviews_col = db[collections["reviews"]]

        place_map = {}
        for doc in places_col.find():
            sid = doc.get("_id", doc.get("source_hotel_id", ""))
            if sid:
                place_map[str(sid)] = {
                    "types": list(doc.get("types", []) or []),
                    "location": doc.get("location", ""),
                    "rating": doc.get("rating", ""),
                    "hotel_name": doc.get("name", doc.get("hotel_name", "")),
                }

        docs = []
        cursor = reviews_col.find({"$or": [{"tokens": {"$exists": True, "$type": "array"}}, {"clean_text": {"$exists": True}}]})
        for row in cursor:
            sid = str(row.get("source_hotel_id", "")).strip()
            if sid not in place_map:
                continue

            tokens_field = row.get("tokens")
            tokens = []
            if isinstance(tokens_field, list) and tokens_field:
                tokens = [str(tok).strip() for tok in tokens_field if str(tok).strip()]
            else:
                clean = str(row.get("clean_text", "")).strip()
                if clean:
                    try:
                        tokens = tokenize_vi(clean)
                    except Exception:
                        tokens = _tokenize_text(clean)

            if not tokens:
                continue

            docs.append({
                "review_id": str(row.get("_id", row.get("review_id", ""))).strip(),
                "source": row.get("source", ""),
                "source_hotel_id": sid,
                "hotel_name": place_map[sid].get("hotel_name", ""),
                "types": list(place_map[sid].get("types", []) or []),
                "location": place_map[sid]["location"],
                "rating": place_map[sid].get("rating", row.get("review_rating", "")),
                "review_rating": row.get("review_rating", ""),
                "review_text": row.get("review_text", ""),
                "tokens": tokens,
                "category_tags": list(row.get("category_tags", []) or []),
                "descriptor_tags": list(row.get("descriptor_tags", []) or []),
            })

        if docs:
            return docs
    except (ValueError, PyMongoError) as exc:
        print(f"MongoDB unavailable ({exc}), fallback sang data/processed...")
    except Exception as exc:
        print(f"MongoDB unexpected error ({exc}), fallback sang data/processed...")

    docs = _fetch_reviews_for_indexing_from_processed_files()
    if docs:
        print(f"Fallback indexing from processed files: {len(docs)} reviews")
    return docs


def build_index_payload(reviews: list[dict]) -> dict:
    tokenized_corpus = []
    documents = []

    for r in reviews:
        doc_tokens = []

        # PHẦN 3: Field Weighting (hotel name: 3,types: 2, location: 2, text: 1)
        _extend_tags(doc_tokens, [r.get("hotel_name", "")], weight=3)
        _extend_tags(doc_tokens, r.get("types", []), weight=2)

        loc_tokens = _tokenize_text(r["location"])
        doc_tokens.extend(loc_tokens * 2)  # location weight = 2

        # Use tokens provided or precomputed Vietnamese tokenizer tokens
        doc_tokens.extend(r.get("tokens", []))

        # Tags mới được nhúng trực tiếp vào corpus để tăng khả năng khớp intent.
        _extend_tags(doc_tokens, r.get("category_tags", []), weight=2)
        _extend_tags(doc_tokens, r.get("descriptor_tags", []), weight=2)

        tokenized_corpus.append(doc_tokens)

        # Schema Document Review-level (PHẦN 2)
        documents.append({
            "_id": r["review_id"],
            "review_id": r["review_id"],
            "source_hotel_id": r["source_hotel_id"],
            "hotel_name": r.get("hotel_name", ""),
            "types": list(r.get("types", []) or []),
            "location": r["location"],
            "review_text": r["review_text"],
            "review_rating": r["review_rating"],
            "source": r["source"],
            "category_tags": list(r.get("category_tags", []) or []),
            "descriptor_tags": list(r.get("descriptor_tags", []) or []),
        })

    bm25 = BM25Okapi(corpus=tokenized_corpus, k1=1.5, b=0.75) if tokenized_corpus else None

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
