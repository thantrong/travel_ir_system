import argparse
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from database.mongo_connection import get_database


def fetch_reviews_for_indexing(min_tokens: int = 1) -> list[dict]:
    db = get_database()
    reviews_col = db["reviews"]
    cursor = reviews_col.find(
        {"tokens": {"$exists": True, "$type": "array"}},
        {
            "_id": 0,
            "review_id": 1,
            "source": 1,
            "source_hotel_id": 1,
            "hotel_name": 1,
            "location": 1,
            "rating": 1,
            "review_rating": 1,
            "review_text": 1,
            "tokens": 1,
        },
    )

    rows = []
    for row in cursor:
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list):
            continue
        token_values = [str(tok).strip() for tok in tokens if str(tok).strip()]
        if len(token_values) < min_tokens:
            continue
        row["tokens"] = token_values
        rows.append(row)
    return rows


def build_index_payload(rows: list[dict]) -> dict:
    tokenized_corpus = [row["tokens"] for row in rows]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    documents = []
    for row in rows:
        documents.append(
            {
                "review_id": row.get("review_id", ""),
                "source": row.get("source", ""),
                "source_hotel_id": row.get("source_hotel_id", ""),
                "hotel_name": row.get("hotel_name", ""),
                "location": row.get("location", ""),
                "rating": row.get("rating", ""),
                "review_rating": row.get("review_rating", ""),
                "review_text": row.get("review_text", ""),
                "tokens": row.get("tokens", []),
            }
        )
    return {
        "bm25": bm25,
        "documents": documents,
        "corpus_size": len(documents),
    }


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from MongoDB reviews")
    parser.add_argument("--output", default="data/index/bm25_index.pkl", help="Output BM25 index path")
    parser.add_argument("--min-tokens", type=int, default=1, help="Minimum tokens per review")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = fetch_reviews_for_indexing(min_tokens=max(1, int(args.min_tokens)))
    payload = build_index_payload(rows)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Indexed reviews: {payload['corpus_size']}")
    print(f"Saved BM25 index: {output_path}")


if __name__ == "__main__":
    main()
