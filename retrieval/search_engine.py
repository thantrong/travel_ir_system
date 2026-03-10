import argparse
import json
import pickle
from pathlib import Path

from retrieval.query_processing import process_query


def load_index(index_path: Path) -> dict:
    with index_path.open("rb") as f:
        return pickle.load(f)


def search_top_hotels(index_payload: dict, query_tokens: list[str], top_k: int = 10) -> list[dict]:
    bm25 = index_payload.get("bm25")
    documents = index_payload.get("documents", [])
    if not bm25 or not query_tokens or not documents:
        return []

    scores = bm25.get_scores(query_tokens)
    hotel_scores = {}
    for idx, score in enumerate(scores):
        doc = documents[idx]
        source = str(doc.get("source", "")).strip().lower()
        source_hotel_id = str(doc.get("source_hotel_id", "")).strip()
        if not source_hotel_id:
            continue
        key = f"{source}|{source_hotel_id}"
        if key not in hotel_scores:
            hotel_scores[key] = {
                "source": source,
                "source_hotel_id": source_hotel_id,
                "hotel_name": doc.get("hotel_name", ""),
                "location": doc.get("location", ""),
                "rating": doc.get("rating", ""),
                "score_sum": 0.0,
                "score_max": float("-inf"),
                "review_count": 0,
                "top_reviews": [],
            }
        item = hotel_scores[key]
        score_value = float(score)
        item["score_sum"] += score_value
        item["score_max"] = max(item["score_max"], score_value)
        item["review_count"] += 1
        if len(item["top_reviews"]) < 3:
            item["top_reviews"].append(
                {
                    "review_id": doc.get("review_id", ""),
                    "review_text": doc.get("review_text", ""),
                    "review_rating": doc.get("review_rating", ""),
                    "bm25_score": round(score_value, 4),
                }
            )

    ranked = []
    for item in hotel_scores.values():
        avg_score = item["score_sum"] / max(1, item["review_count"])
        blended = 0.7 * item["score_max"] + 0.3 * avg_score
        ranked.append(
            {
                "source": item["source"],
                "source_hotel_id": item["source_hotel_id"],
                "hotel_name": item["hotel_name"],
                "location": item["location"],
                "rating": item["rating"],
                "bm25_score": round(blended, 4),
                "matched_reviews": item["review_count"],
                "top_reviews": item["top_reviews"],
            }
        )
    ranked.sort(key=lambda x: x["bm25_score"], reverse=True)
    return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Search top hotels with BM25 index")
    parser.add_argument("--query", required=True, help="Vietnamese user query")
    parser.add_argument("--index-path", default="data/index/bm25_index.pkl", help="BM25 index path")
    parser.add_argument("--top-k", type=int, default=10, help="Top K hotels")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    index_path = (project_root / args.index_path).resolve()
    stopwords_path = project_root / "config" / "stopwords.txt"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    query_tokens = process_query(args.query, stopwords_path)
    payload = load_index(index_path)
    results = search_top_hotels(payload, query_tokens, top_k=max(1, int(args.top_k)))
    print(json.dumps({"query_tokens": query_tokens, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
