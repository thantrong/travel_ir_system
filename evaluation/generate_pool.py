"""
Generate pool results for 200 test queries.
Output:
  - evaluation/pool_results.json (JSON)
  - evaluation/pool_results.csv (CSV)

Mục tiêu:
- lấy top 10 của từng nguồn/mô hình
- nếu cùng hotel_id xuất hiện ở nhiều nguồn thì union lại
- pool cuối cùng là union theo (query_id, hotel_id)
"""
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.search_engine import (
    load_index,
    encode_query,
    understand_query,
    _build_candidate_mask,
    _top_positive_indices,
)

INDEX_DIR = PROJECT_ROOT / "data" / "index"
STOPWORDS_PATH = PROJECT_ROOT / "config" / "stopwords.txt"
QUERIES_FILE = PROJECT_ROOT / "data" / "evaluation" / "test_queries_200_bucketed.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "evaluation" / "pool_results.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "evaluation" / "pool_results.csv"


def _get_top_k_by_score(docs, scores, k):
    if len(docs) == 0:
        return []
    positive = _top_positive_indices(scores, k)
    results = []
    for rank, idx in enumerate(positive[:k], 1):
        results.append({
            "rank": rank,
            "doc": docs[idx],
            "score": float(scores[idx]),
        })
    return results


def generate_pool():
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries")

    if not INDEX_DIR.exists():
        print(f"Error: Index not found at {INDEX_DIR}")
        return

    # Load indices
    bm25_payload = load_index(INDEX_DIR / "bm25_index.pkl")
    vec_payload = load_index(INDEX_DIR / "vector_index.pkl")

    bm25_model = bm25_payload["bm25"]
    bm25_docs = bm25_payload["documents"]
    embeddings = vec_payload["embeddings"]
    vec_docs = vec_payload["documents"]
    model_name = vec_payload["model_name"]

    # Load MongoDB để lấy review texts
    from database.mongo_connection import get_database
    db = get_database()
    reviews_col = db["reviews"]

    pool = {}
    total = len(queries)

    for i, q in enumerate(queries):
        query_id = q["query_id"]
        query_text = q["query"]
        bucket_id = q.get("bucket_id", q.get("bucket", ""))

        try:
            qu = understand_query(query_text, stopwords_path=STOPWORDS_PATH)
            bm25_mask = _build_candidate_mask(qu, bm25_docs)
            vec_mask = _build_candidate_mask(qu, vec_docs)

            # ===== BM25 top 10 =====
            search_tokens = qu.expanded_tokens if qu.expanded_tokens else qu.core_tokens
            bm25_scores = bm25_model.get_scores(search_tokens) if search_tokens else [0.0] * len(bm25_docs)
            if len(bm25_mask) == len(bm25_scores):
                bm25_scores = bm25_scores * bm25_mask.astype(float)
            bm25_scores = bm25_scores / bm25_scores.max() if getattr(bm25_scores, "max", None) and bm25_scores.max() > 0 else bm25_scores
            bm25_top = _get_top_k_by_score(bm25_docs, bm25_scores, 10)

            # ===== Vector top 10 =====
            q_vec = encode_query(query_text, model_name)
            if len(vec_mask) == len(embeddings):
                masked_idx = __import__("numpy").where(vec_mask)[0]
                vector_scores = __import__("numpy").zeros(len(embeddings), dtype=__import__("numpy").float32)
                if len(masked_idx) > 0:
                    vector_scores[masked_idx] = __import__("numpy").dot(embeddings[masked_idx], q_vec.T)
            else:
                vector_scores = __import__("numpy").dot(embeddings, q_vec.T)
            vector_scores = vector_scores / vector_scores.max() if vector_scores.max() > 0 else vector_scores
            vector_top = _get_top_k_by_score(vec_docs, vector_scores, 10)

            # ===== Hybrid top 10 =====
            hotel_b25 = defaultdict(float)
            hotel_vec = defaultdict(float)
            hotel_docs = {}
            hotel_names = {}
            hotel_locations = {}

            for item in bm25_top:
                doc = item["doc"]
                hid = doc.get("source_hotel_id", "")
                idx = bm25_docs.index(doc) if doc in bm25_docs else -1
                if idx >= 0:
                    hotel_b25[hid] += float(bm25_scores[idx])
                hotel_docs[hid] = doc
                hotel_names[hid] = doc.get("hotel_name", "")
                hotel_locations[hid] = doc.get("location", "")

            for item in vector_top:
                doc = item["doc"]
                hid = doc.get("source_hotel_id", "")
                idx = vec_docs.index(doc) if doc in vec_docs else -1
                if idx >= 0:
                    hotel_vec[hid] += float(vector_scores[idx])
                hotel_docs[hid] = doc
                hotel_names[hid] = doc.get("hotel_name", "")
                hotel_locations[hid] = doc.get("location", "")

            all_hids = set(hotel_b25.keys()) | set(hotel_vec.keys())
            hotel_hybrid = {hid: 0.6 * hotel_b25.get(hid, 0) + 0.4 * hotel_vec.get(hid, 0) for hid in all_hids}
            hybrid_sorted = sorted(hotel_hybrid.items(), key=lambda x: x[1], reverse=True)[:10]
            hybrid_rank_map = {hid: rank for rank, (hid, _) in enumerate(hybrid_sorted, 1)}
            bm25_rank_map = {item["doc"].get("source_hotel_id", ""): item["rank"] for item in bm25_top}
            vector_rank_map = {item["doc"].get("source_hotel_id", ""): item["rank"] for item in vector_top}

            # ===== Union theo hotel_id từ top 10 của từng mô hình =====
            all_hids_pool = set(bm25_rank_map.keys()) | set(vector_rank_map.keys()) | set(hybrid_rank_map.keys())

            for hid in all_hids_pool:
                key = (query_id, hid)
                if key in pool:
                    continue

                doc = hotel_docs.get(hid, {})
                hotel_reviews = list(reviews_col.find({"source_hotel_id": hid}).limit(3)) if hid else []
                review_texts = []
                for rv in hotel_reviews:
                    text = rv.get("review_text", "") or rv.get("clean_text", "")
                    if text:
                        review_texts.append(text[:200])

                pool[key] = {
                    "bucket_id": bucket_id,
                    "query_id": query_id,
                    "query": query_text,
                    "hotel_id": hid,
                    "hotel_name": hotel_names.get(hid, doc.get("hotel_name", "")),
                    "hotel_location": hotel_locations.get(hid, doc.get("location", "")),
                    "bm25_rank": bm25_rank_map.get(hid, None),
                    "vector_rank": vector_rank_map.get(hid, None),
                    "hybrid_rank": hybrid_rank_map.get(hid, None),
                    "top_review_1": review_texts[0] if len(review_texts) > 0 else "",
                    "top_review_2": review_texts[1] if len(review_texts) > 1 else "",
                    "top_review_3": review_texts[2] if len(review_texts) > 2 else "",
                    "relevant": None,
                }

            print(f"[{i+1}/{total}] {query_id}: '{query_text[:50]}' -> BM25:{len(bm25_top)} Vector:{len(vector_top)} Hybrid:{len(hybrid_sorted)} Union:{len(all_hids_pool)}")

        except Exception as e:
            print(f"[{i+1}/{total}] {query_id}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    results = list(pool.values())

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"POOL GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Unique (query, hotel) pairs: {len(results)}")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_pool()
