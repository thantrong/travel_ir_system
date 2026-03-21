"""
Build manual annotation pool for IR relevance labeling.

Output file:
- evaluation/annotation_pool.tsv

Each row is one (query, hotel) candidate pooled from top-k results of
BM25-only, Vector-only, and Hybrid runs, with review snippets for manual judging.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import search_engine as se  # noqa: E402


@dataclass
class PoolRow:
    query_id: str
    query: str
    hotel_id: str
    hotel_name: str = ""
    location: str = ""
    rank_bm25: str = ""
    rank_vector: str = ""
    rank_hybrid: str = ""
    score_bm25: str = ""
    score_vector: str = ""
    score_hybrid: str = ""
    retrieved_by: list[str] = field(default_factory=list)
    top_review_1: str = ""
    top_review_2: str = ""
    top_review_3: str = ""
    relevance: str = ""
    note: str = ""


def _sanitize(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).replace("\t", " ").replace("\n", " ").split())


@lru_cache(maxsize=4)
def _load_index_cached(path_str: str):
    return _ORIGINAL_LOAD_INDEX(Path(path_str))


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


def _encode_query_cached(query: str, model_name: str):
    model = _load_sentence_model(model_name)
    return model.encode([query])[0]


_ORIGINAL_LOAD_INDEX = se.load_index
se.load_index = lambda path: _load_index_cached(str(Path(path).resolve()))
se.encode_query = _encode_query_cached


def load_queries(path: Path, limit: int = 50) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, item in enumerate(raw[:limit], start=1):
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        out.append((f"Q{i}", q))
    return out


def run_search(query: str, index_dir: Path, stopwords_path: Path, vector_weight: float, bm25_weight: float):
    results, _ = se.search_hybrid(
        query=query,
        index_dir=index_dir,
        stopwords_path=stopwords_path,
        top_k=10,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        location_boost_factor=1.8,
    )
    return results


def main() -> None:
    queries_path = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
    index_dir = PROJECT_ROOT / "data" / "index"
    stopwords_path = PROJECT_ROOT / "config" / "stopwords.txt"
    out_path = PROJECT_ROOT / "evaluation" / "annotation_pool.tsv"

    queries = load_queries(queries_path, limit=50)
    if not queries:
        raise RuntimeError("Không có query hợp lệ trong test_queries.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "query_id",
        "query",
        "hotel_id",
        "hotel_name",
        "location",
        "retrieved_by",
        "rank_bm25",
        "rank_vector",
        "rank_hybrid",
        "score_bm25",
        "score_vector",
        "score_hybrid",
        "top_review_1",
        "top_review_2",
        "top_review_3",
        "relevance",
        "note",
    ]

    all_rows: list[PoolRow] = []

    for i, (qid, query) in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] Build pool for {qid}")

        bm25 = run_search(query, index_dir, stopwords_path, vector_weight=0.0, bm25_weight=1.0)
        vector = run_search(query, index_dir, stopwords_path, vector_weight=1.0, bm25_weight=0.0)
        hybrid = run_search(query, index_dir, stopwords_path, vector_weight=0.6, bm25_weight=0.4)

        pool: dict[str, PoolRow] = {}

        def absorb(results: list[dict], model_tag: str):
            for rank, r in enumerate(results[:10], start=1):
                hid = str(r.get("source_hotel_id", "")).strip()
                if not hid:
                    continue
                if hid not in pool:
                    top_reviews = r.get("top_reviews", []) or []
                    pool[hid] = PoolRow(
                        query_id=qid,
                        query=query,
                        hotel_id=hid,
                        hotel_name=_sanitize(str(r.get("hotel_name", ""))),
                        location=_sanitize(str(r.get("location", ""))),
                        top_review_1=_sanitize(top_reviews[0] if len(top_reviews) > 0 else ""),
                        top_review_2=_sanitize(top_reviews[1] if len(top_reviews) > 1 else ""),
                        top_review_3=_sanitize(top_reviews[2] if len(top_reviews) > 2 else ""),
                    )

                row = pool[hid]
                row.retrieved_by.append(model_tag)
                score = f"{float(r.get('hybrid_score', 0.0)):.6f}"
                if model_tag == "bm25":
                    row.rank_bm25 = str(rank)
                    row.score_bm25 = score
                elif model_tag == "vector":
                    row.rank_vector = str(rank)
                    row.score_vector = score
                elif model_tag == "hybrid":
                    row.rank_hybrid = str(rank)
                    row.score_hybrid = score

        absorb(bm25, "bm25")
        absorb(vector, "vector")
        absorb(hybrid, "hybrid")

        rows = list(pool.values())
        rows.sort(
            key=lambda x: (
                int(x.rank_hybrid) if x.rank_hybrid else 999,
                int(x.rank_vector) if x.rank_vector else 999,
                int(x.rank_bm25) if x.rank_bm25 else 999,
            )
        )
        all_rows.extend(rows)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        for row in all_rows:
            writer.writerow(
                [
                    row.query_id,
                    row.query,
                    row.hotel_id,
                    row.hotel_name,
                    row.location,
                    ",".join(sorted(set(row.retrieved_by))),
                    row.rank_bm25,
                    row.rank_vector,
                    row.rank_hybrid,
                    row.score_bm25,
                    row.score_vector,
                    row.score_hybrid,
                    row.top_review_1,
                    row.top_review_2,
                    row.top_review_3,
                    row.relevance,
                    row.note,
                ]
            )

    print(f"Saved annotation pool: {out_path}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
