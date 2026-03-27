from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(r"C:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
sys.path.insert(0, str(root))

from retrieval.search_engine import search_hybrid


INPUT = root / "evaluation" / "annotation_pool.csv"
OUTPUT = root / "evaluation" / "annotation_pool.csv"
INDEX_DIR = root / "data" / "index"
STOPWORDS = root / "config" / "stopwords.txt"


def main():
    rows = []
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    query_cache: dict[str, list[dict]] = {}
    # collect queries
    queries = []
    seen = set()
    for row in rows:
        qid = row["query_id"].strip()
        q = row["query"].strip()
        if qid not in seen:
            seen.add(qid)
            queries.append((qid, q))

    for qid, q in queries:
        results, _ = search_hybrid(
            query=q,
            index_dir=INDEX_DIR,
            stopwords_path=STOPWORDS,
            top_k=10,
            vector_weight=0.6,
            bm25_weight=0.4,
            location_boost_factor=1.8,
        )
        query_cache[qid] = results

    # rebuild rows with evidence reviews from the matching hotel in current ranking
    out_rows = []
    for row in rows:
        qid = row["query_id"].strip()
        hid = row["hotel_id"].strip()
        qresults = query_cache.get(qid, [])
        matched = None
        for r in qresults:
            if str(r.get("source_hotel_id", "")).strip() == hid:
                matched = r
                break
        evidence = matched.get("top_reviews", []) if matched else []
        evidence = [str(x).strip() for x in evidence if str(x).strip()]
        out = dict(row)
        out["evidence_review_1"] = evidence[0] if len(evidence) > 0 else ""
        out["evidence_review_2"] = evidence[1] if len(evidence) > 1 else ""
        out["evidence_review_3"] = evidence[2] if len(evidence) > 2 else ""
        out["evidence_reviews"] = " || ".join(evidence)
        out_rows.append(out)

    headers = list(out_rows[0].keys()) if out_rows else []
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Enriched annotation pool saved: {OUTPUT} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
