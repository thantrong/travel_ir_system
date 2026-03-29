from __future__ import annotations

"""Build merged top-10 per query from the existing annotation pool.

This avoids recomputing retrieval and simply re-ranks the pool rows by:
- bm25_score
- vector_score
- hybrid_score

Then it merges duplicates by hotel_id and emits one row per (query_id, hotel_id)
with binary labels:
- 0 = not relevant
- 1 = relevant (original relevance > 0)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "evaluation" / "annotation_pool_bucketed.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "merged_top10_labeled.csv"))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("query_id", "")).strip()
            hid = str(row.get("hotel_id", "")).strip()
            if not qid or not hid:
                continue
            # Normalize graded labels to binary: 0 / 1 only
            # Any positive relevance (including 2) becomes 1.
            row["relevance"] = 1 if to_int(row.get("relevance", 0)) > 0 else 0
            row["bm25_score"] = to_float(row.get("bm25_score", 0))
            row["vector_score"] = to_float(row.get("vector_score", 0))
            row["hybrid_score"] = to_float(row.get("hybrid_score", 0))
            by_query[qid].append(row)

    out_rows: list[dict[str, Any]] = []
    for qid, rows in by_query.items():
        # top 10 per model
        top_bm25 = sorted(rows, key=lambda r: r["bm25_score"], reverse=True)[:10]
        top_vec = sorted(rows, key=lambda r: r["vector_score"], reverse=True)[:10]
        top_hyb = sorted(rows, key=lambda r: r["hybrid_score"], reverse=True)[:10]

        merged: dict[str, dict[str, Any]] = {}
        for model_name, model_rows in (("bm25", top_bm25), ("vector", top_vec), ("hybrid", top_hyb)):
            for rank, row in enumerate(model_rows, start=1):
                hid = row["hotel_id"]
                out = merged.setdefault(hid, {
                    "query_id": qid,
                    "query": row.get("query", ""),
                    "hotel_id": hid,
                    "hotel_name": row.get("hotel_name", ""),
                    "location": row.get("location", ""),
                    "label": 1 if to_int(row.get("relevance", 0)) > 0 else 0,
                    "bm25_rank": "",
                    "bm25_score": "",
                    "vector_rank": "",
                    "vector_score": "",
                    "hybrid_rank": "",
                    "hybrid_score": "",
                })
                out[f"{model_name}_rank"] = rank
                out[f"{model_name}_score"] = row[f"{model_name}_score"]
                # binary label should be 1 if any source row says relevant
                if int(row.get("relevance", 0)) > 0:
                    out["label"] = 1

        # sort merged output by hybrid rank, then bm25 rank, then vector rank
        ordered = sorted(
            merged.values(),
            key=lambda r: (
                r["hybrid_rank"] if r["hybrid_rank"] != "" else 999,
                r["bm25_rank"] if r["bm25_rank"] != "" else 999,
                r["vector_rank"] if r["vector_rank"] != "" else 999,
            ),
        )
        out_rows.extend(ordered)

    fieldnames = [
        "query_id", "query", "hotel_id", "hotel_name", "location", "label",
        "bm25_rank", "bm25_score", "vector_rank", "vector_score", "hybrid_rank", "hybrid_score",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Safety check: output labels must be binary only.
    bad = [r for r in out_rows if str(r.get("label", "")) not in {"0", "1", 0, 1}]
    if bad:
        raise ValueError(f"Found non-binary labels in output: {bad[:3]}")

    print(f"Wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
