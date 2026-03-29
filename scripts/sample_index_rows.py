from __future__ import annotations

"""Print sample rows from bm25_index.pkl and vector_index.pkl.

This version dumps all fields already stored in each indexed document.

Usage:
    python scripts/sample_index_rows.py
    python scripts/sample_index_rows.py --n 3
    python scripts/sample_index_rows.py --output evaluation/index_samples.csv
"""

import argparse
import csv
import json
import pickle
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def pick_samples(items: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    if not items:
        return []
    if len(items) <= n:
        return items
    rnd = random.Random(seed)
    idxs = sorted(rnd.sample(range(len(items)), n))
    return [items[i] for i in idxs]


def stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def summarize_doc(doc: dict[str, Any]) -> dict[str, str]:
    # Keep every field that exists in the indexed document.
    return {k: stringify(v) for k, v in doc.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of sample rows per index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", default="", help="Optional CSV output path")
    args = parser.parse_args()

    index_dir = PROJECT_ROOT / "data" / "index"
    bm25_payload = load_pickle(index_dir / "bm25_index.pkl")
    vector_payload = load_pickle(index_dir / "vector_index.pkl")

    bm25_samples = pick_samples(bm25_payload.get("documents", []), args.n, args.seed)
    vector_samples = pick_samples(vector_payload.get("documents", []), args.n, args.seed)

    rows: list[dict[str, str]] = []
    for source_name, docs in (("bm25", bm25_samples), ("vector", vector_samples)):
        for i, doc in enumerate(docs, start=1):
            row = summarize_doc(doc)
            row["source"] = source_name
            row["sample_rank"] = str(i)
            rows.append(row)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["source", "sample_rank"]
        extra_keys = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames and k not in extra_keys:
                    extra_keys.append(k)
        fieldnames.extend(extra_keys)

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        for row in rows:
            print("-" * 100)
            print(f"[{row['source']}] sample #{row['sample_rank']}")
            for k, v in row.items():
                if k in {"source", "sample_rank"}:
                    continue
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
