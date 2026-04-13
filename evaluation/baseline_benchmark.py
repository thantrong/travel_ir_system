from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.search_engine import search_hybrid


def load_queries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for idx, item in enumerate(payload, start=1):
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        out.append({"query_id": str(item.get("query_id", f"Q{idx}")), "query": q})
    return out


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("query_id", "")).strip()
            hid = str(row.get("hotel_id", "")).strip()
            rel = int(row.get("relevant", "0") or 0)
            if qid and hid:
                out[qid][hid] = rel
    return out


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for idx, hid in enumerate(retrieved[:k], start=1):
        if hid in relevant:
            return 1.0 / idx
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for hid in retrieved[:k] if hid in relevant)
    return hit / len(relevant)


def ndcg_at_k(retrieved: list[str], grades: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, hid in enumerate(retrieved[:k], start=1):
        rel = float(grades.get(hid, 0))
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    ideal = sorted([float(v) for v in grades.values() if v > 0], reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        idcg += rel / math.log2(i + 1)
    return dcg / idcg if idcg else 0.0


def run_benchmark(queries: list[dict], qrels: dict[str, dict[str, int]], top_k: int) -> dict:
    process = psutil.Process()
    latencies_ms: list[float] = []
    rss_samples_mb: list[float] = []
    mrr_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    no_result = 0

    index_dir = PROJECT_ROOT / "data" / "index"
    stopwords = PROJECT_ROOT / "config" / "stopwords.txt"

    for q in queries:
        start = time.perf_counter()
        results, _ = search_hybrid(
            query=q["query"],
            index_dir=index_dir,
            stopwords_path=stopwords,
            top_k=top_k,
        )
        latency = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency)
        rss_samples_mb.append(process.memory_info().rss / (1024 * 1024))

        retrieved = [str(r.get("source_hotel_id", "")).strip() for r in results if str(r.get("source_hotel_id", "")).strip()]
        if not retrieved:
            no_result += 1

        grades = qrels.get(q["query_id"], {})
        relevant = {hid for hid, rel in grades.items() if rel > 0}
        mrr_scores.append(mrr_at_k(retrieved, relevant, top_k))
        recall_scores.append(recall_at_k(retrieved, relevant, top_k))
        ndcg_scores.append(ndcg_at_k(retrieved, grades, top_k))

    lat_sorted = sorted(latencies_ms)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else 0.0
    return {
        "query_count": len(queries),
        "top_k": top_k,
        "MRR": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "Recall@20": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "p95_latency_ms": p95,
        "avg_latency_ms": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "max_rss_mb": max(rss_samples_mb) if rss_samples_mb else 0.0,
        "avg_rss_mb": (sum(rss_samples_mb) / len(rss_samples_mb)) if rss_samples_mb else 0.0,
        "no_result_rate": (no_result / len(queries)) if queries else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline benchmark for IR runtime and quality.")
    parser.add_argument("--queries", default=str(PROJECT_ROOT / "data" / "evaluation" / "test_queries_200_bucketed.json"))
    parser.add_argument("--qrels", default=str(PROJECT_ROOT / "data" / "evaluation" / "pool_results_labeled.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "baseline_metrics.json"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    if args.limit > 0:
        queries = queries[: args.limit]
    qrels = load_qrels(Path(args.qrels))
    payload = run_benchmark(queries, qrels, top_k=args.top_k)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

