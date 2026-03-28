from __future__ import annotations

"""Single entrypoint for evaluation of the refactored travel_ir_system.

Canonical inputs:
- data/evaluation/test_queries.json
- evaluation/annotation_pool.csv
- data/index/bm25_index.pkl
- data/index/vector_index.pkl

Outputs:
- evaluation/runs/run_bm25.tsv
- evaluation/runs/run_vector.tsv
- evaluation/runs/run_hybrid.tsv
- evaluation/qrels.tsv
- evaluation_metrics.json
- evaluation/reliability_report.md
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import search_engine as se  # noqa: E402


_ORIGINAL_LOAD_INDEX = se.load_index


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


se.load_index = lambda path: _load_index_cached(str(Path(path).resolve()))
se.encode_query = _encode_query_cached


@dataclass
class QueryDef:
    query_id: str
    query: str
    relevant_ids: set[str]
    relevance_grades: dict[str, int] = field(default_factory=dict)
    bucket_id: str = ""
    bucket_name: str = ""
    bucket_description: str = ""
    bucket_order: int = 0


@dataclass
class RunRow:
    query_id: str
    rank: int
    hotel_id: str
    score: float


def load_queries(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, it in enumerate(raw, start=1):
        q = str(it.get("query", "")).strip()
        if not q:
            continue
        qid = str(it.get("query_id", f"Q{i}")).strip() or f"Q{i}"
        bucket_id = str(it.get("bucket_id", "")).strip()
        bucket_name = str(it.get("bucket_name", "")).strip()
        bucket_description = str(it.get("bucket_description", "")).strip()
        try:
            bucket_order = int(it.get("bucket_order", 0) or 0)
        except Exception:
            bucket_order = 0
        out.append({
            "query_id": qid,
            "query": q,
            "bucket_id": bucket_id,
            "bucket_name": bucket_name,
            "bucket_description": bucket_description,
            "bucket_order": bucket_order,
        })
    return out


def load_manual_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("query_id", "")).strip()
            hid = str(row.get("hotel_id", "")).strip()
            rel_raw = str(row.get("relevance", "")).strip()
            if not qid or not hid:
                continue
            try:
                rel = int(rel_raw)
            except Exception:
                rel = 0
            qrels.setdefault(qid, {})[hid] = rel
    return qrels


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for h in retrieved[:k] if h in relevant) / float(k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for h in retrieved[:k] if h in relevant) / float(len(relevant))


def ap_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, h in enumerate(retrieved[:k], start=1):
        if h in relevant:
            hits += 1
            score += hits / float(i)
    return score / float(len(relevant))


def ndcg_at_k(retrieved: list[str], relevance_grades: dict[str, int], k: int) -> float:
    if not relevance_grades:
        return 0.0
    dcg = 0.0
    for i, h in enumerate(retrieved[:k], start=1):
        rel = float(relevance_grades.get(h, 0))
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    ideal_rels = sorted((float(v) for v in relevance_grades.values() if v > 0), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += rel / math.log2(i + 1)
    return dcg / idcg if idcg else 0.0


def run_model(query_defs: list[QueryDef], index_dir: Path, stopwords: Path, vw: float, bw: float, loc_boost: float = 1.8) -> dict[str, list[RunRow]]:
    out: dict[str, list[RunRow]] = {}
    for q in query_defs:
        results, _ = se.search_hybrid(
            query=q.query,
            index_dir=index_dir,
            stopwords_path=stopwords,
            top_k=10,
            vector_weight=vw,
            bm25_weight=bw,
            location_boost_factor=loc_boost,
        )
        rows: list[RunRow] = []
        for rank, r in enumerate(results[:10], start=1):
            hid = str(r.get("source_hotel_id", "")).strip()
            if not hid:
                continue
            rows.append(RunRow(q.query_id, rank, hid, float(r.get("hybrid_score", 0.0))))
        out[q.query_id] = rows
    return out


def write_run_file(path: Path, runs: dict[str, list[RunRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for qid in sorted(runs.keys(), key=lambda x: int(x[1:])):
        for row in runs[qid]:
            lines.append(f"{row.query_id}\t{row.rank}\t{row.hotel_id}\t{row.score:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qrels(path: Path, query_defs: list[QueryDef], run_maps: dict[str, dict[str, list[RunRow]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in query_defs:
        candidate_ids = set(q.relevance_grades)
        for runs in run_maps.values():
            candidate_ids.update(r.hotel_id for r in runs.get(q.query_id, []))
        for hid in sorted(candidate_ids):
            rel = int(q.relevance_grades.get(hid, 0))
            rows.append(f"{q.query_id}\t{hid}\t{rel}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def eval_runs(query_defs: list[QueryDef], runs: dict[str, list[RunRow]]) -> dict:
    p5s, p10s, r10s, maps, ndcgs = [], [], [], [], []
    strong_p10s, strong_r10s = [], []
    for q in query_defs:
        retrieved = [r.hotel_id for r in runs.get(q.query_id, [])]
        p5s.append(precision_at_k(retrieved, q.relevant_ids, 5))
        p10s.append(precision_at_k(retrieved, q.relevant_ids, 10))
        r10s.append(recall_at_k(retrieved, q.relevant_ids, 10))
        maps.append(ap_at_k(retrieved, q.relevant_ids, 10))
        ndcgs.append(ndcg_at_k(retrieved, q.relevance_grades, 10))
        strong_ids = {hid for hid, grade in q.relevance_grades.items() if grade >= 2}
        strong_p10s.append(precision_at_k(retrieved, strong_ids, 10))
        strong_r10s.append(recall_at_k(retrieved, strong_ids, 10))
    return {
        "precision@5": mean(p5s) if p5s else 0.0,
        "precision@10": mean(p10s) if p10s else 0.0,
        "recall@10": mean(r10s) if r10s else 0.0,
        "MAP": mean(maps) if maps else 0.0,
        "nDCG@10": mean(ndcgs) if ndcgs else 0.0,
        "strong_precision@10": mean(strong_p10s) if strong_p10s else 0.0,
        "strong_recall@10": mean(strong_r10s) if strong_r10s else 0.0,
    }


def group_queries_by_bucket(query_defs: list[QueryDef]) -> dict[str, list[QueryDef]]:
    grouped: dict[str, list[QueryDef]] = {}
    for q in query_defs:
        bucket_id = q.bucket_id or "__all__"
        grouped.setdefault(bucket_id, []).append(q)
    return grouped


def eval_runs_by_bucket(query_defs: list[QueryDef], runs: dict[str, list[RunRow]]) -> dict[str, dict]:
    grouped = group_queries_by_bucket(query_defs)
    bucket_meta: dict[str, dict] = {}
    for q in query_defs:
        bucket_id = q.bucket_id or "__all__"
        bucket_meta.setdefault(bucket_id, {
            "bucket_id": q.bucket_id or "__all__",
            "bucket_name": q.bucket_name or (q.bucket_id or "all"),
            "bucket_description": q.bucket_description or "",
            "bucket_order": q.bucket_order,
        })

    out: dict[str, dict] = {}
    for bucket_id, items in grouped.items():
        out[bucket_id] = {
            "meta": bucket_meta[bucket_id],
            "count": len(items),
            "metrics": eval_runs(items, runs),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate travel_ir_system retrieval models")
    default_queries = PROJECT_ROOT / "data" / "evaluation" / "test_queries_200_bucketed.json"
    if not default_queries.exists():
        default_queries = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
    default_pool = PROJECT_ROOT / "evaluation" / "annotation_pool_bucketed.csv"
    if not default_pool.exists():
        default_pool = PROJECT_ROOT / "evaluation" / "annotation_pool.csv"
    parser.add_argument("--queries", default=str(default_queries), help="Path to evaluation queries JSON")
    parser.add_argument("--annotation-pool", default=str(default_pool), help="Path to annotation pool CSV")
    parser.add_argument("--index-dir", default=str(PROJECT_ROOT / "data" / "index"), help="Path to index directory")
    parser.add_argument("--stopwords", default=str(PROJECT_ROOT / "config" / "stopwords.txt"), help="Path to stopwords file")
    parser.add_argument("--report-path", default=str(PROJECT_ROOT / "evaluation" / "reliability_report.md"), help="Path to reliability report")
    parser.add_argument("--metrics-path", default=str(PROJECT_ROOT / "evaluation_metrics.json"), help="Path to metrics json")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "evaluation" / "runs"), help="Directory for run files")
    parser.add_argument("--qrels-path", default=str(PROJECT_ROOT / "evaluation" / "qrels.tsv"), help="Output qrels path")
    args = parser.parse_args()

    queries_path = Path(args.queries).resolve()
    pool_path = Path(args.annotation_pool).resolve()
    index_dir = Path(args.index_dir).resolve()
    stopwords = Path(args.stopwords).resolve()
    report_path = Path(args.report_path).resolve()
    metrics_path = Path(args.metrics_path).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    qrels_path = Path(args.qrels_path).resolve()

    q_list = load_queries(queries_path)
    qrels = load_manual_qrels(pool_path)
    query_defs = []
    for it in q_list:
        grades = qrels.get(it["query_id"], {})
        relevant_ids = {hid for hid, rel in grades.items() if rel > 0}
        query_defs.append(
            QueryDef(
                it["query_id"],
                it["query"],
                relevant_ids,
                grades,
                it.get("bucket_id", ""),
                it.get("bucket_name", ""),
                it.get("bucket_description", ""),
                int(it.get("bucket_order", 0) or 0),
            )
        )

    runs_bm25 = run_model(query_defs, index_dir, stopwords, vw=0.0, bw=1.0, loc_boost=1.0)
    runs_vector = run_model(query_defs, index_dir, stopwords, vw=1.0, bw=0.0, loc_boost=1.0)
    runs_hybrid = run_model(query_defs, index_dir, stopwords, vw=0.6, bw=0.4, loc_boost=1.8)

    write_run_file(runs_dir / "run_bm25.tsv", runs_bm25)
    write_run_file(runs_dir / "run_vector.tsv", runs_vector)
    write_run_file(runs_dir / "run_hybrid.tsv", runs_hybrid)
    write_qrels(qrels_path, query_defs, {"bm25": runs_bm25, "vector": runs_vector, "hybrid": runs_hybrid})

    bm25_m = eval_runs(query_defs, runs_bm25)
    vector_m = eval_runs(query_defs, runs_vector)
    hybrid_m = eval_runs(query_defs, runs_hybrid)

    bucket_groups = group_queries_by_bucket(query_defs)
    bucket_metrics: dict[str, dict] = {}
    for bucket_id, items in bucket_groups.items():
        meta = {
            "bucket_id": items[0].bucket_id or "__all__",
            "bucket_name": items[0].bucket_name or (items[0].bucket_id or "all"),
            "bucket_description": items[0].bucket_description or "",
            "bucket_order": items[0].bucket_order,
        }
        bucket_metrics[bucket_id] = {
            "meta": meta,
            "count": len(items),
            "bm25": eval_runs(items, runs_bm25),
            "vector": eval_runs(items, runs_vector),
            "hybrid": eval_runs(items, runs_hybrid),
        }
        winners: dict[str, str] = {}
        for metric_name in ("precision@5", "precision@10", "recall@10", "MAP", "nDCG@10", "strong_precision@10", "strong_recall@10"):
            scored = {
                "bm25": bucket_metrics[bucket_id]["bm25"][metric_name],
                "vector": bucket_metrics[bucket_id]["vector"][metric_name],
                "hybrid": bucket_metrics[bucket_id]["hybrid"][metric_name],
            }
            winners[metric_name] = max(scored.items(), key=lambda kv: kv[1])[0]
        bucket_metrics[bucket_id]["winners"] = winners

    payload = {
        "annotation_source": str(pool_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "query_source": str(queries_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "models": {
            "bm25": bm25_m,
            "vector": vector_m,
            "hybrid": hybrid_m,
        },
        "bucket_metrics": bucket_metrics,
        "threshold_check": {
            "precision@10>0.6": hybrid_m["precision@10"] > 0.6,
            "nDCG@10>0.7": hybrid_m["nDCG@10"] > 0.7,
            "MAP>0.5": hybrid_m["MAP"] > 0.5,
        },
        "conclusion": "deploy_ready" if (
            hybrid_m["precision@10"] > 0.6 and hybrid_m["nDCG@10"] > 0.7 and hybrid_m["MAP"] > 0.5
        ) else "not_deploy_ready",
    }
    payload.update({
        "precision@5": hybrid_m["precision@5"],
        "precision@10": hybrid_m["precision@10"],
        "recall@10": hybrid_m["recall@10"],
        "MAP": hybrid_m["MAP"],
        "nDCG@10": hybrid_m["nDCG@10"],
        "strong_precision@10": hybrid_m["strong_precision@10"],
        "strong_recall@10": hybrid_m["strong_recall@10"],
    })

    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# IR Reliability Report",
        "",
        "## Canonical inputs",
        f"- Query set: `{queries_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Annotation pool: `{pool_path.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Overall results",
        f"- **bm25**: P@5={bm25_m['precision@5']:.3f}, P@10={bm25_m['precision@10']:.3f}, R@10={bm25_m['recall@10']:.3f}, MAP={bm25_m['MAP']:.3f}, nDCG@10={bm25_m['nDCG@10']:.3f}, strongP@10={bm25_m['strong_precision@10']:.3f}, strongR@10={bm25_m['strong_recall@10']:.3f}",
        f"- **vector**: P@5={vector_m['precision@5']:.3f}, P@10={vector_m['precision@10']:.3f}, R@10={vector_m['recall@10']:.3f}, MAP={vector_m['MAP']:.3f}, nDCG@10={vector_m['nDCG@10']:.3f}, strongP@10={vector_m['strong_precision@10']:.3f}, strongR@10={vector_m['strong_recall@10']:.3f}",
        f"- **hybrid**: P@5={hybrid_m['precision@5']:.3f}, P@10={hybrid_m['precision@10']:.3f}, R@10={hybrid_m['recall@10']:.3f}, MAP={hybrid_m['MAP']:.3f}, nDCG@10={hybrid_m['nDCG@10']:.3f}, strongP@10={hybrid_m['strong_precision@10']:.3f}, strongR@10={hybrid_m['strong_recall@10']:.3f}",
        "",
        "## Bucket results",
    ]
    for bucket_id, info in bucket_metrics.items():
        meta = info["meta"]
        report_lines.extend([
            f"### {bucket_id} — {meta['bucket_name']} (n={info['count']})",
            f"- Description: {meta['bucket_description']}",
            f"- bm25: P@10={info['bm25']['precision@10']:.3f}, MAP={info['bm25']['MAP']:.3f}, nDCG@10={info['bm25']['nDCG@10']:.3f}, strongP@10={info['bm25']['strong_precision@10']:.3f}, strongR@10={info['bm25']['strong_recall@10']:.3f}",
            f"- vector: P@10={info['vector']['precision@10']:.3f}, MAP={info['vector']['MAP']:.3f}, nDCG@10={info['vector']['nDCG@10']:.3f}, strongP@10={info['vector']['strong_precision@10']:.3f}, strongR@10={info['vector']['strong_recall@10']:.3f}",
            f"- hybrid: P@10={info['hybrid']['precision@10']:.3f}, MAP={info['hybrid']['MAP']:.3f}, nDCG@10={info['hybrid']['nDCG@10']:.3f}, strongP@10={info['hybrid']['strong_precision@10']:.3f}, strongR@10={info['hybrid']['strong_recall@10']:.3f}",
            f"- winners: P@10={info['winners']['precision@10']}, MAP={info['winners']['MAP']}, nDCG@10={info['winners']['nDCG@10']}, strongP@10={info['winners']['strong_precision@10']}, strongR@10={info['winners']['strong_recall@10']}",
            "",
        ])
    report_lines.extend([
        "## Verdict",
        f"- Precision@10 > 0.6: {'Đạt' if payload['threshold_check']['precision@10>0.6'] else 'Không đạt'} ({hybrid_m['precision@10']:.3f})",
        f"- nDCG@10 > 0.7: {'Đạt' if payload['threshold_check']['nDCG@10>0.7'] else 'Không đạt'} ({hybrid_m['nDCG@10']:.3f})",
        f"- MAP > 0.5: {'Đạt' if payload['threshold_check']['MAP>0.5'] else 'Không đạt'} ({hybrid_m['MAP']:.3f})",
        f"- Conclusion: **{'Có thể deploy có kiểm soát' if payload['conclusion']=='deploy_ready' else 'Chưa nên deploy production'}**",
        "",
        "## Output files",
        f"- `{qrels_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(runs_dir / 'run_bm25.tsv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(runs_dir / 'run_vector.tsv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(runs_dir / 'run_hybrid.tsv').relative_to(PROJECT_ROOT).as_posix()}`",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved qrels to {qrels_path}")
    print(f"Saved runs to {runs_dir}")


if __name__ == "__main__":
    main()
