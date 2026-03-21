"""
Run full IR reliability evaluation for BM25/Vector/Hybrid.

Outputs:
- evaluation/qrels.tsv
- evaluation/runs/run_bm25.tsv
- evaluation/runs/run_vector.tsv
- evaluation/runs/run_hybrid.tsv
- evaluation_metrics.json
- evaluation/reliability_report.md
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import search_engine as se  # noqa: E402
from retrieval.query_understanding import LOCATION_KEYWORDS  # noqa: E402

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
    query_text: str
    relevant_ids: set[str]


@dataclass
class RunItem:
    query_id: str
    rank: int
    hotel_id: str
    score: float
    location: str
    location_matched: bool
    descriptor_matched: bool


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    ret = retrieved[:k]
    hits = sum(1 for hid in ret if hid in relevant)
    return hits / float(k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ret = retrieved[:k]
    hits = sum(1 for hid in ret if hid in relevant)
    return hits / float(len(relevant))


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, hid in enumerate(retrieved[:k], start=1):
        if hid in relevant:
            hits += 1
            total += hits / float(i)
    return total / float(len(relevant))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(items: list[str]) -> float:
        out = 0.0
        for i, hid in enumerate(items, start=1):
            rel = 1.0 if hid in relevant else 0.0
            out += rel / math.log2(i + 1)
        return out

    if not relevant:
        return 0.0
    got = dcg(retrieved[:k])
    ideal_count = min(k, len(relevant))
    ideal_list = ["__REL__"] * ideal_count
    ideal = 0.0
    for i, _ in enumerate(ideal_list, start=1):
        ideal += 1.0 / math.log2(i + 1)
    if ideal == 0:
        return 0.0
    return got / ideal


def load_queries(queries_path: Path, limit: int = 50) -> list[QueryDef]:
    raw = json.loads(queries_path.read_text(encoding="utf-8"))
    selected = raw[:limit]
    out: list[QueryDef] = []
    for i, item in enumerate(selected, start=1):
        qid = f"Q{i}"
        out.append(
            QueryDef(
                query_id=qid,
                query_text=str(item["query"]).strip(),
                relevant_ids={str(x).strip() for x in item.get("relevant_hotel_ids", []) if str(x).strip()},
            )
        )
    return out


def run_model(queries: list[QueryDef], model_name: str, cfg: dict, index_dir: Path, stopwords_path: Path) -> dict[str, list[RunItem]]:
    runs: dict[str, list[RunItem]] = {}
    for idx, q in enumerate(queries, start=1):
        print(f"[{model_name}] {idx}/{len(queries)} - {q.query_id}: {q.query_text}")
        results, _ = se.search_hybrid(
            query=q.query_text,
            index_dir=index_dir,
            stopwords_path=stopwords_path,
            top_k=10,
            vector_weight=cfg["vector_weight"],
            bm25_weight=cfg["bm25_weight"],
            location_boost_factor=cfg["location_boost_factor"],
        )
        rows: list[RunItem] = []
        for rank, r in enumerate(results[:10], start=1):
            rows.append(
                RunItem(
                    query_id=q.query_id,
                    rank=rank,
                    hotel_id=str(r.get("source_hotel_id", "")),
                    score=float(r.get("hybrid_score", 0.0)),
                    location=str(r.get("location", "")),
                    location_matched=bool(r.get("location_matched", False)),
                    descriptor_matched=bool(r.get("descriptor_matched", False)),
                )
            )
        runs[q.query_id] = rows
    return runs


def write_run_file(path: Path, runs: dict[str, list[RunItem]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for qid in sorted(runs.keys(), key=lambda x: int(x[1:])):
        for row in runs[qid]:
            lines.append(f"{row.query_id}\t{row.rank}\t{row.hotel_id}\t{row.score:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_qrels(queries: list[QueryDef], run_maps: dict[str, dict[str, list[RunItem]]], out_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for q in queries:
        candidate_ids = set(q.relevant_ids)
        for _, runs in run_maps.items():
            candidate_ids.update(r.hotel_id for r in runs.get(q.query_id, []))
        qrels[q.query_id] = {hid: (1 if hid in q.relevant_ids else 0) for hid in sorted(candidate_ids)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for qid in sorted(qrels.keys(), key=lambda x: int(x[1:])):
        for hid, rel in qrels[qid].items():
            lines.append(f"{qid}\t{hid}\t{rel}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return qrels


def evaluate_runs(queries: list[QueryDef], runs: dict[str, list[RunItem]]) -> tuple[dict, dict[str, dict]]:
    per_query: dict[str, dict] = {}
    p5_vals = []
    p10_vals = []
    r10_vals = []
    ap_vals = []
    ndcg_vals = []
    for q in queries:
        retrieved = [x.hotel_id for x in runs.get(q.query_id, [])]
        p5 = precision_at_k(retrieved, q.relevant_ids, 5)
        p10 = precision_at_k(retrieved, q.relevant_ids, 10)
        r10 = recall_at_k(retrieved, q.relevant_ids, 10)
        ap = average_precision_at_k(retrieved, q.relevant_ids, 10)
        ndcg10 = ndcg_at_k(retrieved, q.relevant_ids, 10)
        p5_vals.append(p5)
        p10_vals.append(p10)
        r10_vals.append(r10)
        ap_vals.append(ap)
        ndcg_vals.append(ndcg10)
        per_query[q.query_id] = {
            "query": q.query_text,
            "precision@5": p5,
            "precision@10": p10,
            "recall@10": r10,
            "AP@10": ap,
            "nDCG@10": ndcg10,
        }

    overall = {
        "precision@5": mean(p5_vals) if p5_vals else 0.0,
        "precision@10": mean(p10_vals) if p10_vals else 0.0,
        "recall@10": mean(r10_vals) if r10_vals else 0.0,
        "MAP": mean(ap_vals) if ap_vals else 0.0,
        "nDCG@10": mean(ndcg_vals) if ndcg_vals else 0.0,
    }
    return overall, per_query


def _query_has_location_text(query: str) -> bool:
    ql = query.lower()
    for _, kws in LOCATION_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in ql:
                return True
    return False


def detect_failure_reasons(
    queries: list[QueryDef],
    hybrid_runs: dict[str, list[RunItem]],
    per_query_hybrid: dict[str, dict],
    per_query_bm25: dict[str, dict],
    per_query_vector: dict[str, dict],
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for q in queries:
        qid = q.query_id
        query_text = q.query_text
        p10_h = per_query_hybrid[qid]["precision@10"]
        p10_b = per_query_bm25[qid]["precision@10"]
        p10_v = per_query_vector[qid]["precision@10"]
        rows = hybrid_runs.get(qid, [])
        location_ratio = (sum(1 for r in rows if r.location_matched) / len(rows)) if rows else 0.0
        descriptor_ratio = (sum(1 for r in rows if r.descriptor_matched) / len(rows)) if rows else 0.0

        rs: list[str] = []
        if p10_h + 0.1 < max(p10_b, p10_v):
            rs.append("lỗi ranking: hybrid kém hơn model thành phần, dấu hiệu weight/boost chưa cân bằng")
        if _query_has_location_text(query_text) and location_ratio < 0.5:
            rs.append("lỗi location handling: nhiều kết quả top-10 không khớp location")
        if descriptor_ratio < 0.4:
            rs.append("lỗi semantic matching/query understanding: descriptor support thấp")
        if p10_v > p10_b + 0.2 and p10_h <= p10_b:
            rs.append("lỗi semantic mismatch: vector bắt đúng ý hơn BM25 nhưng hybrid chưa tận dụng")
        if not rs:
            rs.append("không phát hiện lỗi nổi bật ở truy vấn này")
        reasons[qid] = rs
    return reasons


def generate_report(
    out_path: Path,
    architecture_findings: list[str],
    overall_metrics: dict,
    queries: list[QueryDef],
    per_query_hybrid: dict[str, dict],
    failure_reasons: dict[str, list[str]],
) -> None:
    ranked = sorted(
        ((qid, m["precision@10"], m["nDCG@10"]) for qid, m in per_query_hybrid.items()),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    best = ranked[:8]
    worst = list(reversed(ranked[-8:]))
    query_text_map = {q.query_id: q.query_text for q in queries}

    lines = []
    lines.append("# IR Reliability Report")
    lines.append("")
    lines.append("## 1. Kiến trúc và rủi ro")
    for f in architecture_findings:
        lines.append(f"- {f}")
    lines.append("")
    lines.append("## 2. Kết quả tổng quan theo model")
    for model, vals in overall_metrics.items():
        lines.append(
            f"- **{model}**: P@5={vals['precision@5']:.3f}, P@10={vals['precision@10']:.3f}, "
            f"R@10={vals['recall@10']:.3f}, MAP={vals['MAP']:.3f}, nDCG@10={vals['nDCG@10']:.3f}"
        )
    lines.append("")
    lines.append("## 3. Query tốt nhất (Hybrid)")
    for qid, p10, nd in best:
        lines.append(f"- {qid} | P@10={p10:.2f}, nDCG@10={nd:.2f} | `{query_text_map[qid]}`")
    lines.append("")
    lines.append("## 4. Query kém nhất (Hybrid)")
    for qid, p10, nd in worst:
        reasons = "; ".join(failure_reasons[qid][:2])
        lines.append(f"- {qid} | P@10={p10:.2f}, nDCG@10={nd:.2f} | `{query_text_map[qid]}` | nguyên nhân: {reasons}")
    lines.append("")
    lines.append("## 5. Đánh giá ngưỡng tin cậy (Hybrid)")
    h = overall_metrics["hybrid"]
    p10_ok = h["precision@10"] > 0.6
    ndcg_ok = h["nDCG@10"] > 0.7
    map_ok = h["MAP"] > 0.5
    lines.append(f"- Precision@10 > 0.6: {'Đạt' if p10_ok else 'Không đạt'} ({h['precision@10']:.3f})")
    lines.append(f"- nDCG@10 > 0.7: {'Đạt' if ndcg_ok else 'Không đạt'} ({h['nDCG@10']:.3f})")
    lines.append(f"- MAP > 0.5: {'Đạt' if map_ok else 'Không đạt'} ({h['MAP']:.3f})")
    verdict = "Có thể deploy có kiểm soát" if (p10_ok and ndcg_ok and map_ok) else "Chưa nên deploy production"
    lines.append(f"- Kết luận: **{verdict}**")
    lines.append("")
    lines.append("## 6. Đề xuất cải thiện")
    lines.append("- Tuning lại `vector_weight`/`bm25_weight` theo grid search, không giữ cứng 0.6/0.4.")
    lines.append("- Giảm biên độ rule-based boost/penalty (POI/type), ưu tiên calibration theo validation set.")
    lines.append("- Cải thiện query understanding cho truy vấn dài/mixed-intent bằng tách intent đa nhãn.")
    lines.append("- Điều chỉnh aggregation để giảm review dilution (ví dụ weighted-topk thay vì sum cứng top 5).")
    lines.append("- Áp dụng spam/noise handling ở ranking stage để giảm ảnh hưởng review cực đoan.")
    lines.append("- Bổ sung semantic re-ranker nhẹ cho top candidates để giảm keyword bias.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    queries_path = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
    index_dir = PROJECT_ROOT / "data" / "index"
    stopwords_path = PROJECT_ROOT / "config" / "stopwords.txt"

    queries = load_queries(queries_path, limit=50)
    print(f"Loaded {len(queries)} queries.")

    model_cfgs = {
        "bm25": {"vector_weight": 0.0, "bm25_weight": 1.0, "location_boost_factor": 1.8},
        "vector": {"vector_weight": 1.0, "bm25_weight": 0.0, "location_boost_factor": 1.8},
        "hybrid": {"vector_weight": 0.6, "bm25_weight": 0.4, "location_boost_factor": 1.8},
    }

    all_runs: dict[str, dict[str, list[RunItem]]] = {}
    for model_name, cfg in model_cfgs.items():
        all_runs[model_name] = run_model(queries, model_name, cfg, index_dir, stopwords_path)
        write_run_file(PROJECT_ROOT / "evaluation" / "runs" / f"run_{model_name}.tsv", all_runs[model_name])

    qrels = build_qrels(queries, all_runs, PROJECT_ROOT / "evaluation" / "qrels.tsv")
    print(f"QRELS size: {sum(len(v) for v in qrels.values())} pairs.")

    overall_metrics: dict[str, dict] = {}
    per_query_metrics: dict[str, dict[str, dict]] = {}
    for model_name, runs in all_runs.items():
        overall, per_query = evaluate_runs(queries, runs)
        overall_metrics[model_name] = overall
        per_query_metrics[model_name] = per_query

    failure_reasons = detect_failure_reasons(
        queries=queries,
        hybrid_runs=all_runs["hybrid"],
        per_query_hybrid=per_query_metrics["hybrid"],
        per_query_bm25=per_query_metrics["bm25"],
        per_query_vector=per_query_metrics["vector"],
    )

    architecture_findings = [
        "Document length bias: BM25 field weighting nhân bản token (`hotel_name*3`, `location*2`) làm lệch chuẩn hóa theo độ dài.",
        "Review dilution: hotel score lấy tổng top review có thể ưu tiên khách sạn nhiều review trung bình thay vì ít review nhưng rất đúng intent.",
        "Keyword bias: nhiều luật từ điển + substring trong ranking khiến lexical signals lấn semantic ở truy vấn tự nhiên dài.",
        "Semantic mismatch: descriptor filter cứng có thể loại oan kết quả ngữ nghĩa đúng nhưng khác biểu đạt.",
        "Ranking inconsistency: hệ số thưởng/phạt lớn ở POI/type (`x2`, `x0.1`, `x0.001`, `x1.5`, `x0.2`) dễ đảo hạng thiếu ổn định.",
    ]

    report_path = PROJECT_ROOT / "evaluation" / "reliability_report.md"
    generate_report(
        out_path=report_path,
        architecture_findings=architecture_findings,
        overall_metrics=overall_metrics,
        queries=queries,
        per_query_hybrid=per_query_metrics["hybrid"],
        failure_reasons=failure_reasons,
    )

    hybrid = overall_metrics["hybrid"]
    export_json = {
        "precision@5": hybrid["precision@5"],
        "precision@10": hybrid["precision@10"],
        "recall@10": hybrid["recall@10"],
        "MAP": hybrid["MAP"],
        "nDCG@10": hybrid["nDCG@10"],
        "models": overall_metrics,
        "per_query": per_query_metrics,
        "failure_reasons": failure_reasons,
        "threshold_check": {
            "precision@10>0.6": hybrid["precision@10"] > 0.6,
            "nDCG@10>0.7": hybrid["nDCG@10"] > 0.7,
            "MAP>0.5": hybrid["MAP"] > 0.5,
        },
        "conclusion": "deploy_ready" if (
            hybrid["precision@10"] > 0.6 and hybrid["nDCG@10"] > 0.7 and hybrid["MAP"] > 0.5
        ) else "not_deploy_ready",
        "artifacts": {
            "qrels": "evaluation/qrels.tsv",
            "run_bm25": "evaluation/runs/run_bm25.tsv",
            "run_vector": "evaluation/runs/run_vector.tsv",
            "run_hybrid": "evaluation/runs/run_hybrid.tsv",
            "report": "evaluation/reliability_report.md",
        },
    }

    (PROJECT_ROOT / "evaluation_metrics.json").write_text(
        json.dumps(export_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved evaluation artifacts and evaluation_metrics.json")


if __name__ == "__main__":
    main()
