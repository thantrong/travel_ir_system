from __future__ import annotations

import csv
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


def load_queries(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, it in enumerate(raw, start=1):
        q = str(it.get("query", "")).strip()
        if q:
            out.append((f"Q{i}", q))
    return out


def load_manual_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row.get("query_id", "").strip()
            hid = row.get("hotel_id", "").strip()
            rel = row.get("relevance", "").strip()
            if not qid or not hid:
                continue
            if qid not in qrels:
                qrels[qid] = set()
            if rel == "1":
                qrels[qid].add(hid)
    return qrels


def p_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for h in retrieved[:k] if h in relevant) / float(k)


def r_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
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


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, h in enumerate(retrieved[:k], start=1):
        rel = 1.0 if h in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_n = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def run_model(queries: list[QueryDef], index_dir: Path, stopwords: Path, vw: float, bw: float) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for q in queries:
        results, _ = se.search_hybrid(
            query=q.query,
            index_dir=index_dir,
            stopwords_path=stopwords,
            top_k=10,
            vector_weight=vw,
            bm25_weight=bw,
            location_boost_factor=1.8,
        )
        out[q.query_id] = [str(r.get("source_hotel_id", "")).strip() for r in results[:10] if str(r.get("source_hotel_id", "")).strip()]
    return out


def eval_model(queries: list[QueryDef], runs: dict[str, list[str]]) -> dict:
    p5s, p10s, r10s, maps, ndcgs = [], [], [], [], []
    for q in queries:
        ret = runs.get(q.query_id, [])
        p5s.append(p_at_k(ret, q.relevant_ids, 5))
        p10s.append(p_at_k(ret, q.relevant_ids, 10))
        r10s.append(r_at_k(ret, q.relevant_ids, 10))
        maps.append(ap_at_k(ret, q.relevant_ids, 10))
        ndcgs.append(ndcg_at_k(ret, q.relevant_ids, 10))
    return {
        "precision@5": mean(p5s) if p5s else 0.0,
        "precision@10": mean(p10s) if p10s else 0.0,
        "recall@10": mean(r10s) if r10s else 0.0,
        "MAP": mean(maps) if maps else 0.0,
        "nDCG@10": mean(ndcgs) if ndcgs else 0.0,
    }


def main() -> None:
    queries_path = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
    pool_path = PROJECT_ROOT / "evaluation" / "annotation_pool.tsv"
    index_dir = PROJECT_ROOT / "data" / "index"
    stopwords = PROJECT_ROOT / "config" / "stopwords.txt"
    report_path = PROJECT_ROOT / "evaluation" / "reliability_report.md"
    metrics_path = PROJECT_ROOT / "evaluation_metrics.json"

    q_list = load_queries(queries_path)
    qrels = load_manual_qrels(pool_path)
    queries = [QueryDef(query_id=qid, query=query, relevant_ids=qrels.get(qid, set())) for qid, query in q_list]

    bm25_runs = run_model(queries, index_dir, stopwords, vw=0.0, bw=1.0)
    vector_runs = run_model(queries, index_dir, stopwords, vw=1.0, bw=0.0)
    hybrid_runs = run_model(queries, index_dir, stopwords, vw=0.6, bw=0.4)

    bm25_m = eval_model(queries, bm25_runs)
    vector_m = eval_model(queries, vector_runs)
    hybrid_m = eval_model(queries, hybrid_runs)

    payload = {
        "precision@5": hybrid_m["precision@5"],
        "precision@10": hybrid_m["precision@10"],
        "recall@10": hybrid_m["recall@10"],
        "MAP": hybrid_m["MAP"],
        "nDCG@10": hybrid_m["nDCG@10"],
        "models": {"bm25": bm25_m, "vector": vector_m, "hybrid": hybrid_m},
        "annotation_source": "evaluation/annotation_pool.tsv",
        "threshold_check": {
            "precision@10>0.6": hybrid_m["precision@10"] > 0.6,
            "nDCG@10>0.7": hybrid_m["nDCG@10"] > 0.7,
            "MAP>0.5": hybrid_m["MAP"] > 0.5,
        },
        "conclusion": "deploy_ready" if (
            hybrid_m["precision@10"] > 0.6 and hybrid_m["nDCG@10"] > 0.7 and hybrid_m["MAP"] > 0.5
        ) else "not_deploy_ready",
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# IR Reliability Report",
        "",
        "## 1. Nguồn nhãn đánh giá",
        "- Báo cáo này dùng nhãn thủ công từ `evaluation/annotation_pool.tsv` (cột `relevance`).",
        "",
        "## 2. Kết quả tổng quan theo model",
        f"- **bm25**: P@5={bm25_m['precision@5']:.3f}, P@10={bm25_m['precision@10']:.3f}, R@10={bm25_m['recall@10']:.3f}, MAP={bm25_m['MAP']:.3f}, nDCG@10={bm25_m['nDCG@10']:.3f}",
        f"- **vector**: P@5={vector_m['precision@5']:.3f}, P@10={vector_m['precision@10']:.3f}, R@10={vector_m['recall@10']:.3f}, MAP={vector_m['MAP']:.3f}, nDCG@10={vector_m['nDCG@10']:.3f}",
        f"- **hybrid**: P@5={hybrid_m['precision@5']:.3f}, P@10={hybrid_m['precision@10']:.3f}, R@10={hybrid_m['recall@10']:.3f}, MAP={hybrid_m['MAP']:.3f}, nDCG@10={hybrid_m['nDCG@10']:.3f}",
        "",
        "## 3. Đánh giá ngưỡng tin cậy (Hybrid)",
        f"- Precision@10 > 0.6: {'Đạt' if payload['threshold_check']['precision@10>0.6'] else 'Không đạt'} ({hybrid_m['precision@10']:.3f})",
        f"- nDCG@10 > 0.7: {'Đạt' if payload['threshold_check']['nDCG@10>0.7'] else 'Không đạt'} ({hybrid_m['nDCG@10']:.3f})",
        f"- MAP > 0.5: {'Đạt' if payload['threshold_check']['MAP>0.5'] else 'Không đạt'} ({hybrid_m['MAP']:.3f})",
        f"- Kết luận: **{'Có thể deploy có kiểm soát' if payload['conclusion']=='deploy_ready' else 'Chưa nên deploy production'}**",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved:")
    print(f"- {metrics_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
