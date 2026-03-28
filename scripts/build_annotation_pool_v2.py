from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
sys.path.insert(0, str(ROOT))

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


def _norm(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d")
    return " ".join(text.split())


def _flatten_review_text(top_reviews) -> str:
    return " ".join(str(t) for t in (top_reviews or []))


def _load_corpus_hotels(index_dir: Path) -> dict[str, dict]:
    payload = se.load_index(index_dir / "bm25_index.pkl")
    docs = payload.get("documents", []) if isinstance(payload, dict) else []
    hotel_map: dict[str, dict] = {}
    for d in docs:
        sid = str(d.get("source_hotel_id", "")).strip()
        if not sid or sid in hotel_map:
            continue
        hotel_map[sid] = {
            "hotel_id": sid,
            "hotel_name": d.get("hotel_name", ""),
            "location": d.get("location", ""),
            "rating": d.get("rating", ""),
            "types": list(d.get("types", []) or []),
        }
    return hotel_map


def _retrieve_models(query: str, index_dir: Path, stopwords: Path):
    hybrid, qu = se.search_hybrid(query, index_dir, stopwords, top_k=15, vector_weight=0.6, bm25_weight=0.4, location_boost_factor=1.8)
    bm25, _ = se.search_hybrid(query, index_dir, stopwords, top_k=15, vector_weight=0.0, bm25_weight=1.0, location_boost_factor=1.0)
    vector, _ = se.search_hybrid(query, index_dir, stopwords, top_k=15, vector_weight=1.0, bm25_weight=0.0, location_boost_factor=1.0)
    return qu, bm25, vector, hybrid


def _score_map(items):
    out = {}
    for rank, r in enumerate(items, start=1):
        hid = str(r.get("source_hotel_id", "")).strip()
        if not hid:
            continue
        out[hid] = {
            "rank": rank,
            "score": float(r.get("hybrid_score", 0.0)),
            "location_matched": bool(r.get("location_matched", False)),
            "descriptor_matched": bool(r.get("descriptor_matched", False)),
            "category_matched": bool(r.get("category_matched", False)),
            "hotel_name": r.get("hotel_name", ""),
            "location": r.get("location", ""),
            "top_reviews": r.get("top_reviews", []),
            "types": list(r.get("types", []) or []),
        }
    return out


def _detect_type_tokens(query: str) -> set[str]:
    q = _norm(query)
    tokens = set()
    mapping = {
        "hotel": ["khach san", "khách sạn", "hotel"],
        "resort": ["resort"],
        "homestay": ["homestay"],
        "villa": ["villa"],
        "hostel": ["hostel"],
        "guesthouse": ["guesthouse"],
        "aparthotel": ["aparthotel"],
        "motel": ["motel"],
        "bungalow": ["bungalow"],
        "nhà nghỉ": ["nha nghi", "nhà nghỉ"],
    }
    for canon, forms in mapping.items():
        if any(f in q for f in forms):
            tokens.add(canon)
    return tokens


def _pick_negative_hotels(query: str, qu, corpus_hotels: dict[str, dict], exclude_ids: set[str], k: int, rng: random.Random) -> list[dict]:
    q_loc = _norm(qu.detected_location or "")
    q_types = _detect_type_tokens(query)

    candidates = []
    for hid, info in corpus_hotels.items():
        if hid in exclude_ids:
            continue
        loc = _norm(info.get("location", ""))
        types = {str(t).strip().lower() for t in info.get("types", []) or []}
        score = 0
        if q_loc and q_loc in loc:
            score += 2
        if q_types and types.intersection(q_types):
            score += 2
        if q_loc and not loc:
            score += 0
        candidates.append((score, rng.random(), info))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    picked = []
    for _, _, info in candidates:
        picked.append(info)
        if len(picked) >= k:
            break
    return picked


def _relevance_grade(query: str, qu, item: dict) -> tuple[int, str]:
    q = _norm(query)
    hotel_name = _norm(item.get("hotel_name", ""))
    location = _norm(item.get("location", ""))
    review_text = _norm(_flatten_review_text(item.get("top_reviews", [])))
    doc_types = {str(t).strip().lower() for t in item.get("types", []) or []}

    q_loc = _norm(qu.detected_location or "")
    q_types = _detect_type_tokens(query)
    q_desc = { _norm(t) for t in (getattr(qu, "descriptor_tokens", []) or []) if str(t).strip() }
    q_cats = { _norm(t) for t in (getattr(qu, "detected_categories", []) or []) if str(t).strip() }

    has_location = bool(q_loc)
    loc_ok = bool(q_loc and q_loc in location)
    type_ok = bool(q_types and (doc_types.intersection(q_types) or any(t in hotel_name for t in q_types)))
    desc_ok = False
    cat_ok = False
    if q_desc:
        desc_ok = any(tok in review_text for tok in q_desc)
    if q_cats:
        cat_ok = any(tok in review_text for tok in q_cats)

    negative_flags = ["không", "chật", "bẩn", "dơ", "hôi", "tệ", "xấu", "xa", "khó", "mắc", "đắt"]
    negative_count = sum(1 for tok in negative_flags if tok in review_text)

    if has_location and not loc_ok:
        return 0, "location_miss"

    score = sum(1 for flag in (loc_ok, type_ok, desc_ok, cat_ok) if flag)

    if negative_count >= 3:
        score = max(0, score - 2)
    elif negative_count >= 1 and score > 0:
        score -= 1

    if loc_ok and (type_ok or desc_ok or cat_ok):
        return 2, "strong_match"
    if not has_location and sum(1 for flag in (type_ok, desc_ok, cat_ok) if flag) >= 2:
        return 2, "strong_match"
    if score >= 2 and (loc_ok or not has_location):
        return 2, "strong_match"
    if score >= 1 and (loc_ok or not has_location):
        return 1, "partial_match"
    if score == 0:
        return 0, "insufficient_signal"
    return 0, "weak_or_mismatch"


def main():
    parser = argparse.ArgumentParser(description="Build v2 bucket-aware annotation pool with graded labels")
    parser.add_argument("--queries", default=str(ROOT / "data" / "evaluation" / "test_queries_200_bucketed.json"))
    parser.add_argument("--output-prefix", default=str(ROOT / "evaluation" / "annotation_pool_bucketed_v2"))
    parser.add_argument("--index-dir", default=str(ROOT / "data" / "index"))
    parser.add_argument("--stopwords", default=str(ROOT / "config" / "stopwords.txt"))
    parser.add_argument("--negatives-per-query", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    out_prefix = Path(args.output_prefix)
    index_dir = Path(args.index_dir)
    stopwords = Path(args.stopwords)
    rng = random.Random(args.seed)

    corpus_hotels = _load_corpus_hotels(index_dir)
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    rows = []
    for item in queries:
        qid = str(item.get("query_id", "")).strip()
        query = str(item.get("query", "")).strip()
        if not qid or not query:
            continue

        bucket_id = str(item.get("bucket_id", "")).strip()
        bucket_name = str(item.get("bucket_name", "")).strip()
        bucket_description = str(item.get("bucket_description", "")).strip()
        bucket_order = item.get("bucket_order", "")

        qu, bm25, vec, hyb = _retrieve_models(query, index_dir, stopwords)
        bm25_map = _score_map(bm25)
        vec_map = _score_map(vec)
        hyb_map = _score_map(hyb)

        candidate_ids = set(bm25_map) | set(vec_map) | set(hyb_map)
        negatives = _pick_negative_hotels(query, qu, corpus_hotels, candidate_ids, args.negatives_per_query, rng)
        for neg in negatives:
            candidate_ids.add(neg["hotel_id"])

        candidate_rows: dict[str, dict] = {}
        for hid in candidate_ids:
            base = hyb_map.get(hid) or bm25_map.get(hid) or vec_map.get(hid)
            if base:
                candidate_rows[hid] = base
            else:
                candidate_rows[hid] = corpus_hotels.get(hid, {"hotel_id": hid, "hotel_name": "", "location": "", "types": []})

        # add negative candidates if not already present in retrieved set
        for neg in negatives:
            hid = neg["hotel_id"]
            candidate_rows[hid] = {
                "hotel_id": hid,
                "hotel_name": neg.get("hotel_name", ""),
                "location": neg.get("location", ""),
                "types": list(neg.get("types", []) or []),
                "top_reviews": [],
            }

        for hid, base in candidate_rows.items():
            b = bm25_map.get(hid, {})
            v = vec_map.get(hid, {})
            h = hyb_map.get(hid, {})
            grade, note = _relevance_grade(query, qu, base)
            rows.append({
                "query_id": qid,
                "bucket_id": bucket_id,
                "bucket_name": bucket_name,
                "bucket_description": bucket_description,
                "bucket_order": bucket_order,
                "query": query,
                "hotel_id": hid,
                "hotel_name": base.get("hotel_name", ""),
                "location": base.get("location", ""),
                "relevance": grade,
                "note": note,
                "query_location": qu.detected_location,
                "query_categories": ",".join(getattr(qu, "detected_categories", []) or []),
                "query_descriptors": ",".join(getattr(qu, "descriptor_tokens", []) or []),
                "bm25_rank": b.get("rank", ""),
                "bm25_score": f"{b.get('score', 0.0):.6f}" if b else "",
                "vector_rank": v.get("rank", ""),
                "vector_score": f"{v.get('score', 0.0):.6f}" if v else "",
                "hybrid_rank": h.get("rank", ""),
                "hybrid_score": f"{h.get('score', 0.0):.6f}" if h else "",
                "bm25_loc_match": b.get("location_matched", False),
                "vector_loc_match": v.get("location_matched", False),
                "hybrid_loc_match": h.get("location_matched", False),
                "bm25_desc_match": b.get("descriptor_matched", False),
                "vector_desc_match": v.get("descriptor_matched", False),
                "hybrid_desc_match": h.get("descriptor_matched", False),
                "bm25_cat_match": b.get("category_matched", False),
                "vector_cat_match": v.get("category_matched", False),
                "hybrid_cat_match": h.get("category_matched", False),
                "source": "bucketed_v2",
            })

    rows.sort(key=lambda r: (int(r["query_id"][1:]), -int(r["relevance"]), int(r["hybrid_rank"]) if str(r.get("hybrid_rank", "")).isdigit() else 999))

    out_csv = out_prefix.with_suffix(".csv")
    out_json = out_prefix.with_suffix(".json")
    out_xlsx = out_prefix.with_suffix(".xlsx")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "annotation_pool_bucketed_v2"
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h, "") for h in headers])
        wb.save(out_xlsx)
    except Exception as e:
        print(f"XLSX export failed: {e}")

    total = len(rows)
    positives = sum(1 for r in rows if int(r["relevance"]) > 0)
    strong = sum(1 for r in rows if int(r["relevance"]) >= 2)
    print(f"Saved {total} rows; positives={positives}; strong={strong}")
    print(out_csv)
    print(out_json)
    print(out_xlsx)


if __name__ == "__main__":
    main()
