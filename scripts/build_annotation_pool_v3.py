from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

root = Path(r"C:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
sys.path.insert(0, str(root))

from retrieval.search_engine import search_hybrid
from retrieval.query_understanding import understand_query


def _model_results(query: str, index_dir: Path, stopwords: Path):
    # Hybrid baseline
    hybrid, qu = search_hybrid(query, index_dir, stopwords, top_k=10, vector_weight=0.6, bm25_weight=0.4, location_boost_factor=1.8)
    bm25, _ = search_hybrid(query, index_dir, stopwords, top_k=10, vector_weight=0.0, bm25_weight=1.0, location_boost_factor=1.0)
    vector, _ = search_hybrid(query, index_dir, stopwords, top_k=10, vector_weight=1.0, bm25_weight=0.0, location_boost_factor=1.0)
    return qu, bm25, vector, hybrid


def _score_map(items):
    out = {}
    for rank, r in enumerate(items, start=1):
        hid = str(r.get("source_hotel_id", "")).strip()
        if hid:
            out[hid] = {
                "rank": rank,
                "score": float(r.get("hybrid_score", 0.0)),
                "location_matched": bool(r.get("location_matched", False)),
                "descriptor_matched": bool(r.get("descriptor_matched", False)),
                "category_matched": bool(r.get("category_matched", False)),
                "hotel_name": r.get("hotel_name", ""),
                "location": r.get("location", ""),
                "top_reviews": r.get("top_reviews", []),
            }
    return out


def _relevance_label(query: str, qu, item: dict) -> tuple[int, str]:
    q = query.lower()
    hotel_name = str(item.get("hotel_name", "")).lower()
    location = str(item.get("location", "")).lower()
    top_reviews = item.get("top_reviews", []) or []
    review_text = " ".join(str(t) for t in top_reviews).lower()

    has_location = bool(qu.detected_location)
    loc_ok = bool(item.get("location_matched", False))
    desc_ok = bool(item.get("descriptor_matched", False))
    cat_ok = bool(item.get("category_matched", False))

    type_hits = any(tok in q for tok in ["resort", "homestay", "villa", "khách sạn", "khach san", "hotel", "nhà nghỉ", "nha nghi"])
    type_ok = any(tok in hotel_name for tok in ["resort", "homestay", "villa", "hotel", "boutique", "motel", "guesthouse", "hostel", "aparthotel", "bungalow"])

    # Conservative label rules:
    # - Need location match when query mentions a location.
    # - Need either descriptor/category signal OR type hint for intent.
    # - Penalize if review text itself suggests mismatch or negative patterns.
    negative_flags = ["không", "chật", "bẩn", "dơ", "hôi", "tệ", "xấu", "xa", "khó", "mắc", "đắt"]
    negative_count = sum(1 for tok in negative_flags if tok in review_text)

    if has_location and not loc_ok:
        return 0, "location_miss"

    if (desc_ok or cat_ok or type_ok) and (loc_ok or not has_location):
        if negative_count >= 3 and not (desc_ok and cat_ok):
            return 0, "too_many_negative_signals"
        if desc_ok and (cat_ok or type_ok or loc_ok):
            return 1, "strong_match"
        if cat_ok and loc_ok and type_ok:
            return 1, "category_location_type_match"
        if type_hits and loc_ok:
            return 1, "type_location_match"
        if not has_location and (desc_ok or cat_ok):
            return 1, "intent_only_match"
        if loc_ok and (desc_ok or cat_ok):
            return 1, "location_and_intent_match"

    return 0, "insufficient_signal"


def main():
    queries_path = root / "data" / "evaluation" / "test_queries.json"
    index_dir = root / "data" / "index"
    stopwords = root / "config" / "stopwords.txt"

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    rows = []

    for i, item in enumerate(queries, start=1):
        qid = f"Q{i}"
        query = item["query"]
        qu, bm25, vec, hyb = _model_results(query, index_dir, stopwords)
        bm25_map = _score_map(bm25)
        vec_map = _score_map(vec)
        hyb_map = _score_map(hyb)

        candidate_ids = set(bm25_map) | set(vec_map) | set(hyb_map)
        for hid in candidate_ids:
            b = bm25_map.get(hid, {})
            v = vec_map.get(hid, {})
            h = hyb_map.get(hid, {})
            rel, note = _relevance_label(query, qu, h or b or v)
            rows.append({
                "query_id": qid,
                "query": query,
                "hotel_id": hid,
                "hotel_name": (h or b or v).get("hotel_name", ""),
                "location": (h or b or v).get("location", ""),
                "relevance": rel,
                "note": note,
                "query_location": qu.detected_location,
                "query_categories": ",".join(qu.detected_categories),
                "query_descriptors": ",".join(qu.descriptor_tokens),
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
            })

    rows.sort(key=lambda r: (int(r["query_id"][1:]), -int(r["relevance"]), int(r["hybrid_rank"]) if str(r.get("hybrid_rank", "")).isdigit() else 999))

    out_csv = root / "evaluation" / "annotation_pool_v3.csv"
    out_json = root / "evaluation" / "annotation_pool_v3.json"
    out_xlsx = root / "evaluation" / "annotation_pool_v3.xlsx"

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
        ws.title = "annotation_pool_v3"
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h, "") for h in headers])
        wb.save(out_xlsx)
    except Exception as e:
        print(f"XLSX export failed: {e}")

    total_rel = sum(1 for r in rows if int(r["relevance"]) == 1)
    print(f"Saved {len(rows)} rows; positives={total_rel}")
    print(out_csv)
    print(out_json)
    print(out_xlsx)


if __name__ == "__main__":
    main()
