from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(r"C:\Users\admin\.openclaw\workspace\projects\travel_ir_system")
sys.path.insert(0, str(root))

from retrieval.search_engine import search_hybrid
from retrieval.query_understanding import understand_query


def main():
    queries = json.loads((root / "data" / "evaluation" / "test_queries.json").read_text(encoding="utf-8"))
    index_dir = root / "data" / "index"
    stopwords = root / "config" / "stopwords.txt"

    rows = []
    for i, item in enumerate(queries, start=1):
        qid = f"Q{i}"
        q = item["query"]
        qu = understand_query(q, stopwords)
        results, _ = search_hybrid(
            q,
            index_dir,
            stopwords,
            top_k=10,
            vector_weight=0.6,
            bm25_weight=0.4,
            location_boost_factor=1.8,
        )
        for rank, r in enumerate(results, start=1):
            hid = str(r.get("source_hotel_id", "")).strip()
            name = r.get("hotel_name", "")
            loc = r.get("location", "")
            cat_match = bool(r.get("category_matched", False))
            desc_match = bool(r.get("descriptor_matched", False))
            loc_match = bool(r.get("location_matched", False))
            hotel_text = f"{name} {loc}".lower()
            ql = q.lower()

            relevance = 0
            note = "manual_review"
            if loc_match and (cat_match or desc_match):
                relevance = 1
                note = "location_and_intent_match"
            elif loc_match and any(tok in hotel_text for tok in ["resort", "homestay", "villa", "hotel", "boutique"]):
                relevance = 1 if any(t in ql for t in ["resort", "homestay", "villa", "khách sạn", "khach san", "hotel", "nhà nghỉ", "nha nghi"]) else 0
                note = "location_match_type_hint" if relevance else "location_only"
            elif (cat_match or desc_match) and qu.detected_location == "":
                relevance = 1
                note = "intent_match_without_location"
            elif cat_match or desc_match:
                relevance = 0
                note = "intent_partial"

            rows.append(
                {
                    "query_id": qid,
                    "query": q,
                    "hotel_id": hid,
                    "hotel_name": name,
                    "location": loc,
                    "retrieved_by": "hybrid",
                    "rank_hybrid": rank,
                    "score_hybrid": f"{float(r.get('hybrid_score', 0.0)):.6f}",
                    "category_matched": cat_match,
                    "descriptor_matched": desc_match,
                    "location_matched": loc_match,
                    "relevance": relevance,
                    "note": note,
                }
            )

    out_json = root / "evaluation" / "annotation_pool_v2.json"
    out_csv = root / "evaluation" / "annotation_pool_v2.csv"
    out_xlsx = root / "evaluation" / "annotation_pool_v2.xlsx"

    if rows:
        headers = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "annotation_pool_v2"
            ws.append(headers)
            for row in rows:
                ws.append([row.get(h, "") for h in headers])
            wb.save(out_xlsx)
        except Exception as e:
            print(f"XLSX creation failed: {e}")

    print(f"Saved {len(rows)} rows")
    print(out_csv)
    print(out_json)
    print(out_xlsx)


if __name__ == "__main__":
    main()
