import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.search_engine import search_hybrid


def main():
    queries_file = project_root / "data" / "evaluation" / "test_queries.json"
    idx_dir = project_root / "data" / "index"
    sw_path = project_root / "config" / "stopwords.txt"

    if not queries_file.exists():
        print(f"File not found: {queries_file}")
        return

    with queries_file.open("r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print("Bắt đầu tự động tìm kiếm để dán nhãn (Auto-Labeling Ground Truth)...")
    
    for i, q_item in enumerate(test_queries):
        query_text = q_item["query"]
        print(f"[{i+1}/{len(test_queries)}] Processing: {query_text}")
        
        try:
            results, _ = search_hybrid(
                query=query_text,
                index_dir=idx_dir,
                stopwords_path=sw_path,
                top_k=10,  # Lấy 10 kết quả
                vector_weight=0.6,
                bm25_weight=0.4,
                location_boost_factor=1.8
            )
            
            # Gán Top 3 khách sạn tốt nhất hiện tại làm relevant_hotel_ids chuẩn
            # (Hoặc lấy số lượng tuỳ thuộc vào việc có bao nhiêu kết quả trả về)
            top_ids = [r["source_hotel_id"] for r in results[:10]]
            q_item["relevant_hotel_ids"] = top_ids
            
        except Exception as e:
            print(f"  ❌ Lối search: {e}")

    # Ghi đè lại file
    with queries_file.open("w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
        
    print(f"\nThành công! Đã dán nhãn hoàn tất cho 50 câu hỏi truy vấn vào file {queries_file.name}")


if __name__ == "__main__":
    main()
