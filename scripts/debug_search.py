"""
Debug Search Pipeline (Hotel-level)
Test kết quả trả về với cấu trúc Ranking Pipeline mới.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from retrieval.query_understanding import understand_query
from retrieval.search_engine import search_hybrid

INDEX_DIR = project_root / "data" / "index"
STOPWORDS_PATH = project_root / "config" / "stopwords.txt"

TEST_QUERIES = [
    "khách sạn view biển đẹp ở phú quốc",
    "resort hồ bơi có nhà hàng nha trang",
    "homestay giá rẻ đà lạt sỉ nhân viên thân thiện",
    "villa sang trọng vũng tàu",
]

def run_debug():
    print("=" * 70)
    print("RANKING PIPELINE TEST - HOTEL LEVEL")
    print("=" * 70)

    for query in TEST_QUERIES:
        print(f"\n📌 Query: \"{query}\"")
        print("-" * 60)

        try:
            results, qu = search_hybrid(
                query=query,
                index_dir=INDEX_DIR,
                stopwords_path=STOPWORDS_PATH,
                top_k=5,
                vector_weight=0.6,
                bm25_weight=0.4,
                location_boost_factor=1.8
            )
        except Exception as e:
            print(f"  ❌ Lỗi search: {e}")
            break

        print(f"  Core tokens: {qu.core_tokens}")
        print(f"  Expanded:    {qu.expanded_tokens[:6]}...")
        print(f"  Location:    {qu.detected_location or '(none)'}")
        print(f"  Descriptors: {qu.descriptor_tokens}")

        print("\n  Top-5 Hotels (Hybrid Score):")
        if not results:
            print("  (Không tìm thấy khách sạn nào)")

        for i, r in enumerate(results, 1):
            loc_flag = "📍(boosted)" if r["location_matched"] else "📍(none)"
            print(f"  {i}. {r['hotel_name']} — {r['location']} {loc_flag}")
            print(f"     => Score: {r['hybrid_score']:.4f} (Vec: {r['vector_score']:.4f} | BM25: {r['bm25_score']:.4f})")
            print(f"     => Reviews đính kèm: {r['review_count']}")
            if r['top_reviews']:
                print(f"        - \"{r['top_reviews'][0][:100]}...\"")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_debug()
