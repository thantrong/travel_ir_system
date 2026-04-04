"""
app_gui.py — Giao diện web tìm kiếm khách sạn (Cấp độ Hotel-Level)
Chạy: streamlit run app_gui.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import streamlit as st
from retrieval.search_engine import search_hybrid
from summarization.debug_logger import save_debug_log


def get_index_dir() -> Path:
    return project_root / "data" / "index"


def get_stopwords_path() -> Path:
    return project_root / "config" / "stopwords.txt"


def run_search(
    query: str,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    loc_boost: float
):
    index_dir = get_index_dir()
    stopwords_path = get_stopwords_path()

    try:
        results, qu = search_hybrid(
            query=query.strip(),
            index_dir=index_dir,
            stopwords_path=stopwords_path,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            location_boost_factor=loc_boost
        )
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Lỗi tìm kiếm: {e}"

    return results, qu, None


def main():
    st.set_page_config(
        page_title="Hotel Search Engine",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .hotel-card {
            padding: 1.25rem; margin-bottom: 1rem;
            border-radius: 12px;
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        .loc-badge { background: #f0fdf4; color: #15803d; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; font-weight:600;}
        .score-badge { background: #eff6ff; color: #1d4ed8; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; }
        .review-box { background: #f1f5f9; padding: 0.75rem; border-radius: 8px; border-left: 4px solid #94a3b8; margin-top: 0.5rem; font-style: italic; font-size: 0.95rem; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏨 Hotel Search Engine (Hybrid IR)")
    st.caption("Tìm kiếm ngữ nghĩa + từ khóa + location boosting trên cấp độ Khách sạn.")

    with st.sidebar:
        st.header("⚙️ Ranking Weights")
        vector_weight = st.slider("Vector Weight (Semantic)", 0.0, 1.0, 0.6, 0.1)
        bm25_weight = st.slider("BM25 Weight (Keyword)", 0.0, 1.0, 0.4, 0.1)
        loc_boost = st.slider("Location Boost Factor", 1.0, 3.0, 1.8, 0.1)
        
        top_k = st.number_input("Số kết quả (Top-K)", min_value=1, max_value=50, value=10)
        
    query = st.text_input(
        "Nhập câu truy vấn (tiếng Việt)",
        placeholder="Ví dụ: khách sạn view biển đẹp ở phú quốc",
        key="query_input",
    )


    if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True):
        if not query or not query.strip():
            st.warning("Vui lòng nhập truy vấn.")
            return

        with st.spinner("Đang trích xuất và tìm kiếm..."):
            results, qu, err = run_search(query, top_k, vector_weight, bm25_weight, loc_boost)

        if err:
            st.error(err)
            return
            
        # Hiển thị Query Understanding
        with st.expander("🧠 Trích xuất Ngữ nghĩa (Query Understanding)", expanded=True):
            cols = st.columns(4)
            cols[0].metric("Tokens cốt lõi", ", ".join(qu.core_tokens) if qu.core_tokens else "Trống")
            cols[1].metric("Tokens nâng cao (Synonyms)", ", ".join(qu.expanded_tokens) if qu.expanded_tokens else "Trống")
            cols[2].metric("Điểm đến (Location)", qu.detected_location.replace("_", " ").title() if qu.detected_location else "Không rõ")
            cols[3].metric("Từ khoá nổi bật", ", ".join(qu.descriptor_tokens) if qu.descriptor_tokens else "Trống")

        # Lưu debug log
        weights_info = {
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "location_boost": loc_boost,
            "top_k": top_k,
        }
        qu_info = {
            "core_tokens": qu.core_tokens,
            "expanded_tokens": qu.expanded_tokens,
            "detected_location": qu.detected_location,
            "descriptor_tokens": qu.descriptor_tokens,
            "detected_categories": qu.detected_categories,
        }
        debug_path = save_debug_log(query, results, qu_info, weights_info)
        st.info(f"📋 Đã lưu debug log: `{debug_path.name}`")

        if not results:
            st.info("Không tìm thấy khách sạn phù hợp với truy vấn.")
            return

        st.success(f"Tìm thấy top {len(results)} khách sạn phù hợp nhất.")

        # Debug Panel
        with st.expander("🔧 Debug: Score Breakdown", expanded=False):
            st.caption("Phân tích điểm số từng khách sạn - Giúp hiểu tại sao khách sạn ở vị trí này")
            for i, r in enumerate(results, 1):
                d = r.get("debug_info", {})
                if d:
                    with st.container(border=True):
                        st.markdown(f"**{i}. {d.get('hotel_name', r.get('hotel_name', ''))}** (Score: {r.get('hybrid_score', 0):.4f})")
                        cols = st.columns([2, 1, 1, 1, 2])
                        cols[0].metric("Aggregation", d.get("score_after_aggregation", 0))
                        cols[1].metric("Category", d.get("score_after_category_boost", 0))
                        cols[2].metric("Location", d.get("score_after_location_boost", 0))
                        cols[3].metric("Final", d.get("score_final", 0))
                        
                        st.markdown(f"- Location detected: `{d.get('query_detected_location', 'Không')}` | Matched: {'✅' if d.get('location_matched') else '❌'}")
                        st.markdown(f"- Categories detected: `{d.get('query_categories_detected', [])}` | Doc: `{d.get('doc_categories', [])}`")
                        st.markdown(f"- Category matched: {'✅' if d.get('category_matched') else '❌'}")
                        st.markdown(f"- Descriptor tokens: `{d.get('descriptor_tokens_used', [])}` | Supported: {'✅' if d.get('descriptor_supported') else '❌'}")
                        type_status = ""
                        if d.get('type_boost_applied'):
                            type_status = f"✅ Boosted x{d.get('types_boot_factor', '')}"
                        elif d.get('type_mismatch_penalty'):
                            type_status = "⚠️ Penalty (0.7x) - Không đúng loại hình"
                        else:
                            type_status = "❌ Không áp dụng"
                        st.markdown(f"- Type boost applied: {type_status}")
                        if d.get('type_boost_keywords_found'):
                            st.markdown(f"- Keywords tìm loại hình: `{d.get('type_boost_keywords_found')}`")
                        st.markdown(f"- Acc types khách sạn: `{d.get('hotel_acc_types', [])}`")
                        if d.get('hotel_type_matched') is not None:
                            st.markdown(f"- Hotel type matched query: {'✅' if d.get('hotel_type_matched') else '❌'}")

        for i, r in enumerate(results, 1):
            name = r.get("hotel_name", "Không tên")
            loc = r.get("location", "Không rõ")
            rating = r.get("rating", "—")
            reviews_count = r.get("review_count", 0)
            
            score_hybrid = r.get("hybrid_score", 0.0)
            score_vec = r.get("vector_score", 0.0)
            score_bm25 = r.get("bm25_score", 0.0)
            
            loc_matched = r.get("location_matched", False)

            with st.container():
                st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
                
                c1, c2 = st.columns([5, 2])
                with c1:
                    badges = ""
                    if loc_matched:
                        badges += ' <span class="loc-badge">📍 Đúng khu vực</span>'
                    st.markdown(f"### {i}. {name} {badges}", unsafe_allow_html=True)
                    st.caption(f"📍 {loc}  •  ⭐ {rating}  •  💬 {reviews_count} đánh giá tham chiếu")
                
                with c2:
                    st.markdown(f'<div style="text-align: right;"><span class="score-badge">🏆 Score: {score_hybrid:.3f}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: right; font-size: 0.8rem; color: #64748b; margin-top: 4px;">Top Vec: {score_vec:.3f} | Top BM25: {score_bm25:.3f}</div>', unsafe_allow_html=True)

                top_reviews = r.get("top_reviews", [])
                if top_reviews:
                    with st.expander("📝 Top đánh giá liên quan nhất từ khách hàng"):
                        for txt in top_reviews:
                            st.markdown(f'<div class="review-box">💬 "{txt}"</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
