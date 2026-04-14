"""
app_gui.py — Giao diện web tìm kiếm khách sạn (Cấp độ Hotel-Level)
Chạy: streamlit run app_gui.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from api.service import answer_with_rag
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


def run_rag(
    query: str,
    top_k_retrieval: int,
    top_k_context: int,
    max_citations: int,
    vector_weight: float,
    bm25_weight: float,
    loc_boost: float,
    chat_history: list[dict],
):
    try:
        payload = answer_with_rag(
            query=query.strip(),
            top_k_retrieval=top_k_retrieval,
            top_k_context=top_k_context,
            max_citations=max_citations,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            location_boost_factor=loc_boost,
            chat_history=chat_history,
            allow_fallback_to_ir=True,
            explain=True,
        )
    except Exception as e:
        return None, f"Lỗi RAG: {e}"
    return payload, None


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

    st.title("🏨 Hotel Advisor Chatbot (RAG)")
    st.caption("Chatbot tư vấn khách sạn theo nhu cầu người dùng, trả lời tự nhiên dựa trên dữ liệu retrieval.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None

    with st.sidebar:
        st.header("🧩 Bối cảnh tư vấn")
        scene = st.selectbox(
            "Khách hàng thuộc nhóm nào?",
            [
                "Không xác định",
                "Gia đình có trẻ nhỏ",
                "Cặp đôi nghỉ dưỡng",
                "Du lịch tiết kiệm",
                "Chuyến công tác",
                "Nhóm bạn trẻ",
            ],
        )
        budget = st.selectbox("Ngân sách", ["Không rõ", "Tiết kiệm", "Trung cấp", "Cao cấp"])
        priority = st.multiselect(
            "Ưu tiên chính",
            ["Gần biển", "Gần trung tâm", "Yên tĩnh", "Ăn sáng ngon", "Hồ bơi", "View đẹp", "Di chuyển thuận tiện"],
            default=["Gần biển"],
        )

        st.header("⚙️ Retrieval Settings")
        vector_weight = st.slider("Vector Weight (Semantic)", 0.0, 1.0, 0.6, 0.1)
        bm25_weight = st.slider("BM25 Weight (Keyword)", 0.0, 1.0, 0.4, 0.1)
        loc_boost = st.slider("Location Boost Factor", 1.0, 3.0, 1.8, 0.1)

        top_k = st.number_input("Top-K retrieval", min_value=1, max_value=50, value=12)
        top_k_context = st.number_input("Số context cho RAG", min_value=1, max_value=20, value=6)
        max_citations = st.number_input("Số citation tối đa", min_value=1, max_value=10, value=4)
        show_debug = st.toggle("Hiện debug payload", value=False)

    st.caption("Mẹo: hỏi theo kiểu hội thoại, ví dụ: 'Tôi đi gia đình 4 người, cần gần biển Đà Nẵng, tầm giá trung cấp'.")
    if st.button("🧹 Xóa hội thoại", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_payload = None

    for turn in st.session_state.chat_history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.write(turn["content"])

    user_input = st.chat_input("Nhập yêu cầu khách sạn của bạn...")
    if user_input:
        scene_context = (
            f"Boi canh khach hang: {scene}. "
            f"Ngan sach: {budget}. "
            f"Uu tien: {', '.join(priority) if priority else 'khong ro'}."
        )
        augmented_query = f"{user_input}\n{scene_context}"

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Đang tư vấn..."):
                rag_payload, err = run_rag(
                    query=augmented_query,
                    top_k_retrieval=int(top_k),
                    top_k_context=int(top_k_context),
                    max_citations=int(max_citations),
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    loc_boost=loc_boost,
                    chat_history=st.session_state.chat_history,
                )

            if err:
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": f"Lỗi hệ thống: {err}"})
                return

            answer = rag_payload.get("answer", "Mình chưa có đủ dữ liệu để tư vấn lúc này.")
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.last_payload = rag_payload

            mode = rag_payload.get("mode", "rag")
            grounded = rag_payload.get("grounded", False)
            fallback_used = rag_payload.get("fallback_used", False)
            st.caption(f"Mode: `{mode}` | Grounded: {'✅' if grounded else '❌'} | Fallback: {'✅' if fallback_used else '❌'}")

            citations = rag_payload.get("citations", [])
            if citations:
                with st.expander("📚 Nguồn tham chiếu", expanded=False):
                    for c in citations:
                        st.markdown(
                            f"- **[{c.get('id')}] {c.get('hotel_name')}** ({c.get('location')})  \n"
                            f"  _{c.get('snippet')}_"
                        )

            ir_results = rag_payload.get("ir_results", [])
            if ir_results:
                with st.expander("🔁 Kết quả IR fallback", expanded=False):
                    for i, r in enumerate(ir_results, 1):
                        st.markdown(
                            f"**{i}. {r.get('hotel_name', 'Không tên')}** — {r.get('location', 'Không rõ')} "
                            f"(score: {r.get('hybrid_score', 0):.3f})"
                        )

    if show_debug and st.session_state.last_payload:
        with st.expander("🔧 Debug payload (lần trả lời gần nhất)", expanded=False):
            st.json(st.session_state.last_payload)
            if st.session_state.last_payload.get("query_understanding"):
                save_debug_log(
                    query=st.session_state.last_payload.get("query", ""),
                    results=st.session_state.last_payload.get("ir_results", []),
                    query_understanding=st.session_state.last_payload.get("query_understanding", {}),
                    weights={
                        "vector_weight": vector_weight,
                        "bm25_weight": bm25_weight,
                        "location_boost": loc_boost,
                        "top_k_retrieval": int(top_k),
                    },
                )

    # Chế độ IR cũ giữ để đối chiếu nhanh khi cần.
    with st.expander("🧪 So sánh nhanh IR truyền thống", expanded=False):
        ir_query = st.text_input("Truy vấn IR nhanh", placeholder="VD: khách sạn gần biển ở Đà Nẵng")
        if st.button("Chạy IR nhanh"):
            results, qu, err = run_search(ir_query, int(top_k), vector_weight, bm25_weight, loc_boost)
            if err:
                st.error(err)
            elif not results:
                st.info("Không có kết quả IR.")
            else:
                st.success(f"IR trả về {len(results)} kết quả.")
                st.write(
                    {
                        "detected_location": qu.detected_location,
                        "detected_categories": qu.detected_categories,
                        "descriptor_tokens": qu.descriptor_tokens,
                    }
                )
                for i, r in enumerate(results[:5], 1):
                    st.markdown(
                        f"**{i}. {r.get('hotel_name', 'Không tên')}** — {r.get('location', 'Không rõ')} "
                        f"(score: {r.get('hybrid_score', 0):.3f})"
                    )

if __name__ == "__main__":
    main()
