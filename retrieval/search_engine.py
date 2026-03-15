"""
Search Engine (Review-Level Indexing -> Hotel-level Aggregation)
PHẦN 6: REVIEW AGGREGATION
PHẦN 7: LOCATION BOOSTING
PHẦN 10: RANKING PIPELINE
"""

import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

from retrieval.query_understanding import understand_query, location_matched, QueryUnderstandingResult
from summarization import summarize_reviews_tfidf
from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi


_CATEGORY_TEXT_HINTS: dict[str, tuple[str, ...]] = {
    "mountain": ("núi", "đồi", "thung lũng", "cao nguyên", "săn mây", "view núi", "hướng núi"),
    "beach": ("biển", "bãi biển", "ven biển", "gần biển", "đảo", "view biển"),
    "family": ("gia đình", "trẻ em", "family", "kid"),
    "budget": ("giá rẻ", "bình dân", "tiết kiệm", "hợp lý"),
    "luxury": ("sang trọng", "cao cấp", "luxury", "5 sao", "resort", "villa"),
<<<<<<< HEAD
    "center": ("trung tâm", "phố đi bộ", "chợ đêm", "gần chợ","phố cổ","hồ gươm", "hồ tây"),
    "amenity_pool": ("hồ bơi", "bể bơi", "pool", "vô cực"),
    "amenity_breakfast": ("ăn sáng", "buffet", "điểm tâm"),
    "airport": ("sân bay", "tân sơn nhất", "nội bài", "ga"),
    "spa_gym": ("gym", "spa", "massage", "thể hình", "xông hơi", "bồn tắm", "jacuzzi"),
    "kitchen": ("bếp", "nấu ăn", "bbq", "nướng"),
    "photo": ("sống ảo", "chụp hình", "vintage", "đẹp", "decor"),
    "quiet": ("yên tĩnh", "nghỉ dưỡng", "đọc sách", "người lớn tuổi"),
    "pet": ("thú cưng", "chó", "mèo", "pet")
=======
    "center": ("trung tâm", "phố đi bộ", "chợ đêm", "gần chợ"),
    "amenity_pool": ("hồ bơi", "bể bơi", "pool", "vô cực"),
    "amenity_breakfast": ("ăn sáng", "buffet", "điểm tâm"),
>>>>>>> 49a78af (cập nhật mới hệ thống)
}


def load_index(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def encode_query(query: str, model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError("Cài đặt: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    return model.encode([query])[0]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    max_v = np.max(scores)
    if max_v > 0:
        return scores / max_v
    return scores


<<<<<<< HEAD
# def _descriptor_supported_by_reviews(descriptor_tokens: list[str], review_texts: list[str]) -> bool:
#     """Check whether at least one descriptor token appears in top review content."""
#     if not descriptor_tokens:
#         return True

#     descriptor_set: set[str] = set()
#     for t in descriptor_tokens:
#         low = (t or "").lower().strip()
#         if not low:
#             continue
#         descriptor_set.add(low)
#         if "_" in low:
#             descriptor_set.update(p for p in low.split("_") if p)

#     if not descriptor_set:
#         return True

#     for text in review_texts:
#         norm = normalize_text(text or "")
#         if not norm:
#             continue
#         tokens: set[str] = set()
#         for tok in tokenize_vi(norm):
#             low = tok.lower()
#             tokens.add(low)
#             if "_" in low:
#                 tokens.update(p for p in low.split("_") if p)
#         if descriptor_set.intersection(tokens):
#             return True
#     return False

def _descriptor_supported_by_reviews(descriptor_tokens: list[str], review_texts: list[str]) -> bool:
    """Check whether descriptors match, robust to NLP tokens with underscores and view synonyms."""
    if not descriptor_tokens:
        return True

    # 1. Lọc từ rác & XỬ LÝ DẤU GẠCH DƯỚI TỪ BỘ NLP
    stop_words = {
        "khách", "sạn", "khách sạn", "phòng", "rất", "có", "là", "và", 
        "những", "cho", "tại", "ở", "của", "resort", "homestay", "villa", 
        "khu nghỉ dưỡng", "chỗ nghỉ", "motel", "hostel", "boutique"
    }
    important_descs = []
    
    for t in descriptor_tokens:
        clean_t = t.replace("_", " ").lower().strip()
        if clean_t and clean_t not in stop_words:
            important_descs.append(clean_t)
            
    if not important_descs:
        return True

    # 2. Gom nhóm từ đồng nghĩa chỉ tầm nhìn
    view_keywords = {"view", "hướng", "nhìn"}
    has_view_req = any(vk in important_descs for vk in view_keywords)
    targets = [w for w in important_descs if w not in view_keywords]

    # 3. Quét từng bài review
    for text in review_texts:
        text_lower = text.lower()
        
        # NẾU LÀ TRUY VẤN TÌM VIEW (VD: "hướng núi", "view biển")
        if has_view_req and targets:
            valid_view = False
            for target in targets:
                phrases = [
                    f"view {target}", 
                    f"hướng {target}", 
                    f"nhìn ra {target}",
                    f"nhìn {target}",
                    f"thấy {target}", 
                    f"{target} view"
                ]
                if any(p in text_lower for p in phrases):
                    valid_view = True
                    break
            
            other_targets_exist = all(t in text_lower for t in targets)
            if valid_view and other_targets_exist:
                return True
                
        # NẾU LÀ TRUY VẤN BÌNH THƯỜNG (Giá rẻ, Trung tâm, Bể bơi...)
        else:
            # FIX Ở ĐÂY: Dùng từ điển đồng nghĩa để không đánh rớt kết quả oan uổng
            match_all_descs = True
            for desc in important_descs:
                # Lấy danh sách từ đồng nghĩa của từ khóa hiện tại
                synonyms = [desc]
                for cat, hints in _CATEGORY_TEXT_HINTS.items():
                    if desc in hints:
                        synonyms = hints # Ví dụ: biến 'bình dân' thành ('giá rẻ', 'bình dân',...)
                        break
                
                # CHỈ CẦN khớp 1 từ đồng nghĩa là ĐẠT
                if not any(syn in text_lower for syn in synonyms):
                    match_all_descs = False
                    break 
                    
            if match_all_descs:
                return True
                
    return False

=======
def _descriptor_supported_by_reviews(descriptor_tokens: list[str], review_texts: list[str]) -> bool:
    """Check whether at least one descriptor token appears in top review content."""
    if not descriptor_tokens:
        return True

    descriptor_set: set[str] = set()
    for t in descriptor_tokens:
        low = (t or "").lower().strip()
        if not low:
            continue
        descriptor_set.add(low)
        if "_" in low:
            descriptor_set.update(p for p in low.split("_") if p)

    if not descriptor_set:
        return True

    for text in review_texts:
        norm = normalize_text(text or "")
        if not norm:
            continue
        tokens: set[str] = set()
        for tok in tokenize_vi(norm):
            low = tok.lower()
            tokens.add(low)
            if "_" in low:
                tokens.update(p for p in low.split("_") if p)
        if descriptor_set.intersection(tokens):
            return True
    return False


>>>>>>> 49a78af (cập nhật mới hệ thống)
def _infer_doc_categories(doc: dict) -> set[str]:
    tags = doc.get("category_tags")
    if isinstance(tags, list) and tags:
        return {str(t).strip().lower() for t in tags if str(t).strip()}

    text = " ".join([
        str(doc.get("hotel_name", "")),
        str(doc.get("location", "")),
        str(doc.get("review_text", "")),
    ]).lower()

    inferred: set[str] = set()
    for cat, hints in _CATEGORY_TEXT_HINTS.items():
        if any(h in text for h in hints):
            inferred.add(cat)
    return inferred


def _build_candidate_mask(qu: QueryUnderstandingResult, docs: list[dict]) -> np.ndarray:
    if not docs:
        return np.array([], dtype=bool)

    need_loc = bool(qu.detected_location)
    need_cat = bool(qu.detected_categories)
    if not need_loc and not need_cat:
        return np.ones(len(docs), dtype=bool)

    mask = np.zeros(len(docs), dtype=bool)
    query_cats = {c.lower() for c in qu.detected_categories}

    for i, doc in enumerate(docs):
        loc_ok = True
        if need_loc:
            loc_ok = location_matched(qu.detected_location, str(doc.get("location", "")))

        cat_ok = True
        if need_cat:
            cat_ok = bool(_infer_doc_categories(doc).intersection(query_cats))

        if loc_ok and cat_ok:
            mask[i] = True

    # Fallback: avoid over-filtering when query/category mapping is imperfect.
    if not mask.any():
        return np.ones(len(docs), dtype=bool)
    return mask


def _top_positive_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=int)
    positive = np.where(scores > 0)[0]
    if len(positive) == 0:
        return np.array([], dtype=int)
    if limit <= 0 or len(positive) <= limit:
        ranked = positive[np.argsort(scores[positive])[::-1]]
        return ranked

    part = np.argpartition(scores[positive], -limit)[-limit:]
    top = positive[part]
    ranked = top[np.argsort(scores[top])[::-1]]
    return ranked


def search_hybrid(
    query: str,
    index_dir: Path,
    stopwords_path: Path,
    top_k: int = 10,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    location_boost_factor: float = 1.8,
    descriptor_mismatch_penalty: float = 0.65,
    strict_descriptor_filter: bool = True,
    review_pool_size: int = 1500,
) -> tuple[list[dict], QueryUnderstandingResult]:
    
    # PHẦN 4: Query Understanding
    qu: QueryUnderstandingResult = understand_query(query, stopwords_path=stopwords_path)

    bm25_path = index_dir / "bm25_index.pkl"
    vec_path = index_dir / "vector_index.pkl"

    if not bm25_path.exists() or not vec_path.exists():
        raise FileNotFoundError("Thiếu index. Hãy chạy python indexing/build_bm25_index.py và python indexing/build_vector_index.py")

    bm25_payload = load_index(bm25_path)
    vec_payload = load_index(vec_path)

    bm25_model = bm25_payload["bm25"]
    bm25_docs = bm25_payload["documents"]
    
    embeddings = vec_payload["embeddings"]
    vec_docs = vec_payload["documents"]
    model_name = vec_payload["model_name"]

    bm25_mask = _build_candidate_mask(qu, bm25_docs)
    vec_mask = _build_candidate_mask(qu, vec_docs)

    # 1. BM25 Retrieval
    search_tokens = qu.expanded_tokens if qu.expanded_tokens else qu.core_tokens
    bm25_scores = bm25_model.get_scores(search_tokens) if search_tokens else np.zeros(len(bm25_docs))
    if len(bm25_mask) == len(bm25_scores):
        bm25_scores = bm25_scores * bm25_mask.astype(float)
    bm25_scores_norm = _normalize_scores(bm25_scores)

    # 2. Vector Retrieval
    q_vec = encode_query(qu.raw_query, model_name)
    if len(vec_mask) == len(embeddings):
        masked_idx = np.where(vec_mask)[0]
        if len(masked_idx) > 0:
            vector_scores = np.zeros(len(embeddings), dtype=np.float32)
            vector_scores[masked_idx] = np.dot(embeddings[masked_idx], q_vec.T)
        else:
            vector_scores = np.zeros(len(embeddings), dtype=np.float32)
    else:
        vector_scores = np.dot(embeddings, q_vec.T)
    vector_scores_norm = _normalize_scores(vector_scores)

    # 3. Hybrid Scoring cấp độ Review (PHẦN 5)
    # Lưu ý: vec_docs và bm25_docs có thứ tự giống nhau vì cùng query từ DB theo cùng logic
    # Tuy nhiên để an toàn, ta dùng hashmap review_id
    
    review_scores = {}
    
    bm25_top_idx = _top_positive_indices(bm25_scores_norm, review_pool_size)
    vec_top_idx = _top_positive_indices(vector_scores_norm, review_pool_size)

    # Push BM25
    for i in bm25_top_idx:
        doc = bm25_docs[i]
        if bm25_scores_norm[i] > 0:
            rid = doc["review_id"]
            review_scores[rid] = {
                "doc": doc,
                "bm25_score": float(bm25_scores_norm[i]),
                "vector_score": 0.0,
            }
            
    # Push Vector
    for i in vec_top_idx:
        doc = vec_docs[i]
        if vector_scores_norm[i] > 0:
            rid = doc["review_id"]
            if rid not in review_scores:
                review_scores[rid] = {
                    "doc": doc,
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                }
            review_scores[rid]["vector_score"] = float(vector_scores_norm[i])
            
    # Tính Final Review Score
    valid_reviews = []
    for data in review_scores.values():
        data["hybrid_score"] = vector_weight * data["vector_score"] + bm25_weight * data["bm25_score"]
        if data["hybrid_score"] > 0:
            valid_reviews.append(data)

    # PHẦN 6: Review Aggregation (Group theo Hotel)
    hotel_groups = defaultdict(list)
    for data in valid_reviews:
        hid = data["doc"]["source_hotel_id"]
        hotel_groups[hid].append(data)
        
    results = []
    
    for hid, r_list in hotel_groups.items():
        # Sort reviews trong cùng 1 khách sạn theo hybrid_score giảm dần
        r_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Lấy top 5 review tốt nhất cho mỗi khách sạn để tính sum (tránh bias review rác)
        top_reviews = r_list[:5]
        
        # hotel_score = sum(top_k_review_scores)
        hotel_score = sum(r["hybrid_score"] for r in top_reviews)
        
        # Trích metadata từ khách sạn (lấy từ review đầu tiên)
        first_doc = top_reviews[0]["doc"]
        
        # Truy xuất top điểm phụ để UI hiển thị (max của khách sạn đó)
        max_v_score = max(r["vector_score"] for r in top_reviews)
        max_b_score = max(r["bm25_score"] for r in top_reviews)

        top_review_texts = [r["doc"].get("review_text", "") for r in top_reviews]
<<<<<<< HEAD
        # descriptor_supported = _descriptor_supported_by_reviews(qu.descriptor_tokens, top_review_texts)

        # # If query includes descriptors (e.g. "view núi"), drop hotels that only match by name.
        # if qu.descriptor_tokens and strict_descriptor_filter and not descriptor_supported:
        #     continue

        # if qu.descriptor_tokens and not descriptor_supported:
        #     hotel_score *= descriptor_mismatch_penalty

        # LỌC DESCRIPTOR
        tokens_to_check = qu.descriptor_tokens
        descriptor_supported = True
        if tokens_to_check:
            descriptor_supported = _descriptor_supported_by_reviews(tokens_to_check, top_review_texts)
            if strict_descriptor_filter and not descriptor_supported:
                continue

# PHẦN 7:POI Enforcer
        raw_q_lower = qu.raw_query.lower()
        
        # 1. Tìm xem người dùng đang muốn nhóm danh mục nào (center, pool, beach...)
        target_categories = {} 
        for cat, hints in _CATEGORY_TEXT_HINTS.items():
            for hint in hints:
                if hint in raw_q_lower:
                    if cat not in target_categories:
                        target_categories[cat] = []
                    target_categories[cat].append(hint)

        if target_categories:
            is_valid_poi = False
            has_dealbreaker = False
            
            for cat, user_hints in target_categories.items():
                valid_synonyms = _CATEGORY_TEXT_HINTS[cat]
                
                # LẬP DANH SÁCH ĐEN (Từ khóa phủ định)
                negative_patterns = []
                if cat == "center":
                    negative_patterns = ["xa trung tâm", "cách xa trung tâm", "xa chợ", "ngoại ô", "cách xa", "hơi xa", "xa thành phố", "bất tiện","xa phố cổ","không gần phố cổ"]
                elif cat == "beach":
                    negative_patterns = ["xa biển", "không gần biển", "cách xa biển", "không có bãi biển"]
                elif cat == "budget":
                    negative_patterns = ["đắt đỏ", "giá cao", "mắc", "giá chát", "giá cắt cổ"]
                cat_matched = False
                
                # Quét qua top reviews
                for text in top_review_texts:
                    text_lower = text.lower()
                    
                    if any(neg in text_lower for neg in negative_patterns):
                        has_dealbreaker = True
                        break
                        
                    # Nếu review khen đúng các từ đồng nghĩa
                    if any(syn in text_lower for syn in valid_synonyms):
                        cat_matched = True
                    
                if cat_matched:
                    is_valid_poi = True

            # 2. RA QUYẾT ĐỊNH (PHẠT / THƯỞNG)
            if has_dealbreaker:
                hotel_score *= 0.001 
            elif is_valid_poi:
                hotel_score *= 2.0
            else:
                hotel_score *= 0.1 

        # PHẦN 8: Location Boosting
        loc_matched = False
        if qu.detected_location and location_matched(qu.detected_location, first_doc.get("location", "")):
            hotel_score *= location_boost_factor
            loc_matched = True
       
       # PHẦN 9: Phân loại hình lưu trú (Accommodation Type Boosting)
        raw_q_lower = qu.raw_query.lower()
        hotel_name_lower = str(first_doc.get("hotel_name", "")).lower()
        
        # 1. Định nghĩa các nhóm loại hình
        acc_types = {
            "resort": ["resort", "khu nghỉ dưỡng", "retreat"],
            "homestay": ["homestay", "lodge", "cabin", "nhà dân"],
            "villa": ["villa", "biệt thự"],
            "hotel": ["hotel", "khách sạn", "boutique"]
        }
        
        # 2. Nhận diện xem người dùng đang muốn tìm loại hình nào
        target_type = None
        for atype, keywords in acc_types.items():
            if any(k in raw_q_lower for k in keywords):
                target_type = atype
                break # Ưu tiên loại hình được nhắc đến đầu tiên
                
        # 3. Đối chiếu và Thưởng/Phạt
        if target_type:
            # Kiểm tra xem tên nơi lưu trú có chứa từ khóa của loại hình đó không
            is_correct_type = any(k in hotel_name_lower for k in acc_types[target_type])
            
            if is_correct_type:
                hotel_score *= 1.5  # THƯỞNG 50% ĐIỂM: Tìm resort ra đúng chữ Resort trong tên
            else:
                # Kiểm tra xem nó có rành rành là một loại hình KHÁC không?
                # VD: Tìm "Resort" nhưng tên lại là "Núi Homestay" -> Loại
                is_conflicting_type = False
                for other_type, other_keywords in acc_types.items():
                    if other_type != target_type:
                        if any(k in hotel_name_lower for k in other_keywords):
                            is_conflicting_type = True
                            break
                
                if is_conflicting_type:
                    hotel_score *= 0.2  # PHẠT NẶNG (Mất 80% điểm): Sai loại hình hoàn toàn
                else:
                    hotel_score *= 0.9

        results.append({
            "source": first_doc.get("source", ""),
            "source_hotel_id": hid,
            "hotel_name": first_doc.get("hotel_name", ""),
            "location": first_doc.get("location", ""),
            "rating": first_doc.get("rating", ""),
            "review_count": len(r_list),  # số review MATCHED với query
            "hybrid_score": float(hotel_score),
            "vector_score": float(max_v_score), # Truyền điểm max vector cho UI
            "bm25_score": float(max_b_score),   # Truyền điểm max bm25 cho UI
            "location_matched": loc_matched,
            "descriptor_matched": descriptor_supported,
            "top_reviews": top_review_texts[:3],
        })

=======
        descriptor_supported = _descriptor_supported_by_reviews(qu.descriptor_tokens, top_review_texts)

        # If query includes descriptors (e.g. "view núi"), drop hotels that only match by name.
        if qu.descriptor_tokens and strict_descriptor_filter and not descriptor_supported:
            continue

        if qu.descriptor_tokens and not descriptor_supported:
            hotel_score *= descriptor_mismatch_penalty

        # PHẦN 7: Location Boosting
        loc_matched = False
        if qu.detected_location and location_matched(qu.detected_location, first_doc.get("location", "")):
            hotel_score *= location_boost_factor
            loc_matched = True

        results.append({
            "source": first_doc.get("source", ""),
            "source_hotel_id": hid,
            "hotel_name": first_doc.get("hotel_name", ""),
            "location": first_doc.get("location", ""),
            "rating": first_doc.get("rating", ""),
            "review_count": len(r_list),  # số review MATCHED với query
            "hybrid_score": float(hotel_score),
            "vector_score": float(max_v_score), # Truyền điểm max vector cho UI
            "bm25_score": float(max_b_score),   # Truyền điểm max bm25 cho UI
            "location_matched": loc_matched,
            "descriptor_matched": descriptor_supported,
            "top_reviews": top_review_texts[:3],
        })

>>>>>>> 49a78af (cập nhật mới hệ thống)
    # Sort top K Hotel
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k], qu
