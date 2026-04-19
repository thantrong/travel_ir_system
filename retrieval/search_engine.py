"""
Search Engine (Review-Level Indexing -> Hotel-level Aggregation)
PHẦN 6: REVIEW AGGREGATION
PHẦN 7: LOCATION BOOSTING
PHẦN 10: RANKING PIPELINE
"""

import pickle
import yaml
from pathlib import Path
from collections import defaultdict
from functools import lru_cache

import numpy as np

from retrieval.query_understanding import understand_query, location_matched, QueryUnderstandingResult
from retrieval.query_tag_extractor import extract_query_tags
from summarization import summarize_reviews_tfidf
from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi


_CATEGORY_TEXT_HINTS: dict[str, tuple[str, ...]] = {
    "mountain": ("núi", "đồi", "thung lũng", "cao nguyên", "săn mây", "view núi", "hướng núi"),
    "beach": ("biển", "bãi biển", "ven biển", "gần biển", "đảo", "view biển"),
    "family": ("gia đình", "trẻ em", "family", "kid"),
    "budget": ("giá rẻ", "bình dân", "tiết kiệm", "hợp lý"),
    "luxury": ("sang trọng", "cao cấp", "luxury", "5 sao", "resort", "villa"),
    "center": ("trung tâm", "phố đi bộ", "chợ đêm", "gần chợ","phố cổ","hồ gươm", "hồ tây"),
    "amenity_pool": ("hồ bơi", "bể bơi", "pool", "vô cực"),
    "amenity_breakfast": ("ăn sáng", "buffet", "điểm tâm"),
    "airport": ("sân bay", "tân sơn nhất", "nội bài", "ga"),
    "spa_gym": ("gym", "spa", "massage", "thể hình", "xông hơi", "bồn tắm", "jacuzzi"),
    "kitchen": ("bếp", "nấu ăn", "bbq", "nướng"),
    "photo": ("sống ảo", "chụp hình", "vintage", "đẹp", "decor"),
    "quiet": ("yên tĩnh", "nghỉ dưỡng", "đọc sách", "người lớn tuổi"),
    "pet": ("thú cưng", "chó", "mèo", "pet")
}

_NEGATIVE_DESCRIPTOR_HINTS: dict[str, set[str]] = {
    "cleanliness": {"sạch", "sạch_sẽ", "vệ_sinh"},
    "breakfast": {"ăn_sáng", "bữa_sáng", "buffet", "điểm_tâm"},
    "food_quality": {"đồ_ăn", "ăn_ngon", "ẩm_thực", "nhà_hàng"},
    "pool": {"hồ_bơi", "bể_bơi", "pool"},
    "bathroom": {"phòng_tắm", "nhà_tắm", "toilet", "wc"},
    "staff_friendly": {"nhân_viên", "phục_vụ", "lễ_tân"},
    "room_spacious": {"phòng_rộng", "rộng_rãi", "giường_rộng"},
    "view_beach": {"view_biển", "hướng_biển"},
    "view_mountain": {"view_núi", "hướng_núi"},
    "near_center": {"trung_tâm", "gần_trung_tâm"},
    "near_beach": {"gần_biển", "ven_biển"},
    "quiet_room": {"yên_tĩnh", "ít_ồn", "cách_âm"},
    "convenience": {"tiện_nghi", "tiện_lợi", "thuận_tiện"},
    "good_service": {"dịch_vụ", "phục_vụ"},
    "ambiance": {"đẹp", "decor", "hiện_đại", "sống_ảo", "chụp_hình"},
}

def _query_forms(tokens: list[str]) -> set[str]:
    out: set[str] = set()
    for tok in tokens:
        low = str(tok).lower().strip()
        if not low:
            continue
        out.add(low)
        if "_" in low:
            out.update(p for p in low.split("_") if p)
    return out

def _required_negative_tags(qu: QueryUnderstandingResult) -> set[str]:
    if not qu.descriptor_tokens:
        return set()
    forms = _query_forms(qu.descriptor_tokens)
    required: set[str] = set()
    for tag, hints in _NEGATIVE_DESCRIPTOR_HINTS.items():
        if forms.intersection(hints):
            required.add(tag)
    return required

def _filter_negative_reviews(reviews: list[dict], required_tags: set[str]) -> list[dict]:
    if not required_tags:
        return reviews
    neg_tags = {f"!{t}" for t in required_tags}
    filtered: list[dict] = []
    for data in reviews:
        doc = data.get("doc", {})
        tags = list(doc.get("descriptor_tags", []) or []) + list(doc.get("category_tags", []) or [])
        tag_set = {str(t).strip().lower() for t in tags if str(t).strip()}
        if tag_set.intersection(neg_tags):
            continue
        filtered.append(data)
    return filtered

# Đường dẫn tới file config synonyms
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_SYNONYMS_PATH = _CONFIG_DIR / "query_synonyms.yaml"

# Cache cho conflict_keys loaded từ YAML
_conflict_keys_cache: dict[str, set[str]] | None = None


def _load_conflict_keys_from_yaml() -> dict[str, set[str]]:
    """Load conflict_keys (ý định query → từ khóa liên quan) từ file YAML synonyms."""
    global _conflict_keys_cache
    if _conflict_keys_cache is not None:
        return _conflict_keys_cache

    default_conflict_keys: dict[str, set[str]] = {
        "cleanliness": {"sạch", "sạch sẽ", "vệ sinh"},
        "near_beach": {"biển", "gần biển", "view biển"},
        "pool": {"hồ bơi", "pool", "bể bơi"},
        "quiet_room": {"yên tĩnh", "cách âm", "ít ồn"},
        "near_center": {"trung tâm", "gần trung tâm"},
        "room_spacious": {"rộng", "rộng rãi", "thoáng"},
        "good_service": {"dịch vụ", "phục vụ", "nhân viên"},
        "food_quality": {"ăn sáng", "buffet", "đồ ăn"},
        "value": {"giá rẻ", "bình dân", "rẻ", "hợp lý"},
        "luxury": {"sang trọng", "cao cấp", "luxury", "5 sao", "đẳng cấp"},
        "family": {"gia đình", "trẻ em", "family", "kid"},
        "couple": {"lãng mạn", "cặp đôi", "honeymoon"},
        "pet": {"thú cưng", "chó", "mèo", "pet"},
        "view": {"view", "nhìn", "hướng", "cảnh", "ban công"},
        "airport": {"sân bay", "đưa đón", "gần sân bay"},
        "spa_gym": {"gym", "spa", "massage", "thể hình", "xông hơi", "fitness"},
        "kitchen": {"bếp", "nấu ăn", "bbq", "nướng", "bếp riêng"},
        "business": {"công tác", "business", "phòng họp", "meeting"},
    }

    yaml_path = _SYNONYMS_PATH
    if yaml_path.exists():
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            synonyms = data.get("synonyms", {})
            if synonyms:
                synonym_to_category = {
                    "sạch sẽ": "cleanliness", "sạch": "cleanliness",
                    "vệ sinh tốt": "cleanliness", "gọn gàng": "cleanliness",
                    "yên tĩnh": "quiet_room", "ít ồn": "quiet_room",
                    "tĩnh lặng": "quiet_room", "riêng tư": "quiet_room",
                    "hồ bơi": "pool", "bể bơi": "pool", "pool": "pool",
                    "swimming pool": "pool",
                    "giá rẻ": "value", "bình dân": "value", "giá mềm": "value",
                    "giá tốt": "value", "tiết kiệm": "value",
                    "buffet sáng": "food_quality", "ăn sáng buffet": "food_quality",
                    "bữa sáng buffet": "food_quality",
                    "gần trung tâm": "near_center", "ngay trung tâm": "near_center",
                    "sát trung tâm": "near_center", "thuận tiện đi lại": "near_center",
                    "gia đình": "family", "cho trẻ em": "family", "đi cùng con nhỏ": "family",
                    "cặp đôi": "couple", "honeymoon": "couple", "lãng mạn": "couple",
                    "công tác": "business", "business": "business", "đi công tác": "business",
                    "cao cấp": "luxury", "sang trọng": "luxury", "upscale": "luxury",
                    "thân thiện": "good_service", "nhiệt tình": "good_service",
                    "hiếu khách": "good_service", "gần sân bay": "airport",
                    "sát sân bay": "airport",
                    "bãi đậu xe": "convenience", "chỗ đậu xe": "convenience",
                    "parking": "convenience", "đỗ xe": "convenience",
                    "view đẹp": "view", "view biển": "view", "view núi": "view",
                    "cảnh đẹp": "view", "góc nhìn đẹp": "view",
                }
                for key, aliases in synonyms.items():
                    category = synonym_to_category.get(str(key).lower())
                    if category:
                        if category not in default_conflict_keys:
                            default_conflict_keys[category] = set()
                        default_conflict_keys[category].add(str(key).lower().replace("_", " "))
                        for alias in aliases:
                            default_conflict_keys[category].add(str(alias).lower().replace("_", " "))
        except Exception:
            pass

    result = {k: v if isinstance(v, set) else set(v) for k, v in default_conflict_keys.items()}
    _conflict_keys_cache = result
    return result


def _load_index_uncached(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def _load_index_cached(path_str: str, mtime_ns: int) -> dict:
    return _load_index_uncached(Path(path_str))


def load_index(path: Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _load_index_cached(str(resolved), int(stat.st_mtime_ns))


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError("Cài đặt: pip install sentence-transformers")
    return SentenceTransformer(model_name)


def encode_query(query: str, model_name: str) -> np.ndarray:
    model = _load_sentence_model(model_name)
    vec = model.encode([query])[0]
    return np.asarray(vec, dtype=np.float32)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    max_v = np.max(scores)
    if max_v > 0:
        return scores / max_v
    return scores


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
            # Dùng từ điển đồng nghĩa để không đánh rớt kết quả oan uổng
            match_all_descs = True
            for desc in important_descs:
                synonyms = [desc]
                for cat, hints in _CATEGORY_TEXT_HINTS.items():
                    if desc in hints:
                        synonyms = hints
                        break
                
                # CHỈ CẦN khớp 1 từ đồng nghĩa là ĐẠT
                if not any(syn in text_lower for syn in synonyms):
                    match_all_descs = False
                    break 
                    
            if match_all_descs:
                return True
                
    return False


def _sentiment_penalty_factor(query: str, review_texts: list[str], query_descriptors: list[str], query_categories: list[str]) -> float:
    """Compute a soft penalty when negative descriptors in reviews conflict with the query intent."""
    q = (query or "").lower()
    review_blob = " ".join(review_texts).lower()

    strong_negative = ["rất tệ", "quá tệ", "tồi tệ", "thất vọng", "không tốt", "không hài lòng", "bẩn", "dơ", "cũ", "ồn", "mùi hôi", "khó chịu", "không sạch", "không yên tĩnh", "không đáng tiền"]
    mild_negative = ["hơi", "hơi nhỏ", "hơi cũ", "hơi ồn", "hơi bụi", "chưa", "chưa tốt", "không được tốt", "không ổn"]

    negative_hits = sum(1 for x in strong_negative if x in review_blob)
    mild_hits = sum(1 for x in mild_negative if x in review_blob)

    # Nếu query có ý định đặc tả tích cực mà review đi ngược lại, phạt mạnh hơn.
    q_forms = _query_forms(query_descriptors + query_categories)
    conflict_keys = _load_conflict_keys_from_yaml()
    conflict_found = any(q_forms.intersection(keys) for keys in conflict_keys.values())

    if negative_hits >= 2:
        return 0.6 if conflict_found else 0.7
    if negative_hits == 1 and mild_hits >= 1:
        return 0.75 if conflict_found else 0.8
    if negative_hits == 1:
        return 0.8 if conflict_found else 0.85
    return 1.0


def _infer_doc_categories(doc: dict) -> set[str]:
    tags = doc.get("category_tags")
    if isinstance(tags, list) and tags:
        cleaned = {str(t).strip().lower() for t in tags if str(t).strip() and not str(t).strip().startswith("!")}
        if cleaned:
            return cleaned

    # Fallback cũ cho dữ liệu legacy chưa backfill tag.
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


def _infer_doc_accommodation_types(doc: dict) -> set[str]:
    """Infer loại hình lưu trú từ field đã crawl/gán nhãn, fallback theo tên."""
    candidate_fields = [
        doc.get("types"),
        doc.get("hotel_types"),
    ]
    out: set[str] = set()
    for value in candidate_fields:
        if isinstance(value, list):
            out.update(str(v).strip().lower() for v in value if str(v).strip())
        elif isinstance(value, str) and value.strip():
            out.add(value.strip().lower())

    if out:
        return out

    text = " ".join([
        str(doc.get("hotel_name", "")),
        str(doc.get("review_text", "")),
    ]).lower()
    for t in ("resort", "homestay", "villa", "hostel", "guesthouse", "aparthotel", "boutique", "hotel"):
        if t in text:
            out.add(t if t != "boutique" else "boutique_hotel")
    return out


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
    types_boot : float = 1.2,
    strict_descriptor_filter: bool = False,
    review_pool_size: int = 9000,
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
    bm25_ids = bm25_payload.get("review_ids", [str(d.get("review_id", "")) for d in bm25_docs])
    bm25_id_to_idx = bm25_payload.get("review_id_to_idx", {})
    
    embeddings = vec_payload["embeddings"]
    vec_docs = vec_payload["documents"]
    vec_ids = vec_payload.get("review_ids", [str(d.get("_id", d.get("review_id", ""))) for d in vec_docs])
    vec_id_to_idx = vec_payload.get("review_id_to_idx", {})
    model_name = vec_payload["model_name"]

    bm25_mask = _build_candidate_mask(qu, bm25_docs)
    vec_mask = _build_candidate_mask(qu, vec_docs)

    # 1. BM25 Retrieval
    search_tokens = qu.expanded_tokens if qu.expanded_tokens else qu.core_tokens
    bm25_scores = (
        bm25_model.get_scores(search_tokens)
        if (bm25_model is not None and search_tokens)
        else np.zeros(len(bm25_docs))
    )
    if len(bm25_mask) == len(bm25_scores):
        bm25_scores = bm25_scores * bm25_mask.astype(float)
    bm25_scores_norm = _normalize_scores(bm25_scores)

    # 2. Vector Retrieval
    vector_scores = np.zeros(len(vec_docs), dtype=np.float32)
    vector_fallback_to_bm25 = False
    if len(vec_docs) > 0 and len(embeddings) > 0:
        try:
            q_vec = encode_query(qu.raw_query, model_name)
            if len(vec_mask) == len(embeddings):
                masked_idx = np.where(vec_mask)[0]
                if len(masked_idx) > 0:
                    vector_scores[masked_idx] = np.dot(embeddings[masked_idx], q_vec.T)
            else:
                vector_scores = np.dot(embeddings, q_vec.T)
        except Exception as exc:
            vector_fallback_to_bm25 = True
            print(f"[WARN] Vector encode failed, fallback to BM25-aligned scores: {exc}")
            for i, doc in enumerate(vec_docs):
                rid = str(doc.get("_id", doc.get("review_id", vec_ids[i] if i < len(vec_ids) else ""))).strip()
                if not rid:
                    continue
                bm_idx = bm25_id_to_idx.get(rid)
                if bm_idx is None or bm_idx >= len(bm25_scores_norm):
                    continue
                vector_scores[i] = float(bm25_scores_norm[bm_idx])
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
            rid = str(doc.get("review_id", doc.get("_id", bm25_ids[i] if i < len(bm25_ids) else ""))).strip()
            if not rid:
                continue
            review_scores[rid] = {
                "doc": doc,
                "bm25_score": float(bm25_scores_norm[i]),
                "vector_score": 0.0,
            }
            
    # Push Vector
    for i in vec_top_idx:
        doc = vec_docs[i]
        if vector_scores_norm[i] > 0:
            rid = str(doc.get("_id", doc.get("review_id", vec_ids[i] if i < len(vec_ids) else ""))).strip()
            if not rid:
                continue
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

    # Tạm thời vô hiệu hóa bộ lọc flashtext/negative-tag filtering để tránh mất thông tin
    # required_negative_tags = _required_negative_tags(qu)
    # if required_negative_tags:
    #     valid_reviews = _filter_negative_reviews(valid_reviews, required_negative_tags)

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

        # LỌC DESCRIPTOR -> giờ là soft signal, không hard-filter mặc định
        tokens_to_check = qu.descriptor_tokens
        descriptor_supported = True
        if tokens_to_check:
            descriptor_supported = _descriptor_supported_by_reviews(tokens_to_check, top_review_texts)
            if strict_descriptor_filter and not descriptor_supported:
                continue

        # PHẦN 7: POI / category boosting - soft boost theo tags và query intent
        query_categories = {c.lower() for c in qu.detected_categories}
        doc_categories = set()
        for rr in top_reviews:
            doc_categories.update({str(t).lower() for t in rr["doc"].get("category_tags", []) if str(t).strip() and not str(t).strip().startswith("!")})
        if not doc_categories:
            doc_categories = _infer_doc_categories(first_doc)

        # PHẦN 8: Location Boosting
        loc_matched = False
        if qu.detected_location and location_matched(qu.detected_location, first_doc.get("location", "")):
            hotel_score *= location_boost_factor
            loc_matched = True

        # PHẦN 8.5: Sentiment penalty - phạt review chê mạnh nếu xung đột intent query
        sentiment_factor = _sentiment_penalty_factor(
            qu.raw_query,
            top_review_texts,
            qu.descriptor_tokens,
            list(query_categories),
        )
        if sentiment_factor != 1.0:
            hotel_score *= sentiment_factor

        # PHẦN 9: Phân loại hình lưu trú (Accommodation Type Boosting)
        raw_q_lower = qu.raw_query.lower()
        acc_types = {
            "resort": ["resort", "khu nghỉ dưỡng", "retreat"],
            "homestay": ["homestay", "lodge", "cabin", "nhà dân", "farmstay", "glamping", "camping", "eco", "sinh_thái"],
            "villa": ["villa", "biệt thự", "bungalow"],
            "hostel": ["hostel"],
            "guesthouse": ["guesthouse", "nhà khách", "nhà nghỉ"],
            "aparthotel": ["aparthotel", "apartment hotel", "căn_hộ", "căn hộ dịch vụ"],
            "boutique_hotel": ["boutique", "boutique hotel"],
            "hotel": ["hotel", "khách sạn"],
        }
        # Boost điểm dựa trên loại hình lưu trú nếu query có gợi ý rõ ràng
        # và khách sạn đó khớp loại hình được yêu cầu trong query
        hotel_acc_types = _infer_doc_accommodation_types(first_doc)
        type_boost_applied = False
        type_mismatch_penalty = False
        type_query_keyword = ""
        
        for acc_type, keywords in acc_types.items():
            if any(k in raw_q_lower for k in keywords):
                type_query_keyword = acc_type
                if acc_type in hotel_acc_types:
                    hotel_score *= types_boot
                    type_boost_applied = True
                else:
                    # Phạt nhẹ hơn khi lệch loại hình, tránh làm rơi candidate đúng nội dung
                    hotel_score *= 0.85
                    type_mismatch_penalty = True
                break

        # Debug info: tracking score qua từng bước boost
        score_after_aggregation = sum(r["hybrid_score"] for r in top_reviews)
        score_after_location = score_after_aggregation
        score_after_type = score_after_aggregation
        
        # Reverse location boost
        if qu.detected_location and loc_matched:
            score_after_location = score_after_aggregation / location_boost_factor
        else:
            score_after_location = score_after_aggregation
            
        # Reverse type boost
        type_boost_keywords_found = [type_query_keyword] if type_query_keyword else []
        if type_query_keyword:
            if type_boost_applied:
                score_after_type = score_after_location / types_boot
            elif type_mismatch_penalty:
                score_after_type = score_after_location / 0.85
            else:
                score_after_type = score_after_location
        else:
            score_after_type = score_after_location

        debug_info = {
            "hotel_id": hid,
            "hotel_name": first_doc.get("hotel_name", ""),
            "review_count": len(r_list),
            "score_after_aggregation": round(score_after_aggregation, 4),
            "score_after_location_boost": round(score_after_location, 4),
            "score_final": round(hotel_score, 4),
            "hotel_acc_types": list(hotel_acc_types),
            "query_detected_location": qu.detected_location,
            "query_categories_detected": list(query_categories),
            "doc_categories": list(doc_categories),
            "location_matched": loc_matched,
            "descriptor_filtered": strict_descriptor_filter and not descriptor_supported,
            "descriptor_supported": descriptor_supported,
            "sentiment_factor": round(sentiment_factor, 4),
            "descriptor_tokens_used": tokens_to_check,
            "type_boost_applied": type_boost_applied,
            "type_mismatch_penalty": type_mismatch_penalty,
            "type_boost_keywords_found": type_boost_keywords_found,
            "types_boot_factor": types_boot,
            "hotel_type_matched": type_query_keyword in hotel_acc_types if type_query_keyword else False,
            "vector_fallback_to_bm25": vector_fallback_to_bm25,
        }

        results.append({
            "source": first_doc.get("source", ""),
            "source_hotel_id": hid,
            "hotel_name": first_doc.get("hotel_name", first_doc.get("source_hotel_id", "")),
            "location": first_doc.get("location", ""),
            "rating": first_doc.get("rating", ""),
            "review_count": len(r_list),
            "hybrid_score": float(hotel_score),
            "vector_score": float(max_v_score),
            "bm25_score": float(max_b_score),
            "location_matched": loc_matched,
            "descriptor_matched": descriptor_supported,
            "top_reviews": top_review_texts[:3],
            "debug_info": debug_info,
        })
    # Sort top K Hotel
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # Chữa cháy nhanh: nếu query có location rõ ràng thì chỉ giữ các kết quả đúng vị trí
    # trong top-K cuối cùng để tránh boost kéo lạc tỉnh / lạc vùng.
    if qu.detected_location:
        matched_results = [r for r in results if r.get("location_matched", False)]
        if matched_results:
            results = matched_results

    return results[:top_k], qu
