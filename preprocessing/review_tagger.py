from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml
import torch
from flashtext import KeywordProcessor
from tqdm import tqdm

from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi
from preprocessing.clean_text import clean_review_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULE_FILE = PROJECT_ROOT / "config" / "review_tag_rules.yaml"


@dataclass
class TagResult:
    category_tags: list[str] = field(default_factory=list)
    descriptor_tags: list[str] = field(default_factory=list)
    descriptor_polarity: dict[str, str] = field(default_factory=dict)
    matched_phrases: dict[str, list[str]] = field(default_factory=dict)
    contexts: list[dict] = field(default_factory=list)


def _load_rules() -> dict:
    if not RULE_FILE.exists():
        return {}
    payload = yaml.safe_load(RULE_FILE.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def rules() -> dict:
    return _load_rules()


def _normalize_phrases(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for v in values:
        nv = normalize_text(str(v).replace("_", " "))
        if nv and nv not in seen:
            seen.add(nv)
            out.append(nv)
    return out


def _prepare_text(review_text: str, hotel_name: str = "", location: str = "") -> str:
    parts = [clean_review_text(review_text), hotel_name, location]
    joined = " ".join(p for p in parts if p)
    return normalize_text(joined)


def _match_phrases(text: str, positive: list[str], negative: list[str]) -> tuple[list[str], list[str]]:
    pos_hits: list[str] = []
    neg_hits: list[str] = []
    for phrase in positive:
        if phrase and phrase in text:
            pos_hits.append(phrase)
    for phrase in negative:
        if phrase and phrase in text:
            neg_hits.append(phrase)
    return pos_hits, neg_hits


def _apply_tag_group(text: str, spec: dict[str, dict]) -> tuple[list[str], dict[str, list[str]]]:
    """
    Apply tag group với logic: nếu cả positive và negative cùng match → ưu tiên negative.
    Dùng FlashText để tìm kiếm nhanh.
    """
    global _CATEGORY_KEYWORD_PROCESSOR
    
    if _CATEGORY_KEYWORD_PROCESSOR is None:
        _build_phrase_maps()
    
    # Use FlashText for fast keyword extraction
    # extract_keywords returns list of keywords (the values we stored)
    keywords_found = _CATEGORY_KEYWORD_PROCESSOR.extract_keywords(text)
    
    # Group by tag
    tag_hits: dict[str, dict[str, list[str]]] = {}  # tag -> {"positive": [...], "negative": [...]}
    for tag, polarity in keywords_found:
        tag_hits.setdefault(tag, {"positive": [], "negative": []})
        tag_hits[tag][polarity].append(f"{tag}_{polarity}")
    
    # Apply conflict resolution
    tags: list[str] = []
    matched_map: dict[str, list[str]] = {}
    for tag, hits in tag_hits.items():
        pos_hits = hits["positive"]
        neg_hits = hits["negative"]
        
        if pos_hits and neg_hits:
            neg_tag = f"!{tag}"
            tags.append(neg_tag)
            matched_map[tag] = [f"!{tag}"]
        elif pos_hits:
            tags.append(tag)
            matched_map[tag] = [tag]
        elif neg_hits:
            neg_tag = f"!{tag}"
            tags.append(neg_tag)
            matched_map[tag] = [f"!{tag}"]
    
    return tags, matched_map


# --- PhoBERT Sentiment Integration ---

# Subject mapping: descriptor tag -> subjects phù hợp
SUBJECT_VALIDATION = {
    "room_spacious": ["phòng", "phòng_", "ksan", "khách_sạn", "cửa_sổ", "giường", "diện_tích", "không_gian"],
    "cleanliness": ["phòng", "bathroom", "phòng_tắm", "nhà_vệ_sinh", "khăn", "ga", "nệm", "sàn", "bàn", "ghế", "ban_công", "tủ_lạnh"],
    "ambiance": ["phòng", "ksan", "khách_sạn", "trang_trí", "nội_thất", "view", "cảnh", "thiết_kế", "kiến_trúc", "lobby", "sảnh"],
    "convenience": ["tiện_nghi", "đồ_đạc", "trang_bị", "thiết_bị", "phòng", "dịch_vụ", "di_chuyển", "đi_lại"],
    "staff_friendly": ["nhân_viên", "lễ_tân", "phục_vụ", "bảo_vệ", "cư_dân", "quản_lý", "chủ", "dân", "bạn"],
    "quiet_room": ["phòng", "cách_âm", "tiếng_ồn", "âm_thanh", "yên_tĩnh", "giấc_ngủ", "đêm"],
    "bathroom": ["phòng_tắm", "nhà_vệ_sinh", "toilet", "wc", "vòi_sen", "nước_nóng"],
    "pool": ["hồ_bơi", "bể_bơi", "pool"],
    "breakfast": ["ăn_sáng", "bữa_sáng", "buffet", "đồ_ăn", "món_ăn", "thực_đơn"],
    "food_quality": ["đồ_ăn", "món_ăn", "ẩm_thực", "nhà_hàng", "buffet", "ăn_uống", "menu"],
    "view_beach": ["view", "hướng", "nhìn_ra", "ban_công", "cửa_sổ", "biển"],
    "view_mountain": ["view", "hướng", "nhìn_ra", "ban_công", "cửa_sổ", "núi", "đồi"],
    "good_service": ["dịch_vụ", "phục_vụ", "hỗ_trợ", "đón", "đưa", "thuê_xe", "dọn_phòng"],
    "parking": ["xe", "bãi", "đậu", "đỗ", "gửi_", "giữ_xe", "phí", "cost"],
    "balcony": ["ban_công", "lan_can", "cửa_sổ", "sân"],
    "wifi": ["wifi", "internet", "mạng", "kết_nối"],
    "elevator": ["thang_máy", "thang bộ", "lầu"],
    "hot_water": ["nước_nóng", "bình_nóng", "vòi_sen", "phòng_tắm"],
    "air_conditioning": ["điều_hòa", "máy_lạnh", "phòng", "nhiệt_độ"],
    "minibar": ["tủ_lạnh", "minibar", "nước_uống"],
    "soundproofing": ["cách_âm", "tiếng_ồn", "âm_thanh", "ồn", "yên_tĩnh"],
    "check_in": ["check_in", "checkin", "nhận_phòng", "lễ_tân"],
    "housekeeping": ["dọn_phòng", "phòng", "dọn_dẹp"],
    "security": ["an_ninh", "bảo_vệ", "an_toàn", "trộm"],
    "value_for_money": ["giá", "tiền", "phí", "chi_phí", "đắt", "rẻ"],
    "noise_level": ["tiếng", "ồn", "âm_thanh", "tiếng_ồn", "phòng", "đêm", "hành_lang"],
    "bed_comfort": ["nệm", "giường", "đệm", "gối", "chăn", "ga"],
    "lighting": ["đèn", "ánh_sáng", "sáng", "tối", "phòng"],
    "smell": ["mùi", "thơm", "hôi", "ẩm_mốc", "thuốc_lá", "phòng"],
    # New descriptors
    "jacuzzi": ["bồn_tắm", "jacuzzi", "bồn_sục", "spa", "phòng_tắm"],
    "gym_fitness": ["gym", "phòng_tập", "fitness", "máy_tập", "thể_dục"],
    "kitchen": ["bếp", "nấu_ăn", "bbq", "tiệc_nướng", "dụng_cụ"],
    "laundry": ["giặt_là", "giặt_ủi", "máy_giặt", "sấy"],
    "spa": ["spa", "massage", "xông_hơi", "sauna", "ngâm_chân", "trị_liệu"],
    "room_amenities": ["máy_sấy", "bàn_ủi", "máy_phà", "tivi", "netflix"],
    "sunset_view": ["view", "hướng", "hoàng_hôn", "sunset", "ban_công"],
    "sports": ["thể_thao", "lướt_ván", "kayak", "lặn", "tennis"],
    "english_speaking": ["tiếng_anh", "nói_tiếng_anh", "giao_tiếp", "hướng_dẫn"],
    "cafe": ["cà_phê", "quán", "espresso", "cocktail", "quầy_bar"],
    "connecting_rooms": ["phòng", "thông_nhau", "liền_kề", "gia_đình"],
    "childcare": ["trông_trẻ", "giữ_trẻ", "babysitting", "club"],
    "wheelchair": ["xe_lăn", "lối_đi", "khuyết_tật", "thang_máy"],
    "yoga": ["yoga", "thiền", "chữa_lành", "healing", "chay"],
    "tour_guide": ["hướng_dẫn", "tour", "dẫn_đoàn", "địa_phương"],
    "karaoke": ["karaoke", "hát", "âm_thanh", "dàn"],
    "river_view": ["view", "hướng", "sông", "nhìn_ra"],
    "rice_terrace_view": ["ruộng", "bậc_thang", "view", "nhìn_xuống"],
    "heritage_view": ["kiến_trúc", "cổ", "cung_đình", "phố_cổ", "di_tích"],
    "garden_view": ["vườn", "sân_vườn", "view", "hoa"],
    "city_view": ["view", "thành_phố", "toàn_cảnh", "panorama"],
    "sunrise_view": ["view", "bình_minh", "sunrise", "hướng_đông"],
}

# Descriptor cần subject validation (generic words)
GENERIC_DESCRIPTORS = {"room_spacious", "cleanliness", "ambiance", "convenience", "soundproofing", "air_conditioning"}

# Exclusion phrases - các trường hợp dễ nhầm lẫn
EXCLUSION_PHRASES = {
    "room_spacious": [
        # Từ chỉ người (nhầm với kích thước)
        "cháu nhỏ", "em nhỏ", "con nhỏ", "người nhỏ", "bạn nhỏ", "bé nhỏ",
        "em bé", "bé", "bé yêu", "con bé", "thằng bé", "con nít",
        "trẻ nhỏ", "đứa nhỏ", "nhóc", "bé con",
        "nhỏ tuổi", "nhỏ tuổi hơn", "nhỏ hơn", "nhỏ nhất",
        "nhỏ lòng", "nhỏ nhen", "nhỏ nhẹ", "nhỏ to",
        "nhỏ xíu", "nhỏ tí", "nhỏ tí hon", "nhỏ kia", "nhỏ đó", "nhỏ này",
        # Từ chỉ số lượng
        "ít", "vài", "một ít", "một vài",
    ],
    "cleanliness": [
        "sạch nợ", "sạch trơn",
    ],
}

# Strong negative keywords
STRONG_NEGATIVE_KEYWORDS = {
    "gián", "chuột", "rận", "bọ", "muỗi", "ruồi", "nhện", "mối", "mọt",
    "phân", "nước tiểu", "mùi hôi", "thối", "khủng khiếp", "kinh khủng",
    "ghê tởm", "ghê", "sợ", "ác mộng", "ám ảnh",
}

# Build descriptor phrase map
_DESCRIPTOR_PHRASES = {}
_CATEGORY_PHRASES = {}

# FlashText keyword processors for fast matching
_CATEGORY_KEYWORD_PROCESSOR = None
_DESCRIPTOR_KEYWORD_PROCESSOR = None

def _build_phrase_maps():
    """Build phrase maps from rules using FlashText for fast matching."""
    global _DESCRIPTOR_PHRASES, _CATEGORY_PHRASES
    global _CATEGORY_KEYWORD_PROCESSOR, _DESCRIPTOR_KEYWORD_PROCESSOR
    
    if _DESCRIPTOR_PHRASES:
        return
    
    r = rules()
    
    # Build FlashText processor for category tags
    _CATEGORY_KEYWORD_PROCESSOR = KeywordProcessor(case_sensitive=False)
    for tag, cfg in r.get("category_tags", {}).items():
        for phrase in cfg.get("positive", []):
            normalized = normalize_text(str(phrase).replace("_", " "))
            if normalized:
                _CATEGORY_KEYWORD_PROCESSOR.add_keyword(normalized, (tag, "positive"))
        for phrase in cfg.get("negative", []):
            normalized = normalize_text(str(phrase).replace("_", " "))
            if normalized:
                _CATEGORY_KEYWORD_PROCESSOR.add_keyword(normalized, (tag, "negative"))
    
    # Build FlashText processor for descriptor tags
    _DESCRIPTOR_KEYWORD_PROCESSOR = KeywordProcessor(case_sensitive=False)
    for tag, cfg in r.get("descriptor_tags", {}).items():
        for phrase in cfg.get("positive", []):
            phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
            if phrase_tokens:
                _DESCRIPTOR_PHRASES[phrase_tokens] = (tag, False)
                _DESCRIPTOR_KEYWORD_PROCESSOR.add_keyword(phrase.replace("_", " "), (tag, "positive", phrase_tokens))
        for phrase in cfg.get("negative", []):
            phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
            if phrase_tokens:
                _DESCRIPTOR_PHRASES[phrase_tokens] = (tag, True)
                _DESCRIPTOR_KEYWORD_PROCESSOR.add_keyword(phrase.replace("_", " "), (tag, "negative", phrase_tokens))


def _find_phrase_in_tokens(tokens: list[str], phrase_tokens: tuple[str, ...]) -> list[int]:
    positions = []
    phrase_len = len(phrase_tokens)
    for i in range(len(tokens) - phrase_len + 1):
        if tuple(tokens[i:i + phrase_len]) == phrase_tokens:
            positions.append(i)
    return positions


def _get_token_window(tokens: list[str], center_idx: int, window: int = 5) -> str:
    start = max(0, center_idx - window)
    end = min(len(tokens), center_idx + window + 1)
    return " ".join(tokens[start:end])


def _check_exclusion(text: str, tag: str) -> bool:
    exclusions = EXCLUSION_PHRASES.get(tag, [])
    text_lower = text.lower()
    for phrase in exclusions:
        if phrase in text_lower:
            return True
    return False


def _validate_subject(tokens: list[str], tag: str, descriptor_pos: int, window: int = 15) -> bool:
    if tag not in GENERIC_DESCRIPTORS:
        return True
    subjects = SUBJECT_VALIDATION.get(tag, [])
    if not subjects:
        return True
    start = max(0, descriptor_pos - window)
    end = min(len(tokens), descriptor_pos + window)
    context_tokens = tokens[start:end]
    for subject in subjects:
        for tok in context_tokens:
            if subject in tok or tok in subject:
                return True
    return False


def _extract_descriptor_contexts(text: str, window: int = 5) -> list[dict]:
    _build_phrase_maps()
    tokens = tokenize_vi(text.lower())
    matches = []
    
    for phrase_tokens, (tag, is_negative) in _DESCRIPTOR_PHRASES.items():
        positions = _find_phrase_in_tokens(tokens, phrase_tokens)
        for start_idx in positions:
            if _check_exclusion(text, tag):
                continue
            if not _validate_subject(tokens, tag, start_idx, window=15):
                continue
            end_idx = start_idx + len(phrase_tokens)
            phrase_str = " ".join(tokens[start_idx:end_idx])
            matches.append((start_idx, end_idx, tag, is_negative, phrase_str))
    
    matches.sort(key=lambda x: x[0])
    
    merged = []
    for match in matches:
        start, end, tag, is_neg, phrase = match
        if merged and start < merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), prev[2] + "|" + tag, prev[3] or is_neg, prev[4] + " + " + phrase)
        else:
            merged.append(match)
    
    results = []
    for start, end, tag, is_neg, phrase in merged:
        center_idx = (start + end) // 2
        context = _get_token_window(tokens, center_idx, window)
        results.append({
            "descriptor": phrase,
            "tag": tag,
            "rule_polarity": "negative" if is_neg else "positive",
            "context": context,
            "phobert_sentiment": "",
            "token_range": (start, end),
        })
    return results


def _extract_category_tags(text: str) -> list[str]:
    _build_phrase_maps()
    tokens = tokenize_vi(text.lower())
    tags = set()
    for phrase_tokens, tag in _CATEGORY_PHRASES.items():
        if _find_phrase_in_tokens(tokens, phrase_tokens):
            tags.add(tag)
    return list(tags)


# PhoBERT model + tokenizer (lazy loaded)
_PHOBERT_TOKENIZER = None
_PHOBERT_MODEL = None
_PHOBERT_DEVICE = None

def _load_phobert_model():
    """Load PhoBERT model và tự động dùng GPU nếu có."""
    global _PHOBERT_TOKENIZER, _PHOBERT_MODEL, _PHOBERT_DEVICE
    
    if _PHOBERT_MODEL is not None:
        return _PHOBERT_MODEL, _PHOBERT_TOKENIZER, _PHOBERT_DEVICE
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from warnings import filterwarnings
        filterwarnings("ignore")
        
        model_name = "wonrax/phobert-base-vietnamese-sentiment"
        _PHOBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _PHOBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Tự động dùng GPU nếu có
        if torch.cuda.is_available():
            _PHOBERT_DEVICE = torch.device("cuda")
            _PHOBERT_MODEL = _PHOBERT_MODEL.to(_PHOBERT_DEVICE)
            print(f"[PhoBERT] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            _PHOBERT_DEVICE = torch.device("cpu")
            print("[PhoBERT] Using CPU")
        
        _PHOBERT_MODEL.eval()
    except Exception as e:
        _PHOBERT_MODEL = "lexicon"
        _PHOBERT_DEVICE = "cpu"
    
    return _PHOBERT_MODEL, _PHOBERT_TOKENIZER, _PHOBERT_DEVICE


def _phobert_batch_predict(contexts: list[str], batch_size: int = 32) -> list[str]:
    """Batch inference cho PhoBERT - NHANH HƠN inference từng câu."""
    model, tokenizer, device = _load_phobert_model()
    
    if model == "lexicon":
        return [_lexicon_sentiment(ctx) for ctx in contexts]
    
    results = []
    
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if device.type == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            for pred in predictions:
                label = model.config.id2label[pred.item()]
                if label == "POS":
                    results.append("positive")
                elif label == "NEG":
                    results.append("negative")
                else:
                    results.append("neutral")
    
    return results


def _phobert_sentiment_predict(context: str) -> str:
    # Check strong negative keywords first
    context_lower = context.lower()
    for kw in STRONG_NEGATIVE_KEYWORDS:
        if kw in context_lower:
            return "negative"
    
    model, tokenizer, device = _load_phobert_model()
    
    if model == "lexicon":
        return _lexicon_sentiment(context)
    else:
        try:
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            if device.type == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                label = model.config.id2label[predictions[0].item()]
            
            if label == "POS":
                return "positive"
            elif label == "NEG":
                return "negative"
            else:
                return "neutral"
        except Exception:
            return _lexicon_sentiment(context)


def _lexicon_sentiment(context: str) -> str:
    positive_words = {"tốt", "đẹp", "sạch", "thoải_mái", "nhiệt_tình", "thân_thiện", 
                      "tiện_nghi", "hợp_lý", "xịn", "mới", "rộng", "gọn_gàng", 
                      "chu_đáo", "hài_lòng", "tuyệt_vời", "ok", "ổn", "rộng_rãi",
                      "sạch_sẽ", "tiện_lợi", "thuận_tiện", "vui_vẻ", "dễ_gần",
                      "hỗ_trợ", "yên_tĩnh", "tạm_ổn"}
    negative_words = {"tệ", "dở", "bẩn", "chật", "nhỏ", "cũ", "hỏng", "kém", 
                      "bất_tiện", "xa", "đắt", "ồn", "ồn_ào", "lạnh", "nóng", "thiếu",
                      "kinh_khủng", "khó_chịu", "tối", "gãy", "trống", "nồng",
                      "khó_ngủ", "không_tốt", "cao", "chưa"}
    
    tokens = set(context.split())
    pos_count = len(tokens & positive_words)
    neg_count = len(tokens & negative_words)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def _assign_tags_from_sentiment(contexts: list[dict]) -> tuple[list[str], dict[str, str]]:
    """
    Gán tag dựa trên sentiment. Nếu cùng tag có cả positive và negative → ưu tiên negative.
    """
    # Thu thập tất cả polarity cho mỗi tag
    tag_polarities: dict[str, list[str]] = {}
    for ctx in contexts:
        sentiment = ctx.get("phobert_sentiment", "")
        rule_polarity = ctx.get("rule_polarity", "")
        tag = ctx["tag"].split("|")[0]
        
        if sentiment == "positive":
            tag_polarities.setdefault(tag, []).append("positive")
        elif sentiment == "negative":
            tag_polarities.setdefault(tag, []).append("negative")
        else:
            # Neutral → dùng rule_polarity
            if rule_polarity in ("positive", "negative"):
                tag_polarities.setdefault(tag, []).append(rule_polarity)
    
    # Assign tags: nếu có cả positive và negative → ưu tiên negative
    descriptor_tags = []
    descriptor_polarity = {}
    for tag, polarities in tag_polarities.items():
        if "negative" in polarities:
            neg_tag = f"!{tag}"
            descriptor_tags.append(neg_tag)
            descriptor_polarity[tag] = "negative"
        elif "positive" in polarities:
            descriptor_tags.append(tag)
            descriptor_polarity[tag] = "positive"
    
    return descriptor_tags, descriptor_polarity


def tag_review(review_text: str, hotel_name: str = "", location: str = "") -> TagResult:
    """Rule-based + PhoBERT sentiment tagging for a single review."""
    payload = rules()
    text = _prepare_text(review_text, hotel_name, location)
    if not text:
        return TagResult()
    
    # Category tags (rule-based)
    category_rules = payload.get("category_tags", {})
    category_tags, category_matches = _apply_tag_group(text, category_rules)
    
    # Descriptor tags with PhoBERT sentiment
    contexts = _extract_descriptor_contexts(text, window=5)
    
    # Predict sentiment for each context
    for ctx in contexts:
        ctx["phobert_sentiment"] = _phobert_sentiment_predict(ctx["context"])
    
    # Assign tags based on sentiment
    descriptor_tags, descriptor_polarity = _assign_tags_from_sentiment(contexts)
    
    # Build matched phrases for backward compatibility
    descriptor_matches = {}
    for ctx in contexts:
        tag = ctx["tag"].split("|")[0]
        sentiment = ctx.get("phobert_sentiment", ctx.get("rule_polarity"))
        if sentiment == "positive":
            descriptor_matches.setdefault(tag, []).append(ctx["descriptor"])
        elif sentiment == "negative":
            descriptor_matches.setdefault(f"!{tag}", []).append(ctx["descriptor"])
    
    return TagResult(
        category_tags=category_tags,
        descriptor_tags=descriptor_tags,
        descriptor_polarity=descriptor_polarity,
        matched_phrases={
            "category_tags": category_matches,
            "descriptor_tags": descriptor_matches,
        },
        contexts=contexts,
    )


def tag_records_batch(records: list[dict], phobert_batch_size: int = 32) -> list[dict]:
    """Batch processing cho nhiều reviews - TỐI ƯU GPU."""
    # Bước 1: Extract contexts cho tất cả reviews
    all_contexts = []
    review_context_ranges = []  # (start_idx, end_idx) cho mỗi review
    
    print("[Step 1/3] Extracting descriptor contexts...")
    for record in tqdm(records, desc="Extracting contexts"):
        text = _prepare_text(
            str(record.get("review_text", "")),
            str(record.get("hotel_name", "")),
            str(record.get("location", ""))
        )
        if not text:
            review_context_ranges.append((0, 0))
            continue
        
        contexts = _extract_descriptor_contexts(text, window=5)
        start = len(all_contexts)
        all_contexts.extend(contexts)
        review_context_ranges.append((start, len(all_contexts)))
    
    # Bước 2: Batch PhoBERT inference
    if all_contexts:
        print(f"\n[Step 2/3] Running PhoBERT inference on {len(all_contexts)} contexts (batch_size={phobert_batch_size})...")
        sentiments = []
        for i in tqdm(range(0, len(all_contexts), phobert_batch_size), desc="PhoBERT inference"):
            batch_contexts = [ctx["context"] for ctx in all_contexts[i:i+phobert_batch_size]]
            batch_sentiments = _phobert_batch_predict(batch_contexts, batch_size=phobert_batch_size)
            sentiments.extend(batch_sentiments)
        
        for ctx, sent in zip(all_contexts, sentiments):
            ctx["phobert_sentiment"] = sent
    else:
        print("\n[Step 2/3] No contexts to process, skipping PhoBERT.")
    
    # Bước 3: Assign tags cho mỗi review
    print("\n[Step 3/3] Assigning tags to reviews...")
    output = []
    for i, record in enumerate(tqdm(records, desc="Assigning tags")):
        start, end = review_context_ranges[i]
        contexts = all_contexts[start:end]
        
        descriptor_tags, descriptor_polarity = _assign_tags_from_sentiment(contexts)
        
        # Category tags
        text = _prepare_text(
            str(record.get("review_text", "")),
            str(record.get("hotel_name", "")),
            str(record.get("location", ""))
        )
        if not text:
            continue
            
        category_rules = rules().get("category_tags", {})
        category_tags, category_matches = _apply_tag_group(text, category_rules)
        
        descriptor_matches = {}
        for ctx in contexts:
            tag = ctx["tag"].split("|")[0]
            sentiment = ctx.get("phobert_sentiment", ctx.get("rule_polarity"))
            if sentiment == "positive":
                descriptor_matches.setdefault(tag, []).append(ctx["descriptor"])
            elif sentiment == "negative":
                descriptor_matches.setdefault(f"!{tag}", []).append(ctx["descriptor"])
        
        tagged = dict(record)
        tagged["category_tags"] = category_tags
        tagged["descriptor_tags"] = descriptor_tags
        tagged["descriptor_polarity"] = descriptor_polarity
        tagged["matched_phrases"] = {
            "category_tags": category_matches,
            "descriptor_tags": descriptor_matches,
        }
        tagged["contexts"] = contexts
        output.append(tagged)
    
    return output


def tag_record(record: dict) -> dict:
    review_text = str(record.get("review_text", ""))
    hotel_name = str(record.get("hotel_name", ""))
    location = str(record.get("location", ""))
    res = tag_review(review_text, hotel_name=hotel_name, location=location)
    
    tagged = dict(record)
    tagged["category_tags"] = res.category_tags
    tagged["descriptor_tags"] = res.descriptor_tags
    tagged["descriptor_polarity"] = res.descriptor_polarity
    tagged["matched_phrases"] = res.matched_phrases
    tagged["contexts"] = res.contexts
    return tagged