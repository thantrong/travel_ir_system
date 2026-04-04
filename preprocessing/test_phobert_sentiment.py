"""
Test script tích hợp PhoBERT để phân loại sentiment cho từng descriptor context.
Pipeline:
1. Trích xuất descriptor + context (từ test_descriptor_extraction.py)
2. Kiểm tra subject validation: descriptor phải đi với subject phù hợp
3. Đưa context vào PhoBERT sentiment classifier
4. Gán polarity dựa trên kết quả classifier

Model: wonrax/phobert-base-vietnamese-sentiment
Link: https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment
Labels: NEG (negative), POS (positive), NEU (neutral)
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi
import yaml

# Load rules
RULE_FILE = PROJECT_ROOT / "config" / "review_tag_rules.yaml"
rules = yaml.safe_load(RULE_FILE.read_text(encoding="utf-8"))

# Subject mapping: descriptor tag -> subjects phù hợp
# Nếu descriptor generic cần subject validation
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
    "soundproofing": ["cách_âm", "tiếng_ồn", "âm_thanh", "ồn", "yên_tĩnh", "phòng", "đêm"],
    "check_in": ["check_in", "checkin", "nhận_phòng", "lễ_tân", "phòng"],
    "housekeeping": ["dọn_phòng", "phòng", "dọn_dẹp"],
    "security": ["an_ninh", "bảo_vệ", "an_toàn", "trộm"],
}

# Descriptor cần subject validation (generic words)
GENERIC_DESCRIPTORS = {"room_spacious", "cleanliness", "ambiance", "convenience", "soundproofing", "air_conditioning"}

# Exclusion phrases: nếu context chứa các từ này → skip match cho tag tương ứng
# Dùng để tránh match sai như "cháu nhỏ" → room_spacious
EXCLUSION_PHRASES = {
    "room_spacious": [
        "cháu nhỏ", "em nhỏ", "con nhỏ", "người nhỏ", "bạn nhỏ", "bé nhỏ",
        "nhỏ tuổi", "nhỏ tuổi hơn", "nhỏ hơn", "nhỏ nhất",
        "nhỏ lòng", "nhỏ nhen", "nhỏ nhẹ", "nhỏ to",
        "nhỏ xíu", "nhỏ tí", "nhỏ tí hon",
    ],
    "cleanliness": [
        "sạch nợ", "sạch trơn",
    ],
}

# Build descriptor phrase map
descriptor_phrases = {}
for tag, cfg in rules.get("descriptor_tags", {}).items():
    for phrase in cfg.get("positive", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            descriptor_phrases[phrase_tokens] = (tag, False)
    for phrase in cfg.get("negative", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            descriptor_phrases[phrase_tokens] = (tag, True)

category_phrases = {}
for tag, cfg in rules.get("category_tags", {}).items():
    for phrase in cfg.get("positive", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            category_phrases[phrase_tokens] = tag
    for phrase in cfg.get("negative", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            category_phrases[phrase_tokens] = tag


def find_phrase_in_tokens(tokens: list[str], phrase_tokens: tuple[str, ...]) -> list[int]:
    positions = []
    phrase_len = len(phrase_tokens)
    for i in range(len(tokens) - phrase_len + 1):
        if tuple(tokens[i:i + phrase_len]) == phrase_tokens:
            positions.append(i)
    return positions


def get_token_window(tokens: list[str], center_idx: int, window: int = 5) -> str:
    start = max(0, center_idx - window)
    end = min(len(tokens), center_idx + window + 1)
    return " ".join(tokens[start:end])


def check_exclusion(text: str, tag: str) -> bool:
    """
    Kiểm tra xem text có chứa exclusion phrase cho tag không.
    Nếu có → trả về True (bị loại).
    """
    exclusions = EXCLUSION_PHRASES.get(tag, [])
    text_lower = text.lower()
    for phrase in exclusions:
        if phrase in text_lower:
            return True
    return False


def validate_subject(tokens: list[str], tag: str, descriptor_pos: int, window: int = 15) -> bool:
    """
    Kiểm tra xem có subject phù hợp với descriptor tag trong context.
    Window rộng hơn context sentiment để catch subject ở xa.
    Nếu tag không cần validation → trả về True.
    """
    if tag not in GENERIC_DESCRIPTORS:
        return True
    
    subjects = SUBJECT_VALIDATION.get(tag, [])
    if not subjects:
        return True
    
    # Search trong window quanh descriptor
    start = max(0, descriptor_pos - window)
    end = min(len(tokens), descriptor_pos + window)
    context_tokens = tokens[start:end]
    
    for subject in subjects:
        for tok in context_tokens:
            if subject in tok or tok in subject:
                return True
    return False


def extract_descriptor_contexts(text: str, window: int = 5) -> list[dict]:
    tokens = tokenize_vi(text.lower())
    matches = []
    
    for phrase_tokens, (tag, is_negative) in descriptor_phrases.items():
        positions = find_phrase_in_tokens(tokens, phrase_tokens)
        for start_idx in positions:
            # Exclusion check: skip nếu text chứa exclusion phrase
            if check_exclusion(text, tag):
                continue
            
            # Subject validation (window rộng hơn để catch subject xa)
            if not validate_subject(tokens, tag, start_idx, window=15):
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
        context = get_token_window(tokens, center_idx, window)
        results.append({
            "descriptor": phrase,
            "tag": tag,
            "rule_polarity": "negative" if is_neg else "positive",
            "context": context,
            "phobert_sentiment": "",
            "token_range": (start, end),
            "subject_validated": True,
        })
    return results


def extract_category_tags(text: str) -> list[str]:
    tokens = tokenize_vi(text.lower())
    tags = set()
    for phrase_tokens, tag in category_phrases.items():
        if find_phrase_in_tokens(tokens, phrase_tokens):
            tags.add(tag)
    return list(tags)


# --- PhoBERT Sentiment Classifier ---
_PHOBERT_PIPELINE = None

def _load_phobert_model():
    """
    Load PhoBERT model đã fine-tune cho sentiment analysis.
    Model: wonrax/phobert-base-vietnamese-sentiment
    Labels: NEG, POS, NEU
    """
    global _PHOBERT_PIPELINE
    if _PHOBERT_PIPELINE is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            from warnings import filterwarnings
            filterwarnings("ignore")
            
            model_name = "wonrax/phobert-base-vietnamese-sentiment"
            print(f"Loading model: {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            _PHOBERT_PIPELINE = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1  # CPU
            )
            print("Loaded PhoBERT sentiment model")
        except Exception as e:
            print(f"Không thể load PhoBERT model: {e}")
            print("Sử dụng lexicon-based sentiment làm fallback")
            _PHOBERT_PIPELINE = "lexicon"
    return _PHOBERT_PIPELINE


# Các từ negative mạnh: nếu context chứa các từ này → force negative
STRONG_NEGATIVE_KEYWORDS = {
    "gián", "chuột", "rận", "bọ", "muỗi", "ruồi", "nhện", "mối", "mọt",
    "phân", "nước tiểu", "mùi hôi", "thối", "khủng khiếp", "kinh khủng",
    "ghê tởm", "ghê", "sợ", "ác mộng", "ám ảnh",
}

def phobert_sentiment_predict(context: str) -> str:
    """
    Dự đoán sentiment của context sử dụng PhoBERT fine-tuned.
    Output: "positive", "negative", "neutral"
    """
    # Check strong negative keywords first
    context_lower = context.lower()
    for kw in STRONG_NEGATIVE_KEYWORDS:
        if kw in context_lower:
            return "negative"
    
    model = _load_phobert_model()
    
    if model == "lexicon":
        return _lexicon_sentiment(context)
    else:
        try:
            result = model(context)[0]
            label = result["label"]
            score = result["score"]
            
            # Map label từ model sang output chuẩn
            if label == "POS":
                return "positive"
            elif label == "NEG":
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            print(f"Prediction error: {e}")
            return _lexicon_sentiment(context)


def _lexicon_sentiment(context: str) -> str:
    """Lexicon-based sentiment làm fallback."""
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


def assign_tags_from_sentiment(contexts: list[dict]) -> dict:
    """
    Gán tag cho review dựa trên sentiment kết quả.
    Logic xử lý:
    - PhoBERT positive → gán tag (positive)
    - PhoBERT negative → gán !tag (negative)
    - PhoBERT neutral → Dùng rule_polarity từ matched phrase
      - Nếu rule_polarity là negative → gán !tag
      - Nếu rule_polarity là positive → gán tag
      - Nếu không rõ → không gán
    """
    descriptor_tags = []
    descriptor_polarity = {}
    neutral_contexts = []
    
    for ctx in contexts:
        sentiment = ctx.get("phobert_sentiment", "")
        rule_polarity = ctx.get("rule_polarity", "")
        tag = ctx["tag"].split("|")[0]
        
        if sentiment == "positive":
            if tag not in descriptor_tags:
                descriptor_tags.append(tag)
            descriptor_polarity[tag] = "positive"
        elif sentiment == "negative":
            neg_tag = f"!{tag}"
            if neg_tag not in descriptor_tags:
                descriptor_tags.append(neg_tag)
            descriptor_polarity[tag] = "negative"
        else:
            # PhoBERT trả về neutral → dùng rule_polarity
            neutral_contexts.append(ctx)
            if rule_polarity == "negative":
                neg_tag = f"!{tag}"
                if neg_tag not in descriptor_tags:
                    descriptor_tags.append(neg_tag)
                descriptor_polarity[tag] = "negative"
            elif rule_polarity == "positive":
                if tag not in descriptor_tags:
                    descriptor_tags.append(tag)
                descriptor_polarity[tag] = "positive"
    
    # Log neutral contexts để debug
    if neutral_contexts:
        print(f"  [DEBUG] {len(neutral_contexts)} context(s) neutral - dùng rule_polarity:")
        for ctx in neutral_contexts:
            tag = ctx["tag"].split("|")[0]
            print(f"    - {ctx['descriptor']} (rule: {ctx['rule_polarity']}) → {descriptor_polarity.get(tag, 'not assigned')}")
    
    return {
        "descriptor_tags": descriptor_tags,
        "descriptor_polarity": descriptor_polarity,
    }


# Load raw data from multiple sources
RAW_FILES = [
    PROJECT_ROOT / "data" / "raw" / "traveloka_raw_final.json",
    PROJECT_ROOT / "data" / "raw" / "traveloka_checkpoint.jsonl"
]

raw_reviews = []
for raw_file in RAW_FILES:
    if not raw_file.exists():
        continue
    if raw_file.suffix == ".json":
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                raw_reviews.extend(data)
    elif raw_file.suffix == ".jsonl":
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_reviews.append(json.loads(line))

# Deduplicate by review_id
seen = set()
deduped = []
for r in raw_reviews:
    rid = r.get("review_id", r.get("id", ""))
    if rid not in seen:
        seen.add(rid)
        deduped.append(r)
raw_reviews = deduped

print("=" * 80)
print(f"TEST PHOBERT SENTIMENT + TAG ASSIGNMENT ({len(raw_reviews)} reviews)")
print(f"Model: wonrax/phobert-base-vietnamese-sentiment")
print(f"Subject validation: {list(GENERIC_DESCRIPTORS)}")
print("=" * 80)

all_results = []

for review in raw_reviews:
    review_id = review.get("review_id", "")
    review_text = review.get("review_text", "")
    review_rating = review.get("review_rating", "")
    
    print(f"\n{'='*80}")
    print(f"Review ID: {review_id}")
    print(f"Rating: {review_rating}")
    print(f"Text: {review_text[:80]}...")
    print(f"{'='*80}")
    
    categories = extract_category_tags(review_text)
    contexts = extract_descriptor_contexts(review_text, window=5)
    
    for ctx in contexts:
        ctx["phobert_sentiment"] = phobert_sentiment_predict(ctx["context"])
    
    result = assign_tags_from_sentiment(contexts)
    
    print(f"  Categories: {categories}")
    print(f"  Descriptor tags: {result['descriptor_tags']}")
    print(f"  Polarity: {result['descriptor_polarity']}")
    
    all_results.append({
        "review_id": review_id,
        "rating": review_rating,
        "category_tags": categories,
        "descriptor_tags": result["descriptor_tags"],
        "descriptor_polarity": result["descriptor_polarity"],
        "contexts": contexts,
    })

# Save results
OUTPUT_FILE = PROJECT_ROOT / "preprocessing" / "test_results.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print(f"RESULTS SAVED TO: {OUTPUT_FILE}")
print(f"Total reviews processed: {len(all_results)}")
print("=" * 80)