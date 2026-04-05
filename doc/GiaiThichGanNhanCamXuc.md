# GIẢI THÍCH LUỒNG XỬ LÝ GÁN NHÃN CẢM XÚC

---

## TỔNG QUAN

Hệ thống gán nhãn cảm xúc cho review khách sạn với **2 loại tags**:

| Loại tag | Mô tả | Ví dụ |
|----------|-------|-------|
| **Category Tags** | Phân loại khách sạn (phù hợp cho ai, ở đâu) | `beach`, `budget`, `luxury`, `center` |
| **Descriptor Tags** | Đánh giá chất lượng (tốt/tệ, có/không) | `cleanliness`, `room_spacious`, `staff_friendly` |

---

## LUỒNG 1: OFFLINE TAGGING (Khi preprocess reviews)

### Bước 1: Chuẩn bị text
```python
Input: review_text, hotel_name, location
     ↓
clean_review_text() → Loại HTML, ký tự đặc biệt
     ↓
normalize_text() → Unicode, lowercase
     ↓
tokenize_vi() → Tách từ tiếng Việt
```

**Ví dụ:**
```
"Phòng rộng rãi, sạch sẽ. Nhân viên thân thiện!"
→ "phòng rộng_rãi sạch_sẽ nhân_viên thân_thiện"
```

---

### Bước 2: Category Tags (FlashText matching)
```python
Input: normalized text
     ↓
_build_phrase_maps() → Load rules YAML vào KeywordProcessor
     ↓
FlashText.extract_keywords(text) → Quét 1 lần O(m)
     ↓
Found: [("beach", "positive"), ("budget", "negative"), ...]
     ↓
Conflict Resolution:
  - Cùng tag có cả + và - → ưu tiên NEGATIVE
  - "gần biển" (+) và "xa biển" (-) → "!beach"
     ↓
Output: category_tags = ["beach", "!budget"]
```

**Cách hoạt động FlashText:**
```
Rules YAML có 200+ keywords:
  - "gần biển" → beach (positive)
  - "xa biển" → beach (negative)
  - "giá rẻ" → budget (positive)
  - "đắt đỏ" → budget (negative)
  ...

Query: "khách sạn gần biển, giá rẻ"
FlashText quét 1 lần → tìm thấy: "gần biển", "giá rẻ"
→ Tags: beach (+), budget (+)
```

---

### Bước 3: Descriptor Tags (Phức tạp hơn)

#### 3a. Descriptor Phrase Matching
```python
Input: normalized text
     ↓
Tokenize: ["phòng", "rộng_rãi", "sạch_sẽ"]
     ↓
_DDESCRIPTOR_PHRASES dict:
  - ("rộng", "rãi") → room_spacious (positive)
  - ("chật", "chội") → room_spacious (negative)
  - ("sạch", "sẽ") → cleanliness (positive)
  - ("bẩn", "thỉu") → cleanliness (negative)
     ↓
_find_phrase_in_tokens() → Tìm vị trí exact trong tokens
→ Found: room_spacious at pos 1, cleanliness at pos 2
```

#### 3b. Subject Validation (tránh match sai)
```python
Mục đích: "nhỏ" trong "cháu nhỏ" ≠ "nhỏ" trong "phòng nhỏ"

GENERIC_DESCRIPTORS = {room_spacious, cleanliness, ...}

Ví dụ: "Cháu nhỏ đáng yêu, phòng thì nhỏ"
     ↓
- "nhỏ" thứ 1 → subject = "cháu" → KHÔNG match room_spacious ✓
- "nhỏ" thứ 2 → subject = "phòng" → match room_spacious ✓

SUBJECT_VALIDATION:
  - room_spacious → cần subject: "phòng", "khách_sạn", "giường"...
  - cleanliness → cần subject: "phòng", "bathroom", "nệm"...
  - staff_friendly → cần subject: "nhân_viên", "lễ_tân"...
```

#### 3c. Exclusion Check (loại false positives)
```python
EXCLUSION_PHRASES:
  - room_spacious: ["cháu nhỏ", "em nhỏ", "nhỏ xíu", "nhỏ tí hon"]
  - cleanliness: ["sạch nợ", "sạch trơn"]

Ví dụ: "sạch nợ luôn" → KHÔNG match cleanliness ✓
```

#### 3d. Extract Context ±5 tokens
```python
Mục đích: Lấy context quanh descriptor để đưa vào PhoBERT

Review: "Phòng rộng rãi nhưng ban công không được sạch"
     ↓
Descriptor "rộng rãi" tại pos 1:
  Context = "phòng rộng rãi nhưng ban" (±5 tokens)
     ↓
Descriptor "sạch" tại pos 6:
  Context = "nhưng ban công không được sạch" (±5 tokens)
```

---

### Bước 4: PhoBERT Sentiment Prediction
```python
Input: context string (từ bước 3d)
     ↓
STRONG_NEGATIVE_KEYWORDS check trước:
  - Context có "gián", "chuột", "khủng khiếp" → NEGATIVE ngay
     ↓
Nằm ngoài → PhoBERT model:
  Model: wonrax/phobert-base-vietnamese-sentiment
  Input: "ban công không được sạch"
  Output: NEG (negative)
     ↓
Fallback: Nếu PhoBERT lỗi → _lexicon_sentiment()
  - Đếm positive/negative words trong context
  - pos > neg → positive
  - neg > pos → negative
  - bằng nhau → neutral
```

---

### Bước 5: Conflict Resolution (Final Tag Assignment)
```python
Input: contexts với phobert_sentiment
     ↓
Thu thập polarities cho mỗi tag:
  - room_spacious: ["positive"]
  - cleanliness: ["negative", "positive"] ← có cả 2!
     ↓
Rule: Nếu có cả positive và negative → ưu tiên NEGATIVE
     ↓
Output:
  descriptor_tags = ["room_spacious", "!cleanliness"]
  descriptor_polarity = {room_spacious: "positive", cleanliness: "negative"}
```

---

## LUỒNG 2: ONLINE QUERY TAGGING (Khi user search)

### File: retrieval/query_tag_extractor.py

```python
User Query: "khách sạn gần biển giá rẻ ở Vũng Tàu"
     ↓
FlashText.extract_keywords(query) → Quét query 1 lần
     ↓
Found: ["gần biển" → beach, "giá rẻ" → budget]
     ↓
Output:
  {
    "category_tags": ["beach", "budget"],
    "descriptor_tags": [],
    "all_tags": ["beach", "budget"]
  }
     ↓
Dùng để filter MongoDB:
  reviews có category_tags chứa "beach" HOẶC "budget"
```

**Lưu ý:** Query tagging chỉ dùng FlashText (nhanh), KHÔNG dùng PhoBERT (chậm).

---

## VÍ DỤ HOÀN CHỈNH

### Input Review:
```
"Khách sạn gần biển, giá hợp lý. Phòng rộng rãi, sạch sẽ. 
Nhân viên thân thiện nhưng cách âm tệ, ồn cả đêm. Ăn sáng ngon."
```

### Output Tags:
```python
{
  "category_tags": ["beach", "budget"],
  "descriptor_tags": [
    "room_spacious",      # positive
    "cleanliness",        # positive
    "staff_friendly",     # positive
    "!soundproofing",     # negative (cách âm tệ)
    "food_quality",       # positive (ăn sáng ngon)
  ],
  "descriptor_polarity": {
    "room_spacious": "positive",
    "cleanliness": "positive",
    "staff_friendly": "positive",
    "soundproofing": "negative",
    "food_quality": "positive"
  }
}
```

---

## TÓM TẮT

| Bước | Chức năng | Công nghệ |
|------|-----------|-----------|
| 1. Prepare text | Clean + normalize + tokenize | NLP utils |
| 2. Category tags | FlashText keyword matching | FlashText O(m) |
| 3a. Descriptor match | Token-level phrase matching | Dict lookup |
| 3b. Subject validation | Kiểm tra subject đúng | Window matching |
| 3c. Exclusion check | Loại false positives | String contains |
| 3d. Extract context | Lấy ±5 tokens quanh descriptor | Token slicing |
| 4. Sentiment predict | PhoBERT prediction | Transformers |
| 5. Conflict resolution | POS + NEG → ưu tiên NEG | Rule-based |