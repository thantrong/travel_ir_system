# TÀI LIỆU HỆ THỐNG TÌM KIẾM KHÁCH SẠN

---

## 1. TỔNG QUAN HỆ THỐNG

Hệ thống tìm kiếm khách sạn với 2 pipeline chính:
- **Offline Pipeline**: Xử lý dữ liệu, gán nhãn cảm xúc, build index
- **Online Pipeline**: Nhận query, tìm kiếm, ranking, trả về kết quả

---

## 2. OFFLINE PIPELINE (Chạy khi có data mới)

### Bước 1: Crawl Data
```
File: crawler/traveloka_crawler.py
Input: URL trang web Traveloka
Output: 
  - data/raw/traveloka_raw_final.json (19 reviews)
  - data/raw/traveloka_checkpoint.jsonl (76 reviews)

Cấu trúc 1 review:
{
  "review_id": "traveloka_9000000796940_126228038",
  "source_hotel_id": "9000000796940",
  "hotel_name": "Khách sạn ABC",
  "location": "Vũng Tàu",
  "rating": "4.2",
  "review_text": "Phòng rộng rãi, sạch sẽ...",
  "review_date": "2024-01-01"
}
```

### Bước 2: Preprocessing
```
File: main.py
Input: Raw reviews JSON
Output: data/processed/reviews_processed.json

Luồng xử lý:
┌─────────────────────────────────────────────────────┐
│ 1. clean_review_text()                              │
│    - Loại HTML tags, ký tự đặc biệt                 │
│    - Chuẩn hóa khoảng trắng                         │
│                                                     │
│ 2. normalize_text()                                 │
│    - Unicode normalization (á → á)                  │
│    - Lowercase                                      │
│                                                     │
│ 3. tokenize_vi()                                    │
│    - Tách từ tiếng Việt                             │
│    - "khách sạn" → ["khách_sạn"]                    │
│                                                     │
│ 4. remove_stopwords()                               │
│    - Loại: là, và, của, ở, tại...                   │
│                                                     │
│ 5. tag_record() ← review_tagger.py                  │
│    - Gán category_tags                              │
│    - Gán descriptor_tags + polarity                 │
│                                                     │
│ 6. Lưu kết quả                                      │
└─────────────────────────────────────────────────────┘
```

### Bước 3: Tagging (review_tagger.py)
```
┌─────────────────────────────────────────────────────┐
│ Input: review_text, hotel_name, location            │
│                                                     │
│ 1. FlashText Category Matching                      │
│    - Build KeywordProcessor từ rules YAML           │
│    - Extract keywords trong 1 lần quét O(m)         │
│    - VD: "gần biển" → ("beach", "positive")         │
│                                                     │
│ 2. Descriptor Phrase Matching                       │
│    - Tokenize review                                │
│    - Tìm phrases trong _DESCRIPTOR_PHRASES dict     │
│    - Lưu vị trí (start_idx, end_idx)                │
│                                                     │
│ 3. Subject Validation                               │
│    - "phòng nhỏ" → room_spacious ✓                  │
│    - "cháu nhỏ" → room_spacious ✗                   │
│    - Window = ±15 tokens quanh descriptor           │
│                                                     │
│ 4. Exclusion Check                                  │
│    - "sạch nợ" → KHÔNG match cleanliness            │
│    - "nhỏ xíu" → KHÔNG match room_spacious          │
│                                                     │
│ 5. Extract Context ±5 tokens                        │
│    - VD: "ban công không được sạch . giá phòng"     │
│                                                     │
│ 6. PhoBERT Sentiment Prediction                     │
│    - Model: wonrax/phobert-base-vietnamese-sentiment│
│    - Input: context string                          │
│    - Output: POS/NEG/NEU                            │
│                                                     │
│ 7. Strong Negative Override                         │
│    - Context có "gián", "chuột" → NEG ngay          │
│                                                     │
│ 8. Conflict Resolution                              │
│    - Cùng tag có cả POS + NEG → ưu tiên NEG         │
│                                                     │
│ Output:                                              │
│ {                                                   │
│   "category_tags": ["beach", "budget"],             │
│   "descriptor_tags": ["cleanliness", "!room_spacious"],
│   "descriptor_polarity": {                          │
│     "cleanliness": "positive",                      │
│     "room_spacious": "negative"                     │
│   }                                                 │
│ }                                                   │
└─────────────────────────────────────────────────────┘
```

### Bước 4: Load vào MongoDB
```
File: database/data_loader.py
Input: reviews_processed.json
Output: MongoDB collections

Places collection:
{
  "_id": "9000000796940",
  "types": ["hotel"],
  "location": "Vũng Tàu",
  "rating": "4.2",
  "hotel_name": "Khách sạn ABC"
}

Reviews collection:
{
  "_id": "traveloka_9000000796940_126228038",
  "source_hotel_id": "9000000796940",
  "review_text": "Phòng rộng rãi, sạch sẽ...",
  "clean_text": "phòng rộng_rãi sạch_sẽ",
  "tokens": ["phòng", "rộng_rãi", "sạch_sẽ"],
  "category_tags": ["beach"],
  "descriptor_tags": ["cleanliness", "room_spacious"]
}
```

### Bước 5: Build Indexes
```
Files:
  - indexing/build_bm25_index.py
  - indexing/build_vector_index.py

Output:
  - indexes/bm25_index.pkl
    {
      "bm25": BM25Okapi object,
      "documents": [doc1, doc2, ...],
      "review_ids": ["id1", "id2", ...]
    }
  
  - indexes/vector_index.pkl
    {
      "embeddings": np.array([[0.1, 0.2, ...], ...]),
      "documents": [doc1, doc2, ...],
      "review_ids": ["id1", "id2", ...],
      "model_name": "multi-qa-MiniLM-L6-cos-v1"
    }
```

---

## 3. ONLINE PIPELINE (Chạy khi user search)

### Luồng tìm kiếm chi tiết:

```
User Query: "khách sạn gần biển giá rẻ ở Vũng Tàu"
              ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 1: Query Understanding                         │
│ File: retrieval/query_understanding.py              │
│                                                     │
│ - Detect location: "Vũng Tàu"                       │
│ - Detect categories: ["beach", "budget"]            │
│ - Core tokens: ["khách", "sạn", "biển", "giá", "rẻ"]│
│ - Descriptor tokens: []                             │
│ - Expanded tokens: [...] (synonyms)                 │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 2: FlashText Tag Extraction                    │
│ File: retrieval/query_tag_extractor.py              │
│                                                     │
│ - KeywordProcessor.extract_keywords(query)          │
│ - Found: ["gần biển", "giá rẻ"]                     │
│ - Map to tags: ["beach", "budget"]                  │
│ - Category tags: ["beach", "budget"]                │
│ - Descriptor tags: []                               │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 3: Candidate Filtering (_build_candidate_mask) │
│                                                     │
│ - Location filter: "Vũng Tàu" match → True/False    │
│ - Category filter: "beach" HOẶC "budget" → True/False│
│ - Output: mask = [True, False, True, ...]           │
│ - Kết quả: ~5,000 reviews (từ 50,000)               │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 4: BM25 Retrieval                              │
│                                                     │
│ - search_tokens = expanded_tokens                   │
│ - bm25_scores = bm25.get_scores(search_tokens)      │
│ - Áp dụng mask: scores * mask                       │
│ - Normalize: scores / max_score                     │
│ - Top 4000 reviews có score > 0                     │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 5: Vector Retrieval                            │
│                                                     │
│ - Encode query → q_vec (384 dimensions)             │
│ - Cosine similarity: dot(embeddings, q_vec)         │
│ - Áp dụng mask                                      │
│ - Normalize                                         │
│ - Top 4000 reviews có score > 0                     │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 6: Hybrid Scoring                              │
│                                                     │
│ - hybrid = 0.4 * vector + 0.6 * bm25                │
│ - review_scores = {review_id: {doc, scores}}        │
│ - Filter: chỉ giữ reviews có hybrid_score > 0       │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 7: Negative Tag Filter                         │
│                                                     │
│ - Nếu query có "không sạch" → required_tags = {"cleanliness"}
│ - Loại reviews có "!cleanliness" trong descriptor_tags
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 8: Review Aggregation (Group by Hotel)         │
│                                                     │
│ - Group reviews theo source_hotel_id                │
│ - Mỗi hotel: sort reviews theo hybrid_score giảm    │
│ - Lấy top 5 reviews                                 │
│ - hotel_score = sum(top 5 hybrid_scores)            │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 9: Descriptor Filter                           │
│                                                     │
│ - Nếu query có descriptor tokens                    │
│ - Kiểm tra xem top reviews có chứa descriptors không│
│ - strict_descriptor_filter=True → loại nếu không match
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 10: Location Boosting                          │
│                                                     │
│ - Nếu query có location VÀ review match location    │
│ - hotel_score *= 1.8                                │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 11: Accommodation Type Boosting                │
│                                                     │
│ - Query có "resort" → tìm hotels có types=["resort"]│
│ - Match → hotel_score *= 1.2                        │
│ - Mismatch → hotel_score *= 0.7                     │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ BƯỚC 12: Sort + Return Top K                        │
│                                                     │
│ - Sort hotels theo hybrid_score giảm dần            │
│ - Nếu query có location → chỉ giữ kết quả match     │
│ - Return top_k results                              │
└─────────────────────────────────────────────────────┘
```

---

## 4. CẤU TRÚC DỮ LIỆU

### Review Document (sau preprocessing):
```json
{
  "_id": "review_id",
  "source_hotel_id": "hotel_id",
  "review_text": "Nội dung gốc",
  "clean_text": "Đã clean + normalize",
  "tokens": ["token1", "token2"],
  "category_tags": ["beach", "budget"],
  "descriptor_tags": ["cleanliness", "!room_spacious"],
  "rating": "4.2"
}
```

### Kết quả Search:
```json
{
  "source": "traveloka",
  "source_hotel_id": "hotel_id",
  "hotel_name": "Khách sạn ABC",
  "location": "Vũng Tàu",
  "rating": "4.2",
  "review_count": 15,
  "hybrid_score": 3.45,
  "vector_score": 0.78,
  "bm25_score": 0.65,
  "location_matched": true,
  "descriptor_matched": true,
  "top_reviews": ["review1", "review2", "review3"],
  "debug_info": {
    "score_after_aggregation": 2.1,
    "score_after_location_boost": 3.78,
    "score_final": 3.45
  }
}
```

---

## 5. TÓM TẮT CÁC THÀNH PHẦN

| Thành phần | File | Chức năng |
|------------|------|-----------|
| Crawler | crawler/traveloka_crawler.py | Crawl reviews từ web |
| Preprocessing | main.py | Clean, tokenize, remove stopwords |
| Tagger | preprocessing/review_tagger.py | Gán nhãn cảm xúc |
| Rules | config/review_tag_rules.yaml | Từ vựng cho tags |
| FlashText | flashtext library | Keyword matching nhanh |
| PhoBERT | wonrax/phobert-base-vietnamese-sentiment | Sentiment analysis |
| MongoDB | database/data_loader.py | Lưu trữ data |
| BM25 Index | indexing/build_bm25_index.py | Build BM25 index |
| Vector Index | indexing/build_vector_index.py | Build sentence embeddings |
| Query Understanding | retrieval/query_understanding.py | Phân tích query |
| Query Tag Extractor | retrieval/query_tag_extractor.py | FlashText extract tags |
| Search Engine | retrieval/search_engine.py | Hybrid search + ranking |