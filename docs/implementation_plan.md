# Kế hoạch triển khai Hệ thống truy hồi thông tin du lịch

Hệ thống truy hồi thông tin cho phép người dùng nhập câu hỏi bằng ngôn ngữ tự nhiên tiếng Việt để tìm kiếm các khách sạn phù hợp nhất và trả về Top 10 kết quả cùng với tóm tắt review.

## Architecture & Tech Stack (As-built)

- **Ngôn ngữ**: Python 3.x
- **Crawler Tool**: `playwright` (headless browser) + `playwright-stealth` (anti-bot) + `BeautifulSoup` (parser)
- **Nguồn dữ liệu**:
  - Traveloka (crawler Playwright)
  - Dataset ngoài (CSV/JSON/JSONL) qua `dataset_pipeline.py` + config nguồn
- **Cơ sở dữ liệu**: MongoDB Atlas (`travel_ir`)
- **NLP / Text Processing**: `underthesea` (hoặc `pyvi`), `langdetect`
- **Retrieval Engine**: `rank_bm25` (mô hình BM25)
- **Tóm tắt Review (Summarization)**: Rule-based Extractive Summarization (TextRank / TF-IDF)
- **Evaluation**: Các text metrics tiêu biểu

> **NGUYÊN TẮC THIẾT KẾ & TRIỂN KHAI**
>
> - Xây dựng từ dưới lên (Bottom-up).
> - Ưu tiên xây dựng module **Crawler** trước. Crawler phải hoạt động ổn định, có sample data hoàn chỉnh thì mới tiến hành làm NLP/Retrieval.
> - Bắt buộc Test và Review từng bước. Khâu trước hoàn thiện mới đi tiếp khâu sau.
> - Ưu tiên sửa, mở rộng trên code hiện tại; tránh viết lại module khi không cần thiết.

## Cấu trúc thư mục

```txt
travel_ir_system/
├── config/
│   ├── config.yaml                 # Cấu hình crawler params, DB connection
│   ├── cities.yaml                 # Danh sách city/province để crawl Traveloka
│   └── dataset_sources.yaml        # Cấu hình nhiều nguồn dataset ngoài
├── data/
│   ├── raw/                        # JSON/JSONL thô từ crawler
│   └── processed/                  # Dữ liệu đã clean + tokenize
├── crawler/
│   └── traveloka_crawler.py        # Playwright crawler cho Traveloka (API-first)
├── preprocessing/
│   ├── clean_text.py
│   ├── language_filter.py
│   └── remove_spam.py
├── nlp/
│   ├── tokenizer.py
│   ├── stopwords.py
│   └── normalization.py
├── database/
│   ├── mongo_connection.py
│   └── data_loader.py
├── geo_id_tool.py                  # Tool dò geo_id Traveloka cho cities.yaml
├── dataset_pipeline.py             # Pipeline nạp nhiều dataset ngoài về schema chung
├── indexing/
│   └── build_bm25_index.py
├── retrieval/
│   ├── query_processing.py
│   └── search_engine.py
├── summarization/
│   └── review_summary.py
├── evaluation/
│   └── metrics.py
├── api/
│   └── search_api.py
├── docs/
│   └── implementation_plan.md
├── chrome_profile/                 # Persistent browser profile (anti-bot)
└── main.py
```

## Pipeline chi tiết

### 1. Data Collection (Crawler) ✅ Đã triển khai và vận hành

**Nguồn**: Traveloka (API-first + fallback động).

**Pipeline Crawler hiện tại:**

```txt
Search Hotels by City (geo_id từ config/cities.yaml)
        ↓
Hotel List (infinite scroll, h3 tags)
        ↓  click hotel → new tab
Hotel Detail Page
        ↓  capture/replay API getReviews
Extract + filter: translationStatus=ORIGINAL, reviewOriginalText, tiếng Việt
Save checkpoint (.jsonl) + final (.json)
```

**Anti-bot measures:**

- `playwright-stealth` plugin
- Persistent browser context (giữ cookie)
- `ignore_default_args=["--enable-automation"]`
- Random delay + human-like scroll
- User-Agent Chrome 131

**Schema dữ liệu review hiện tại (chuẩn chung):**

| Trường | Mô tả | Ví dụ |
| --- | --- | --- |
| `review_id` | `<source>_<source_hotel_id>_<source_review_id>` | `tripadvisor_14775963_654797953` |
| `source_review_id` | ID review từ nguồn | `654797953` |
| `source_hotel_id` | ID hotel từ nguồn | `14775963` |
| `hotel_name` | Tên khách sạn | `Hana Riverside Villa` |
| `location` | Tỉnh/Thành phố | `Quảng Ngãi` |
| `rating` | Điểm trung bình (thang 5) | `4.2` |
| `review_text` | Nội dung review | `Nhân viên thân thiện...` |
| `review_date` | Ngày đánh giá | `2019-02-25` |
| `review_rating` | Điểm riêng review (thang 5) | `4.7` |
| `source` | Nguồn | `traveloka` |

> Lưu ý: đã bỏ các trường `place_id`, `review_origin`, `reviewer_name` khỏi schema chuẩn theo yêu cầu hiện tại.

**Danh sách city/province crawl:**

- Quản lý qua `config/cities.yaml` (không hardcode trong plan).
- Đã chuyển sang danh sách 34 đơn vị cấp tỉnh sau sáp nhập; bật/tắt bằng `enabled`.

### 2. Database Schema (MongoDB Atlas) ✅ Đã triển khai

**Collection: `places`**

```json
{
  "source": "tripadvisor",
  "source_hotel_id": "14775963",
  "name": "Dao Ngoc Hotel",
  "type": "hotel",
  "location": "Phú Quốc",
  "rating": 4.2
}
```

**Collection: `reviews`**

```json
{
  "review_id": "tripadvisor_14775963_654797953",
  "source_review_id": "654797953",
  "source_hotel_id": "14775963",
  "review_text": "Khách sạn sạch sẽ, nhân viên thân thiện",
  "clean_text": "khách sạn sạch sẽ nhân viên thân thiện",
  "tokens": ["khách_sạn", "sạch_sẽ", "nhân_viên", "thân_thiện"],
  "review_rating": 4.7,
  "source": "tripadvisor"
}
```

**Collection: `evaluation_metrics`**

```json
{
  "model": "BM25",
  "precision_at_10": 0.72,
  "recall_at_10": 0.65,
  "map": 0.68,
  "ndcg_at_10": 0.74
}
```

### 3. Pipeline Xử lý Dữ liệu & NLP ✅ Đã triển khai

- **Preprocessing:** `clean_text.py` → `language_filter.py` (chỉ giữ tiếng Việt) → `remove_spam.py`
- **NLP:** `normalization.py` (lowercase, remove punctuation) → `tokenizer.py` (word tokenization) → `stopwords.py` (loại bỏ stopword)
- Load vào MongoDB qua `data_loader.py`
- Pipeline ngoài crawler: `dataset_pipeline.py` + `config/dataset_sources.yaml` để nạp nhiều dataset cùng schema chuẩn

### 4. Indexing & Retrieval ⏳ Chưa triển khai

- BM25 index (`build_bm25_index.py`)
- Xử lý query tiếng Việt (`query_processing.py`)
- Top 10 kết quả (`search_engine.py`)

### 5. Summarization ⏳ Chưa triển khai

- Gom nhóm review theo địa điểm
- Tạo tóm tắt qua `review_summary.py`

### 6. Evaluation ⏳ Chưa triển khai

- **Metrics**: Precision@k, Recall@k, F1-score, MAP, nDCG@k
- **Ground Truth**: Query Set + Relevance Judgments
- Lưu kết quả vào `evaluation_metrics`

## Verification Plan

### Automated Tests

- [ ] Evaluation Model với Dataset Ground Truth nhỏ
- [ ] Unit test crawler parser
- [ ] Test pipeline Clean → NLP Tokenization

### Manual Verification

- [x] Chạy crawler giới hạn 1 city, 3 hotels
- [x] Kiểm tra dữ liệu review gốc tiếng Việt từ API
- [x] Kiểm tra pipeline dataset ngoài (Reviews.csv + hotel_coordinate.csv)
- [ ] Chạy search engine CLI với truy vấn tiếng Việt
- [ ] Đọc Evaluation Report

## Trạng thái tổng quan hiện tại

- **Đã xong:** Crawler API-first, chuẩn hóa schema review, pipeline preprocess/NLP, nạp dataset ngoài đa nguồn, upsert Mongo.
- **Đang làm:** Làm sạch cấu hình city/geo_id và ổn định chất lượng dữ liệu theo city.
- **Chưa làm:** BM25 indexing/retrieval, summarization, evaluation metrics end-to-end.
