# Hướng dẫn sử dụng hệ thống Travel IR

Tài liệu này mô tả đầy đủ cách vận hành hệ thống tìm kiếm khách sạn, kiến trúc xử lý, nơi lưu trữ dữ liệu, schema MongoDB và luồng xử lý end-to-end.

## 1. Mục tiêu hệ thống

Hệ thống nhận truy vấn tiếng Việt và trả về top khách sạn phù hợp theo cơ chế Hybrid Retrieval:

- BM25 (từ khoá)
- Vector similarity (ngữ nghĩa)
- Query understanding (location, descriptor, mở rộng từ đồng nghĩa)
- Review-level scoring rồi aggregate lên hotel-level

## 2. Chuẩn bị môi trường

Chạy trong thư mục dự án:

```bash
pip install -r requirements.txt
```

Lưu ý:

- Cần cấu hình MongoDB trong file `config/config.yaml`.
- Không commit credentials thật lên git.

## 3. Cấu trúc lưu trữ dữ liệu

### 3.1. Dữ liệu file trong project

- Raw dữ liệu: `data/raw/`
- Dữ liệu đã xử lý: `data/processed/`
- Index tìm kiếm: `data/index/`
- Dữ liệu đánh giá: `data/evaluation/test_queries.json`
- Báo cáo đánh giá: `evaluation_metrics.json`

Các file index chính:

- `data/index/bm25_index.pkl`
- `data/index/vector_index.pkl`

### 3.2. Lưu trữ MongoDB

Hệ thống lưu dữ liệu vào database MongoDB (mặc định: `travel_ir`) với các collection chính:

- `places`
- `reviews`

Trong config có thể khai báo thêm `queries`, `evaluation_metrics`, nhưng pipeline hiện tại tập trung vào `places` và `reviews`.

## 4. Schema dữ liệu

### 4.1. Collection `places`

Mỗi document đại diện một khách sạn theo khoá chính tổng hợp:

- `_id`: `<source>_<source_hotel_id>`
- `name`: tên khách sạn
- `type`: `hotel`
- `location`: tỉnh/thành phố (đã chuẩn hoá nhãn khu vực)
- `rating`: điểm trung bình ở cấp khách sạn
- `source`: nguồn dữ liệu
- `source_hotel_id`: id khách sạn từ nguồn gốc

Ví dụ:

```json
{
  "_id": "tripadvisor_14775963",
  "name": "Dao Ngoc Hotel",
  "type": "hotel",
  "location": "Phú Quốc",
  "rating": "4.2",
  "source": "tripadvisor",
  "source_hotel_id": "14775963"
}
```

### 4.2. Collection `reviews`

Mỗi document đại diện một review đã qua preprocessing:

- `_id`: `review_id`
- `review_id`: id review chuẩn hoá
- `source_review_id`: id review từ nguồn gốc
- `source_hotel_id`: id khách sạn từ nguồn
- `review_text`: nội dung gốc
- `clean_text`: nội dung đã làm sạch
- `tokens`: token sau chuẩn hoá và bỏ stopwords
- `review_date`: ngày review (nếu có)
- `review_rating`: điểm review riêng
- `source`: nguồn dữ liệu

Ví dụ:

```json
{
  "_id": "tripadvisor_14775963_654797953",
  "review_id": "tripadvisor_14775963_654797953",
  "source_review_id": "654797953",
  "source_hotel_id": "14775963",
  "review_text": "Khách sạn sạch sẽ, nhân viên thân thiện",
  "clean_text": "khách sạn sạch sẽ nhân viên thân thiện",
  "tokens": ["khách_sạn", "sạch_sẽ", "nhân_viên", "thân_thiện"],
  "review_rating": "4.7",
  "source": "tripadvisor"
}
```

## 5. Luồng xử lý tổng thể

```text
Nguồn dữ liệu (Traveloka crawler / CSV-JSON-JSONL)
    -> Chuẩn hoá schema review
    -> Preprocessing (clean, spam filter, language filter, normalize, tokenize, remove stopwords)
    -> Load vào MongoDB (places, reviews)
    -> Build BM25 index + Vector index từ MongoDB
    -> Nhận query từ UI/CLI
    -> Query understanding (location, descriptor, synonyms, category)
    -> Hybrid scoring ở review-level
    -> Aggregate review thành hotel-level + location boosting
    -> Trả top K khách sạn + top review minh hoạ
```

## 6. Quy trình chạy chuẩn end-to-end

### Bước 1. Nạp dữ liệu

Có 2 hướng:

1. Từ crawler Traveloka:

```bash
python crawler/traveloka_crawler.py
```

1. Từ dataset ngoài theo config:

```bash
python dataset_pipeline.py --load-mongo
```

Nguồn và mapping khai báo tại `config/dataset_sources.yaml`.

Tùy chọn hữu ích:

```bash
python dataset_pipeline.py --source-name tripadvisor --limit-per-source 500 --load-mongo
python dataset_pipeline.py --only-vietnamese --load-mongo
```

### Bước 2. Build index

```bash
python indexing/build_bm25_index.py
python indexing/build_vector_index.py
```

Kết quả sinh tại `data/index/`.

### Bước 3. Chạy giao diện tìm kiếm

```bash
streamlit run app_gui.py
```

UI cho phép chỉnh trực tiếp:

- `vector_weight` (mặc định 0.6)
- `bm25_weight` (mặc định 0.4)
- `location_boost_factor` (mặc định 1.8)
- `top_k`

### Bước 4. Debug nhanh pipeline truy hồi

```bash
python scripts/debug_search.py
```

Script sẽ in:

- Core tokens
- Expanded tokens
- Detected location
- Top kết quả hotel-level và điểm số hybrid/vector/BM25

## 7. Cách hoạt động của truy hồi Hybrid

Trong `retrieval/search_engine.py`:

1. Parse query bằng `understand_query`
2. Lọc candidate theo location/category (nếu có)
3. Tính điểm BM25 và vector ở review-level
4. Chuẩn hoá điểm về cùng thang
5. Trộn điểm: `hybrid = vector_weight * vector_score + bm25_weight * bm25_score`
6. Group review theo `source_hotel_id`
7. Cộng điểm top review trong mỗi khách sạn để ra hotel score
8. Boost theo location nếu khớp
9. Trả top K khách sạn

Điểm quan trọng:

- Index lưu ở review-level để giữ ngữ cảnh chi tiết.
- Kết quả cuối cùng trả về ở hotel-level để phù hợp trải nghiệm tìm khách sạn.

## 8. Đánh giá mô hình

### Bước 1. Tạo dummy relevance (nếu chưa có nhãn)

```bash
python scripts/generate_dummy_labels.py
```

### Bước 2. Chạy evaluation

```bash
python evaluation/evaluate_pipeline.py
```

Các metric được xuất ra `evaluation_metrics.json` gồm:

- Mean_P@10
- Mean_R@10
- MAP
- Mean_nDCG

Lưu ý: nếu query test chưa có ground-truth chuẩn, pipeline có cơ chế fallback relevance giả định để kiểm thử kỹ thuật.

## 9. Khi nào cần build lại index

Bắt buộc build lại `bm25_index.pkl` và `vector_index.pkl` khi:

- Có dữ liệu review mới vào MongoDB
- Thay đổi logic preprocessing/tokenization/stopwords
- Thay đổi biểu diễn text đầu vào cho vector index

## 10. Checklist vận hành nhanh

```text
[ ] Kiểm tra config MongoDB
[ ] Nạp dữ liệu (crawler hoặc dataset_pipeline)
[ ] Build BM25 index
[ ] Build Vector index
[ ] Chạy UI Streamlit
[ ] Test truy vấn mẫu
[ ] Chạy evaluation và lưu metrics
```
