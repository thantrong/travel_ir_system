# memory_project_fix.md

## Mục tiêu
Ghi nhớ các điểm cần khắc phục để tối ưu và dễ bảo trì hơn trong `travel_ir_system`, chưa chỉnh code ở bước này.

## Các vấn đề cần sửa sau

### 1) `retrieval/search_engine.py` đang ôm quá nhiều trách nhiệm
- Gộp query understanding, candidate filtering, BM25 scoring, vector scoring, aggregation, boosting và formatting trong một file.
- Khó test, khó debug, khó mở rộng.
- Nên tách thành các module/hàm riêng: candidate filtering, ranking, aggregation, boosting, format output.

### 2) Model SentenceTransformer bị load lại mỗi lần search
- `encode_query()` tạo mới model mỗi lần gọi.
- Đây là điểm nghẽn hiệu năng lớn khi dùng GUI hoặc search liên tục.
- Nên cache model hoặc dùng singleton/lazy loader.

### 3) Trùng logic join dữ liệu ở indexing
- `indexing/build_bm25_index.py` và `indexing/build_vector_index.py` đều tự fetch `places` + `reviews` rồi join lại.
- Dễ lệch schema, khó bảo trì.
- Nên tách hàm dùng chung để fetch/join dữ liệu.

### 4) Nhiều bước infer metadata chạy runtime
- `_infer_doc_categories()`, `location_matched()`, `_descriptor_supported_by_reviews()` đều quét văn bản lúc query.
- Nếu corpus lớn sẽ chậm.
- Nên precompute category tags / location tags / descriptor tags khi build index.

### 5) Đồng bộ BM25 và vector index còn phụ thuộc thứ tự document
- Hiện đang giả định thứ tự docs giữa 2 index giống nhau.
- Giả định này dễ vỡ nếu pipeline đổi.
- Nên lưu mapping theo `review_id` rõ ràng.

### 6) Upsert MongoDB từng record một
- `database/data_loader.py` dùng `update_one` trong vòng lặp.
- Với dữ liệu lớn sẽ chậm.
- Nên cân nhắc `bulk_write` hoặc batch insert/upsert.

### 7) Descriptor / POI rules còn hard-coded
- Logic lọc descriptor, negative patterns, category hints đang nằm trực tiếp trong code.
- Muốn mở rộng phải sửa code.
- Nên đưa phần rule này sang config.

### 8) `strict_descriptor_filter=True` có thể làm giảm recall
- Hợp lệ cho precision nhưng dễ loại mất kết quả tốt nếu query mơ hồ.
- Nên cân nhắc chuyển sang soft-penalty hoặc chỉ áp dụng cho query rõ ràng.

## Ưu tiên khi sửa
1. Cache model SentenceTransformer.
2. Tách `search_engine.py` thành các khối nhỏ hơn.
3. Precompute metadata lúc build index.
4. Làm mapping `review_id` rõ ràng cho BM25/vector.
5. Tối ưu MongoDB upsert theo batch.
