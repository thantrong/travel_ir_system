# Báo cáo Kiến trúc và Luồng Xử Lý Mô Hình Information Retrieval (IR)
*Hệ thống Tìm kiếm Khách sạn dựa trên Đánh giá (Review-based Hotel Search Engine)*

---

## 1. MỤC TIÊU VÀ BỐI CẢNH
Hệ thống cũ gặp phải các vấn đề về sai lệch kết quả do lập chỉ mục theo cấp độ khách sạn (gộp tất cả review thành một tài liệu khổng lồ), dẫn đến hiện tượng nhiễu từ khoá (length bias) và không hiểu đúng ý định truy vấn tự nhiên. 
**Giải pháp mới** thực hiện thiết kế lại toàn bộ luồng IR, chuyển sang **Review-Level Indexing**, tăng cường xử lý tiếng Việt và xếp hạng dựa trên tổng điểm của các đánh giá tiêu biểu nhất.

## 2. KIẾN TRÚC TỔNG THỂ CỦA HỆ THỐNG
Mô hình IR mới được phân nhỏ thành 5 module chính độc lập nhưng liên kết chặt chẽ với nhau:

1. **Query Understanding** (`retrieval/query_understanding.py`)
2. **Indexing (Review-Level)** (`indexing/build_bm25_index.py`, `indexing/build_vector_index.py`)
3. **Retrieval & Hybrid Search** (`retrieval/search_engine.py`)
4. **Aggregation & Ranking** (`retrieval/search_engine.py`)
5. **Evaluation Pipeline** (`evaluation/evaluate_pipeline.py`)

---

## 3. CHI TIẾT CÁC LUỒNG XỬ LÝ (PROCESSING PIPELINE)

### Bước 1: Tiền xử lý Truy vấn Tự nhiên (Query Understanding)
**Nhiệm vụ:** Biến 1 câu nói tự nhiên thành tín hiệu tìm kiếm có cấu trúc.
- **Lọc Stopwords:** Bỏ các từ vô nghĩa mang tính hội thoại (vd: "tôi", "muốn", "tìm", "khách sạn", "địa điểm").
- **Nhận diện Vị trí (Location Extraction):** 
  - Trích xuất N-gram (Trigram, Bigram, Unigram) từ câu truy vấn và đối chiếu với danh mục địa lý Việt Nam.
  - Hỗ trợ các tên gọi thay thế (vd: `Hồ Chí Minh` $\to$ `tp hcm`, `Sài Gòn`).
- **Phân tách Từ khóa (Descriptor Extraction):** Tách các nhu cầu tĩnh (vd: `"gần biển"`, `"view đẹp"`, `"giá rẻ"`).
- **Mở rộng Từ vựng (Synonym Expansion):** Bổ sung các từ đồng nghĩa (vd: "rẻ" $\to$ ["rẻ", "bình dân", "tiết kiệm"]).

### Bước 2: Lập chỉ mục cấp độ Đánh giá (Review-Level Indexing)
**Nhiệm vụ:** Chia nhỏ dữ liệu, không lập chỉ mục cả khách sạn mà lập chỉ mục **từng bài review độc lập**.
- Thay vì một văn bản 10.000 chữ, hệ thống giữ nguyên các review ngắn (30-200 chữ).
- **Trọng số trường ảo (Field Weighting):** Để không mất đi ngữ cảnh khách sạn, dữ liệu được "độn" thêm vào Review trước khi đưa vào Vector/BM25:
  - Tên khách sạn: x3 lần
  - Vị trí/Khu vực: x2 lần
  - Nội dung Review: x1 lần
- **Đầu ra:** 2 file tạo thành hệ thống Index khổng lồ: `bm25_index.pkl` (từ vựng) và `vector_index.pkl` (ngữ nghĩa).

### Bước 3: Truy hồi Kết hợp (Hybrid Retrieval)
**Nhiệm vụ:** Tìm ra hàng nghìn *Review* có khả năng khớp với truy vấn nhất.
- Lấy câu hỏi sinh Vector (Sentence-Transformers) $\to$ Cosine Similarity với Vector Index để ra `vector_score`.
- Lấy từ khóa sinh chuỗi token $\to$ BM25 model để ra `bm25_score`.
- Trộn lẫn với tỷ lệ chuẩn: **`hybrid_score_review = 0.6 * vector_score + 0.4 * bm25_score`**.

### Bước 4: Gom nhóm và Xếp hạng Khách sạn (Review Aggregation & Ranking)
**Nhiệm vụ:** Chuyển điểm của các "Review" thành điểm của "Khách sạn".
- **Chống Bias Spam:** Thuật toán duyệt qua list Review đã chấm điểm ở trên, tự động phân nhóm (Group By) chúng theo `hotel_id`.
- Sắp xếp thứ tự các Review trong cùng 1 khách sạn để lấy ra **Top 5 Review xuất sắc nhất**.
- **Tính điểm ròng:** Điểm xếp hạng Khách sạn = Tổng hợp điểm của 5 Review đó ($\sum Top\_5$). Cách này triệt tiêu hiện tượng 1 review rác làm hỏng cả chuỗi hoặc khách sạn nhiều review rác ăn gian được hạng.

### Bước 5: Phạt/Thưởng Cục bộ Hành vi (Location Boosting)
- Kiểm tra chéo *Location* trích xuất ở Bước 1 và *Location* của khách sạn hiện tại. Nếu khớp, nhân tổng điểm khách sạn với hệ số **x1.8**. Khách sạn đúng ý định vùng miền sẽ vọt lên Top đầu.

### Bước 6: Trích xuất Dữ liệu Giao diện (Review Extraction / Summarization)
- Với mỗi khách sạn xuất hiện trên Top Query, trích xuất chính những Raw Reviews vừa đóng góp điểm lớn nhất để chứng minh lý do khách sạn này được lên Top.
- *Ghi chú:* Module `summarization/` sử dụng TextRank (Extractive TF-IDF) cũng được chuẩn bị sẵn nếu muốn thu gọn 5 review thành 1 đoạn văn tóm tắt.

### Bước 7: Đánh giá Chất lượng Tự động (Evaluation Metrics)
- `evaluate_pipeline.py` hoạt động ở nền tảng Backend, tự động ném các truy vấn khó vào hệ thống và so sánh kết quả tự trả về để tính độ chính xác.
- Tính 4 bộ thông số chính xác IR: **Precision@10, Recall@10, MAP (Mean Average Precision), và nDCG**.

---

## 4. ƯU ĐIỂM CỦA KIẾN TRÚC MỚI
1. **Tránh nhiễu dữ liệu:** Tách rời Review giúp chống lại việc một khách sạn dài hàng ngàn Review nhưng chỉ có 1 câu chứa từ khóa mà vẫn lọt Top.
2. **Minh bạch hóa Kết quả:** Việc giải thích lý do Khách sạn đạt Top được thể hiện rõ thông qua chính các Top Review mà hệ thống show ra dưới mỗi kết quả.
3. **Hiểu ngữ cảnh ngữ nghĩa Tiếng Việt:** Phân tách rõ ràng giữa Vị trí địa lý và các Mô tả dịch vụ, giúp hệ thống không nhầm "Khách sạn Hà Nội" ở "Hồ Chí Minh" thành đích đến Hà Nội.
4. **Cấu trúc Thư mục Clean Architecture:** Chia rẽ các Package: Database, Preprocessing, Query Understanding, Indexing, Retrieval, Evaluation, Summarization giúp dễ dàng Scalability sau này.
