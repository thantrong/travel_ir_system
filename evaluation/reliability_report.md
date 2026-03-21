# IR Reliability Report

## 1. Nguồn nhãn đánh giá
- Báo cáo này dùng nhãn thủ công từ `evaluation/annotation_pool.tsv` (cột `relevance`).

## 2. Kết quả tổng quan theo model
- **bm25**: P@5=0.768, P@10=0.724, R@10=0.721, MAP=0.677, nDCG@10=0.814
- **vector**: P@5=0.736, P@10=0.666, R@10=0.636, MAP=0.581, nDCG@10=0.732
- **hybrid**: P@5=0.764, P@10=0.714, R@10=0.728, MAP=0.683, nDCG@10=0.826

## 3. Đánh giá ngưỡng tin cậy (Hybrid)
- Precision@10 > 0.6: Đạt (0.714)
- nDCG@10 > 0.7: Đạt (0.826)
- MAP > 0.5: Đạt (0.683)
- Kết luận: **Có thể deploy có kiểm soát**
