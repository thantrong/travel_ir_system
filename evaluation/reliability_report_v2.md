# IR Reliability Report

## Canonical inputs
- Query set: `data/evaluation/test_queries_200_bucketed.json`
- Annotation pool: `evaluation/annotation_pool_bucketed_v2.csv`

## Overall results
- **bm25**: P@5=0.480, P@10=0.413, R@10=0.238, MAP=0.220, nDCG@10=0.419, strongP@10=0.151, strongR@10=0.191
- **vector**: P@5=0.588, P@10=0.506, R@10=0.368, MAP=0.347, nDCG@10=0.559, strongP@10=0.150, strongR@10=0.240
- **hybrid**: P@5=0.595, P@10=0.521, R@10=0.378, MAP=0.352, nDCG@10=0.557, strongP@10=0.161, strongR@10=0.245

## Bucket results
### bucket_1_short — short_1_2_attributes (n=40)
- Description: Truy vấn ngắn, thường chỉ 1-2 thuộc tính, ví dụ loại chỗ ở + địa điểm hoặc thêm 1 tiêu chí cơ bản.
- bm25: P@10=0.443, MAP=0.244, nDCG@10=0.436, strongP@10=0.207, strongR@10=0.191
- vector: P@10=0.497, MAP=0.324, nDCG@10=0.546, strongP@10=0.242, strongR@10=0.280
- hybrid: P@10=0.502, MAP=0.326, nDCG@10=0.524, strongP@10=0.225, strongR@10=0.269
- winners: P@10=hybrid, MAP=hybrid, nDCG@10=vector, strongP@10=vector, strongR@10=vector

### bucket_2_long_context — long_context_rich (n=40)
- Description: Mô tả dài, có ngữ cảnh, nhiều điều kiện đồng thời như nhóm đi, ngân sách, tiện ích, mục đích chuyến đi.
- bm25: P@10=0.492, MAP=0.224, nDCG@10=0.455, strongP@10=0.225, strongR@10=0.275
- vector: P@10=0.465, MAP=0.214, nDCG@10=0.427, strongP@10=0.185, strongR@10=0.218
- hybrid: P@10=0.508, MAP=0.236, nDCG@10=0.457, strongP@10=0.212, strongR@10=0.262
- winners: P@10=hybrid, MAP=hybrid, nDCG@10=hybrid, strongP@10=bm25, strongR@10=bm25

### bucket_3_geo_diverse — geo_diverse_priority_minor_provinces (n=40)
- Description: Đa dạng vị trí, ưu tiên tỉnh/thành ít phổ biến hoặc ngoài các điểm du lịch lớn để kiểm tra khả năng hiểu địa điểm hiếm.
- bm25: P@10=0.217, MAP=0.149, nDCG@10=0.276, strongP@10=0.020, strongR@10=0.072
- vector: P@10=0.405, MAP=0.461, nDCG@10=0.581, strongP@10=0.045, strongR@10=0.212
- hybrid: P@10=0.378, MAP=0.432, nDCG@10=0.523, strongP@10=0.040, strongR@10=0.162
- winners: P@10=vector, MAP=vector, nDCG@10=vector, strongP@10=vector, strongR@10=vector

### bucket_4_natural_semantics — natural_language_semantic_queries (n=40)
- Description: Câu truy vấn gần ngôn ngữ tự nhiên, diễn đạt mục tiêu/ngữ nghĩa thay vì chỉ nêu từ khóa.
- bm25: P@10=0.520, MAP=0.290, nDCG@10=0.540, strongP@10=0.122, strongR@10=0.158
- vector: P@10=0.550, MAP=0.348, nDCG@10=0.589, strongP@10=0.113, strongR@10=0.182
- hybrid: P@10=0.568, MAP=0.362, nDCG@10=0.605, strongP@10=0.125, strongR@10=0.207
- winners: P@10=hybrid, MAP=hybrid, nDCG@10=hybrid, strongP@10=hybrid, strongR@10=hybrid

### bucket_5_random_mix — random_mix_robustness (n=40)
- Description: Các truy vấn hỗn hợp/ngẫu nhiên, có thể ngắn, mơ hồ, thiếu thuộc tính hoặc pha nhiều tín hiệu khác nhau để kiểm tra độ bền của mô hình.
- bm25: P@10=0.395, MAP=0.194, nDCG@10=0.390, strongP@10=0.180, strongR@10=0.259
- vector: P@10=0.610, MAP=0.390, nDCG@10=0.650, strongP@10=0.165, strongR@10=0.308
- hybrid: P@10=0.652, MAP=0.402, nDCG@10=0.674, strongP@10=0.203, strongR@10=0.322
- winners: P@10=hybrid, MAP=hybrid, nDCG@10=hybrid, strongP@10=hybrid, strongR@10=hybrid

## Verdict
- Precision@10 > 0.6: Không đạt (0.521)
- nDCG@10 > 0.7: Không đạt (0.557)
- MAP > 0.5: Không đạt (0.352)
- Conclusion: **Chưa nên deploy production**

## Output files
- `evaluation/qrels_v2.tsv`
- `evaluation/runs_v2/run_bm25.tsv`
- `evaluation/runs_v2/run_vector.tsv`
- `evaluation/runs_v2/run_hybrid.tsv`