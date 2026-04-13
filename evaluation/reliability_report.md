# IR Reliability Report

## Canonical inputs
- Query set: `data/evaluation/test_queries_200_bucketed.json`
- Annotation pool: `data/evaluation/pool_results_labeled.csv`

## Overall results
- **bm25**: P@5=0.000, P@10=0.000, R@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- **vector**: P@5=0.000, P@10=0.000, R@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- **hybrid**: P@5=0.000, P@10=0.000, R@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000

## Bucket results
### bucket_1_short — short_1_2_attributes (n=40)
- Description: Short queries with 1?2 attributes.
- bm25: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- vector: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- hybrid: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- winners: P@10=bm25, MAP=bm25, nDCG@10=bm25, strongP@10=bm25, strongR@10=bm25

### bucket_2_long_context — long_context_rich (n=40)
- Description: Long, context-rich queries with multiple constraints.
- bm25: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- vector: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- hybrid: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- winners: P@10=bm25, MAP=bm25, nDCG@10=bm25, strongP@10=bm25, strongR@10=bm25

### bucket_3_geo_diverse — geo_diverse_priority_minor_provinces (n=40)
- Description: Geo-diverse normal queries with location emphasis.
- bm25: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- vector: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- hybrid: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- winners: P@10=bm25, MAP=bm25, nDCG@10=bm25, strongP@10=bm25, strongR@10=bm25

### bucket_4_natural_language_semantic — natural_language_semantic_queries (n=40)
- Description: Location-only queries across many provinces.
- bm25: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- vector: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- hybrid: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- winners: P@10=bm25, MAP=bm25, nDCG@10=bm25, strongP@10=bm25, strongR@10=bm25

### bucket_5_random_mix — random_mix_robustness (n=40)
- Description: Mixed and noisy robustness queries.
- bm25: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- vector: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- hybrid: P@10=0.000, MAP=0.000, nDCG@10=0.000, strongP@10=0.000, strongR@10=0.000
- winners: P@10=bm25, MAP=bm25, nDCG@10=bm25, strongP@10=bm25, strongR@10=bm25

## Verdict
- Precision@10 > 0.6: Không đạt (0.000)
- nDCG@10 > 0.7: Không đạt (0.000)
- MAP > 0.5: Không đạt (0.000)
- Conclusion: **Chưa nên deploy production**

## Output files
- `evaluation/qrels.tsv`
- `evaluation/runs/run_bm25.tsv`
- `evaluation/runs/run_vector.tsv`
- `evaluation/runs/run_hybrid.tsv`