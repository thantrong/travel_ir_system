# IR Reliability Report

## Canonical inputs
- Query set: `data/evaluation/test_queries.json`
- Annotation pool: `evaluation/annotation_pool_v3.csv`

## Overall results
- **bm25**: P@5=0.000, P@10=0.000, R@10=0.000, MAP=0.000, nDCG@10=0.000
- **vector**: P@5=0.000, P@10=0.000, R@10=0.000, MAP=0.000, nDCG@10=0.000
- **hybrid**: P@5=0.000, P@10=0.000, R@10=0.000, MAP=0.000, nDCG@10=0.000

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