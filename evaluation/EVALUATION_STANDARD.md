# Evaluation Standard for travel_ir_system

## Canonical files
- Query set: `data/evaluation/test_queries.json`
- Annotation pool: `evaluation/annotation_pool_v3.csv` / `.json` / `.xlsx`
- Qrels: `evaluation/qrels.tsv`
- Run files:
  - `evaluation/runs/run_bm25.tsv`
  - `evaluation/runs/run_vector.tsv`
  - `evaluation/runs/run_hybrid.tsv`
- Metrics summary: `evaluation_metrics.json`
- Reliability report: `evaluation/reliability_report.md`

## Naming rules
- Use lowercase, snake_case file names.
- Use one version suffix only, e.g. `*_v3.*`.
- Do not create mixed-case duplicates like `V3`, `v03`, or `final2`.
- Annotation pools must be the single source of truth for labels used in evaluation.

## Record schema
Each annotation row should contain:
- `query_id`
- `query`
- `hotel_id`
- `hotel_name`
- `location`
- `relevance`
- `note`
- `query_location`
- `query_categories`
- `query_descriptors`
- model-specific rank/score columns when available

## Relevance policy
- `1` = strong or sufficient relevance according to query intent and location/descriptor/category evidence.
- `0` = not relevant or insufficient signal.
- Avoid labeling everything `1`; keep the pool discriminative so BM25, vector, and hybrid can be compared fairly.

## Evaluation policy
- BM25, vector, and hybrid must be scored on the same query set and same annotation pool.
- Hybrid is the primary model, but BM25 and vector must remain in the report for baseline comparison.
- If a query is ambiguous, label conservatively rather than forcing relevance.
