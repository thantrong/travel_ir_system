# Bucket definitions for cross-bucket evaluation

Use `data/evaluation/test_queries_200_bucketed.json` as the labeled query set.

## Bucket 1 — short_1_2_attributes
- Intent: short queries with only 1–2 attributes.
- Typical form: accommodation type + location, or add one simple qualifier.
- What it tests:
  - keyword matching
  - location recognition
  - type recognition
- Model that is strong here usually has good lexical recall and light type/location boosting.

## Bucket 2 — long_context_rich
- Intent: long, context-rich queries with multiple constraints.
- Typical form: use-case + group size + budget + amenities + location.
- What it tests:
  - query understanding
  - constraint aggregation
  - relevance filtering under many conditions
- Model that is strong here usually has better semantic retrieval and better descriptor handling.

## Bucket 3 — geo_diverse_priority_minor_provinces
- Intent: location-heavy queries across diverse provinces/regions, especially less common places.
- What it tests:
  - location normalization
  - rare place-name coverage
  - robustness on geographically sparse signals
- Model that is strong here should handle tail locations without collapsing to major cities.

## Bucket 4 — natural_language_semantic_queries
- Intent: natural-language queries that express intent rather than keywords.
- What it tests:
  - semantic matching
  - intent interpretation
  - soft constraint handling
- Model that is strong here usually benefits from vector search and good query understanding.

## Bucket 5 — random_mix_robustness
- Intent: mixed and somewhat noisy queries.
- What it tests:
  - robustness to ambiguity
  - fallback behavior
  - stability under underspecified input
- Model that is strong here should degrade gracefully and avoid overfitting to one signal.

## Suggested reporting
For each model, report per-bucket metrics instead of only a global mean:
- MAP
- nDCG@10
- Precision@10
- Recall@10

Also report the winner per bucket so it is obvious which model is strongest for which query style.
