# Phase 0 Audit Report

## Scope

- Architecture and data flow audit (`raw -> clean -> Mongo -> index -> retrieval -> UI`)
- NLP and sentiment pipeline review (aspect extraction + PhoBERT inference)
- Runtime audit for long-running workloads (RAM/latency/device handling)
- Baseline benchmark outputs from `evaluation/baseline_benchmark.py`

## P0 Findings

1. **Credential leakage risk**  
   - `config/config.yaml` contained a committed MongoDB URI credential.  
   - Fix applied: URI removed from config default, runtime reads from `MONGODB_URI`.

2. **PhoBERT CUDA inference dtype bug**  
   - `preprocessing/review_tagger.py` converted tokenizer tensors with `.half()`, including `input_ids`.  
   - This can crash or produce undefined behavior with Transformer embeddings.  
   - Fix applied: keep input tensors in native integer dtype, only use autocast/model fp16 on CUDA.

## P1 Findings

1. **Collection name coupling**  
   - Multiple modules hardcoded `places`/`reviews`, bypassing config.  
   - Fix applied: central `get_collection_names()` and use across loader/index builders.

2. **Search runtime inefficiency**  
   - Index payload loaded from disk each query.  
   - Fix applied: cache index payload by `(path, mtime)` in `retrieval/search_engine.py`.

3. **Score explain mismatch**  
   - Type mismatch penalty uses `0.85` but debug reverse path used `0.7`.  
   - Fix applied: debug reverse aligned to `0.85`.

## P2 Findings

1. **Batch preprocessing memory growth and O(n^2) file writes**  
   - Full JSON rewrite each batch in `main.py`.  
   - Fix applied: JSONL append output + explicit GC and CUDA cache cleanup.

2. **Cross-platform runtime gap**  
   - No explicit MPS handling for Apple Silicon.  
   - Fix applied: `DEVICE_MODE` resolver with `cuda -> mps -> cpu`.

## Applied Architecture Adjustments

- Added minimal backend API layer:
  - `api/app.py`
  - `api/service.py`
  - `api/schemas.py`
- Endpoint pattern: `Route -> Service -> IR Adapter`.
- Minimal endpoints:
  - `GET /health`
  - `GET /ready`
  - `POST /search`
  - `GET /hotels/{hotel_id}`
  - `GET /suggestions`
  - `GET /metrics`

## Safety Gates (Pass/Fail)

Any change must pass these conditions before merge:

1. `NDCG@10` not below baseline by more than `0.02`.
2. `MRR` not below baseline by more than `0.02`.
3. `Recall@20` not below baseline by more than `0.03`.
4. `p95_latency_ms` not worse than baseline by more than `25%`.
5. `max_rss_mb` not worse than baseline by more than `20%`.
6. `no_result_rate` not increase by more than `0.03`.

## Roadmap (Execution Order)

1. Stabilize and benchmark (completed in this phase).
2. Refactor IR core in small slices:
   - sentiment context policy
   - multi-source ingest schema normalization
   - model lifecycle optimization
3. Build and harden endpoint layer (minimal API first, then expand).
4. Add web frontend only after API and IR KPI gates are stable.

## Rollback Strategy

- Keep previous index artifacts (`bm25_index.pkl`, `vector_index.pkl`) by timestamp.
- Keep previous config and lexicon snapshots.
- Use feature flags for ranking behavior changes where possible.
- If any safety gate fails, restore last green baseline artifacts/config.

