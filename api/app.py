import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from api.schemas import SearchRequest, SuggestionResponse
from api.service import INDEX_DIR, search_hotels

app = FastAPI(title="Travel IR API", version="0.1.0")
_METRICS = {
    "requests_total": 0,
    "search_requests": 0,
    "search_no_result": 0,
    "search_latency_ms": [],
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    bm25_ok = (INDEX_DIR / "bm25_index.pkl").exists()
    vec_ok = (INDEX_DIR / "vector_index.pkl").exists()
    return {"ready": bm25_ok and vec_ok, "index_dir": str(INDEX_DIR), "bm25": bm25_ok, "vector": vec_ok}


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    if payload.vector_weight + payload.bm25_weight == 0:
        raise HTTPException(status_code=400, detail="vector_weight + bm25_weight must be > 0")

    start = time.perf_counter()
    _METRICS["requests_total"] += 1
    _METRICS["search_requests"] += 1
    response = search_hotels(
        query=payload.query,
        top_k=payload.top_k,
        vector_weight=payload.vector_weight,
        bm25_weight=payload.bm25_weight,
        location_boost_factor=payload.location_boost_factor,
        explain=payload.explain,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    _METRICS["search_latency_ms"].append(latency_ms)
    _METRICS["search_latency_ms"] = _METRICS["search_latency_ms"][-500:]
    if response["count"] == 0:
        _METRICS["search_no_result"] += 1
    return response


@app.get("/hotels/{hotel_id}")
def hotel_detail(hotel_id: str) -> dict:
    # Reuse search results with the hotel_id filter in-memory for now (minimal endpoint stage).
    payload = search_hotels(query=hotel_id, top_k=20, explain=True)
    for row in payload["results"]:
        if str(row.get("source_hotel_id", "")).strip() == hotel_id:
            return row
    raise HTTPException(status_code=404, detail="Hotel not found")


@app.get("/suggestions", response_model=SuggestionResponse)
def suggestions(q: str = Query(..., min_length=1)) -> SuggestionResponse:
    # Lightweight local suggestions from query text.
    terms = [x.strip() for x in q.lower().split() if x.strip()]
    uniq = []
    for t in terms:
        if t not in uniq:
            uniq.append(t)
    candidates = [q]
    if uniq:
        candidates.extend([
            "khách sạn " + " ".join(uniq),
            "resort " + " ".join(uniq),
            "homestay " + " ".join(uniq),
        ])
    return SuggestionResponse(suggestions=candidates[:5])


@app.get("/metrics")
def metrics() -> dict:
    lat = _METRICS["search_latency_ms"]
    avg_latency = sum(lat) / len(lat) if lat else 0.0
    p95 = sorted(lat)[int(0.95 * (len(lat) - 1))] if len(lat) > 1 else (lat[0] if lat else 0.0)
    return {
        **_METRICS,
        "search_avg_latency_ms": round(avg_latency, 2),
        "search_p95_latency_ms": round(p95, 2),
    }

