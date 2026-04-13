from pathlib import Path
from typing import Any

from retrieval.search_engine import search_hybrid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
STOPWORDS_PATH = PROJECT_ROOT / "config" / "stopwords.txt"


def search_hotels(
    query: str,
    top_k: int = 10,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    location_boost_factor: float = 1.8,
    explain: bool = False,
) -> dict[str, Any]:
    results, qu = search_hybrid(
        query=query,
        index_dir=INDEX_DIR,
        stopwords_path=STOPWORDS_PATH,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        location_boost_factor=location_boost_factor,
    )
    if not explain:
        for row in results:
            row.pop("debug_info", None)

    return {
        "query": query,
        "query_understanding": {
            "detected_location": qu.detected_location,
            "detected_categories": qu.detected_categories,
            "descriptor_tokens": qu.descriptor_tokens,
            "expanded_tokens": qu.expanded_tokens,
        },
        "count": len(results),
        "results": results,
    }

