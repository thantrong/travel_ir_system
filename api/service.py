from pathlib import Path
from typing import Any

from rag import answer_from_ir_results
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


def answer_with_rag(
    query: str,
    top_k_retrieval: int = 12,
    top_k_context: int = 6,
    max_citations: int = 4,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    location_boost_factor: float = 1.8,
    allow_fallback_to_ir: bool = True,
    explain: bool = False,
) -> dict[str, Any]:
    try:
        ir_payload = search_hotels(
            query=query,
            top_k=top_k_retrieval,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            location_boost_factor=location_boost_factor,
            explain=True,
        )
    except Exception:
        if not allow_fallback_to_ir:
            raise
        return {
            "mode": "ir_fallback",
            "query": query,
            "answer": "Khong the truy xuat retrieval luc nay. Thu lai sau.",
            "citations": [],
            "grounded": False,
            "fallback_used": True,
            "ir_count": 0,
            "ir_results": [],
            "query_understanding": {},
        }
    ir_results = ir_payload["results"]
    if not ir_results:
        return {
            "mode": "rag",
            "query": query,
            "answer": "Khong tim thay du lieu phu hop de tra loi.",
            "citations": [],
            "grounded": False,
            "fallback_used": False,
            "ir_count": 0,
        }

    try:
        rag_payload = answer_from_ir_results(
            query=query,
            ir_results=ir_results,
            top_k_context=top_k_context,
            max_citations=max_citations,
        )

        if not rag_payload.get("grounded", False):
            raise ValueError("Ungrounded RAG response.")

        response = {
            "mode": "rag",
            "query": query,
            "answer": rag_payload["answer"],
            "citations": rag_payload["citations"],
            "grounded": rag_payload["grounded"],
            "fallback_used": False,
            "ir_count": len(ir_results),
            "query_understanding": ir_payload["query_understanding"],
        }
        if explain:
            response["debug_info"] = {
                "context_items_used": rag_payload.get("context_items_used", 0),
                "top_k_retrieval": top_k_retrieval,
                "top_k_context": top_k_context,
            }
        return response
    except Exception:
        if not allow_fallback_to_ir:
            raise
        fallback = {
            "mode": "ir_fallback",
            "query": query,
            "answer": "He thong RAG tam thoi khong on dinh. Da fallback sang ket qua IR.",
            "citations": [],
            "grounded": False,
            "fallback_used": True,
            "ir_count": len(ir_results),
            "ir_results": ir_results[: min(5, len(ir_results))],
            "query_understanding": ir_payload["query_understanding"],
        }
        if not explain:
            for row in fallback["ir_results"]:
                row.pop("debug_info", None)
        return fallback

