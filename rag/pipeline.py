import re
from typing import Any

from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"[.!?;\n]+", text or "")
    return [c.strip() for c in chunks if c and c.strip()]


def _token_set(text: str) -> set[str]:
    normalized = normalize_text(text or "")
    tokens = tokenize_vi(normalized)
    return {t.lower().strip() for t in tokens if t and t.strip()}


def _build_context_chunks(results: list[dict[str, Any]], max_chunks: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for hotel in results:
        hotel_name = hotel.get("hotel_name", "")
        location = hotel.get("location", "")
        base_score = float(hotel.get("hybrid_score", 0.0))
        for review_text in hotel.get("top_reviews", []) or []:
            for sent in _split_sentences(str(review_text)):
                chunks.append(
                    {
                        "text": sent,
                        "hotel_name": hotel_name,
                        "location": location,
                        "score": base_score,
                        "source_hotel_id": hotel.get("source_hotel_id", ""),
                    }
                )
                if len(chunks) >= max_chunks:
                    return chunks
    return chunks


def _rerank_chunks(query: str, chunks: list[dict[str, Any]], top_k_context: int) -> list[dict[str, Any]]:
    q_tokens = _token_set(query)
    if not chunks:
        return []

    for c in chunks:
        c_tokens = _token_set(c["text"])
        overlap = len(q_tokens.intersection(c_tokens))
        c["rerank_score"] = float(c.get("score", 0.0)) + 0.2 * overlap

    chunks.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return chunks[:top_k_context]


def _build_citations(chunks: list[dict[str, Any]], max_citations: int) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for idx, c in enumerate(chunks[:max_citations], start=1):
        citations.append(
            {
                "id": f"C{idx}",
                "source_hotel_id": c.get("source_hotel_id", ""),
                "hotel_name": c.get("hotel_name", ""),
                "location": c.get("location", ""),
                "snippet": c.get("text", ""),
                "score": round(float(c.get("rerank_score", 0.0)), 4),
            }
        )
    return citations


def _compose_extractive_answer(query: str, citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "Khong du ngu canh dang tin cay de tao cau tra loi RAG."

    summary_lines = [
        f"Tom tat cho truy van: '{query}'.",
    ]
    for c in citations[:2]:
        summary_lines.append(
            f"- {c['hotel_name']} ({c['location']}): {c['snippet']} [{c['id']}]"
        )
    if len(citations) > 2:
        summary_lines.append(f"- Co them {len(citations) - 2} bang chung lien quan tu tap review.")
    return "\n".join(summary_lines)


def _grounding_ok(answer: str, citations: list[dict[str, Any]]) -> bool:
    if not answer or not citations:
        return False
    return any(f"[{c['id']}]" in answer for c in citations)


def answer_from_ir_results(
    query: str,
    ir_results: list[dict[str, Any]],
    top_k_context: int = 6,
    max_citations: int = 4,
) -> dict[str, Any]:
    chunks = _build_context_chunks(ir_results, max_chunks=36)
    ranked_chunks = _rerank_chunks(query, chunks, top_k_context=top_k_context)
    citations = _build_citations(ranked_chunks, max_citations=max_citations)
    answer = _compose_extractive_answer(query, citations)
    grounded = _grounding_ok(answer, citations)

    return {
        "answer": answer,
        "citations": citations,
        "grounded": grounded,
        "context_items_used": len(ranked_chunks),
    }
