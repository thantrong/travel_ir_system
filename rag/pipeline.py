import re
import json
from urllib import request, error
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


def _compose_llm_prompt(query: str, citations: list[dict[str, Any]]) -> str:
    evidence_lines: list[str] = []
    for c in citations:
        evidence_lines.append(
            f"[{c['id']}] Hotel: {c['hotel_name']} | Location: {c['location']} | Snippet: {c['snippet']}"
        )

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "Khong co bang chung."
    return (
        "Ban la tro ly tra loi du lich dua tren bang chung.\n"
        "Chi duoc su dung bang chung duoi day. Neu khong du thong tin, noi ro khong du du lieu.\n"
        "Bat buoc gan citation theo dinh dang [C1], [C2]... cho moi y chinh.\n\n"
        f"Cau hoi: {query}\n\n"
        f"Bang chung:\n{evidence_block}\n\n"
        "Tra loi ngan gon, dung tieng Viet."
    )


def _history_to_text(chat_history: list[dict[str, str]] | None) -> str:
    if not chat_history:
        return "Khong co lich su hoi thoai."
    lines: list[str] = []
    for turn in chat_history[-8:]:
        role = "Khach" if turn.get("role") == "user" else "Tro ly"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "Khong co lich su hoi thoai."


def _top_hotel_brief(ir_results: list[dict[str, Any]], limit: int = 3) -> str:
    lines: list[str] = []
    for idx, row in enumerate(ir_results[:limit], start=1):
        lines.append(
            f"{idx}. {row.get('hotel_name', 'Khong ro')} | {row.get('location', 'Khong ro')} | "
            f"rating: {row.get('rating', 'N/A')} | score: {round(float(row.get('hybrid_score', 0.0)), 3)}"
        )
    return "\n".join(lines) if lines else "Khong co de xuat khach san."


def _compose_chatbot_prompt(
    query: str,
    citations: list[dict[str, Any]],
    ir_results: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None,
) -> str:
    evidence_lines: list[str] = []
    for c in citations:
        evidence_lines.append(
            f"[{c['id']}] Hotel: {c['hotel_name']} | Location: {c['location']} | Snippet: {c['snippet']}"
        )

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "Khong co bang chung."
    history_block = _history_to_text(chat_history)
    hotel_brief = _top_hotel_brief(ir_results)
    return (
        "Ban la nhan vien tu van khach san, noi chuyen tu nhien nhu chat that voi khach.\n"
        "Muc tieu: giup khach chon duoc noi luu tru phu hop nhu cau.\n"
        "Quy tac bat buoc:\n"
        "- Khong che du lieu, chi dua tren bang chung.\n"
        "- Khi dua ra nhan dinh cu the ve khach san, chen citation [C1], [C2]...\n"
        "- Neu thong tin thieu, hoi lai nhe nhang de lam ro.\n"
        "- Khong ep theo mau co dinh, tra loi linh hoat theo ngu canh hoi thoai.\n\n"
        f"Lich su hoi thoai:\n{history_block}\n\n"
        f"Cau hoi hien tai: {query}\n\n"
        f"Tom tat top khach san:\n{hotel_brief}\n\n"
        f"Bang chung:\n{evidence_block}\n\n"
        "Hay tra loi bang tieng Viet tu nhien, ngan gon vua du, than thien va de hieu.\n"
    )


def _call_ollama_generate(base_url: str, model: str, prompt: str, timeout_s: float) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    req = request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return str(parsed.get("response", "")).strip()


def _grounding_ok(answer: str, citations: list[dict[str, Any]]) -> bool:
    if not answer or not citations:
        return False
    return any(f"[{c['id']}]" in answer for c in citations)


def _ensure_citation_markers(answer: str, citations: list[dict[str, Any]]) -> str:
    if not answer or not citations:
        return answer
    if _grounding_ok(answer, citations):
        return answer
    marker = " ".join(f"[{c['id']}]" for c in citations[:2])
    return f"{answer.strip()}\n\nNguon tham chieu: {marker}"


def answer_from_ir_results(
    query: str,
    ir_results: list[dict[str, Any]],
    top_k_context: int = 6,
    max_citations: int = 4,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout_s: float = 30.0,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    chunks = _build_context_chunks(ir_results, max_chunks=36)
    ranked_chunks = _rerank_chunks(query, chunks, top_k_context=top_k_context)
    citations = _build_citations(ranked_chunks, max_citations=max_citations)
    answer_mode = "extractive"
    answer = _compose_extractive_answer(query, citations)

    if llm_base_url and llm_model and citations:
        try:
            prompt = _compose_chatbot_prompt(
                query=query,
                citations=citations,
                ir_results=ir_results,
                chat_history=chat_history,
            )
            llm_answer = _call_ollama_generate(
                base_url=llm_base_url,
                model=llm_model,
                prompt=prompt,
                timeout_s=llm_timeout_s,
            )
            if llm_answer:
                answer = _ensure_citation_markers(llm_answer, citations)
                answer_mode = "llm"
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            # Fallback to extractive answer on any remote LLM failures.
            answer_mode = "extractive_fallback"

    grounded = _grounding_ok(answer, citations)

    return {
        "answer": answer,
        "citations": citations,
        "grounded": grounded,
        "context_items_used": len(ranked_chunks),
        "answer_mode": answer_mode,
    }
