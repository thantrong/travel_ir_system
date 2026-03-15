"""
Build Vector Index (Review-level)
PHẦN 2: DESIGN DATA REPRESENTATION (REVIEW-LEVEL)
PHẦN 3: XÂY DỰNG INDEXING PIPELINE
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from database.mongo_connection import get_database


def fetch_reviews_for_indexing() -> list[dict]:
    db = get_database()
    places_col = db["places"]
    reviews_col = db["reviews"]

    place_map = {}
    for doc in places_col.find():
        sid = doc.get("source_hotel_id", "")
        if sid:
            place_map[sid] = {
                "hotel_name": doc.get("name", ""),
                "location": doc.get("location", ""),
            }

    docs = []
    cursor = reviews_col.find({"clean_text": {"$exists": True}})
    for row in cursor:
        sid = str(row.get("source_hotel_id", "")).strip()
        if sid not in place_map:
            continue
            
        rtxt = str(row.get("clean_text", "")).strip()
        if not rtxt:
            continue
            
        docs.append({
            "review_id": row.get("review_id", ""),
            "source": row.get("source", ""),
            "source_hotel_id": sid,
            "hotel_name": place_map[sid]["hotel_name"],
            "location": place_map[sid]["location"],
            "rating": place_map[sid].get("rating", row.get("review_rating", "")),
            "review_rating": row.get("review_rating", ""),
            "review_text": row.get("review_text", ""),
            "clean_text": rtxt,
        })
        
    return docs


def build_vector_index(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> dict:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError("Cài đặt: pip install sentence-transformers")

    reviews = fetch_reviews_for_indexing()
    print(f"Đang chuẩn bị {len(reviews)} reviews để embed...")

    texts = []
    documents = []
    
    for r in reviews:
        parts = []
        if r["hotel_name"]:
            parts.append(f"Hotel: {r['hotel_name']}")
        if r["location"]:
            parts.append(f"Location: {r['location']}")
            
        parts.append(f"Review: {r['clean_text']}")
        
        full_text = ". ".join(parts)
        texts.append(full_text)
        
        documents.append({
            "review_id": r["review_id"],
            "source_hotel_id": r["source_hotel_id"],
            "hotel_name": r["hotel_name"],
            "location": r["location"],
            "review_text": r["review_text"],
            "review_rating": r["review_rating"],
            "source": r["source"]
        })

    model = SentenceTransformer(model_name)
    print(f"Đang chạy Sentence-BERT ({model_name}) cho {len(texts)} review documents...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 50)

    return {
        "embeddings": np.array(embeddings, dtype=np.float32),
        "documents": documents,
        "model_name": model_name,
        "corpus_size": len(documents),
    }


def main():
    parser = argparse.ArgumentParser(description="Build Review-Level Vector index")
    parser.add_argument("--output", default="data/index/vector_index.pkl", help="Output Index path")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_vector_index(model_name=args.model)
    
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Indexed reviews: {payload['corpus_size']}")
    print(f"Saved Vector review-level index: {output_path}")


if __name__ == "__main__":
    main()
