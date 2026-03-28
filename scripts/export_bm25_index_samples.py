from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "bm25_index.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "index" / "bm25_index_samples.json"


def load_index(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Không thể mở bm25_index.pkl vì môi trường thiếu dependency để unpickle object. "
            "Hãy cài rank_bm25 (và các package liên quan) trước khi export samples."
        ) from exc


def build_samples(payload: dict, limit: int = 5) -> list[dict]:
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return []

    samples = []
    for doc in documents[: max(0, limit)]:
        if not isinstance(doc, dict):
            continue
        samples.append(
            {
                "review_id": doc.get("review_id", ""),
                "source_hotel_id": doc.get("source_hotel_id", ""),
                "hotel_name": doc.get("hotel_name", ""),
                "location": doc.get("location", ""),
                "review_rating": doc.get("review_rating", ""),
                "source": doc.get("source", ""),
                "review_text": doc.get("review_text", ""),
                "tokens": doc.get("tokens", []),
                "category_tags": doc.get("category_tags", []),
                "descriptor_tags": doc.get("descriptor_tags", []),
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Export sample BM25 index documents for documentation")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="Path to bm25_index.pkl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path")
    parser.add_argument("--limit", type=int, default=5, help="Number of sample docs to export")
    args = parser.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = load_index(index_path)
    samples = build_samples(payload, limit=args.limit)

    out = {
        "corpus_size": payload.get("corpus_size", len(samples)),
        "sample_count": len(samples),
        "samples": samples,
    }
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded index: {index_path}")
    print(f"Exported samples: {len(samples)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
