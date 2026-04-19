import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str]) -> None:
    print(f"\n[STEP] {name}")
    print(" ".join(args))
    completed = subprocess.run(args, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")


def smoke_test(query: str, top_k: int) -> None:
    print("\n[STEP] Smoke test RAG")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from api.service import answer_with_rag

    payload = answer_with_rag(
        query=query,
        top_k_retrieval=max(1, top_k),
        allow_fallback_to_ir=True,
        explain=True,
    )
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise RuntimeError("Smoke test failed: empty answer")
    print(f"Mode: {payload.get('mode')}")
    print(f"Grounded: {payload.get('grounded')}")
    print(f"IR count: {payload.get('ir_count')}")
    print(f"Answer: {answer[:280]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline for Travel IR RAG")
    parser.add_argument("--cities", type=str, default="", help="City codes for crawler, e.g. HN,DN")
    parser.add_argument("--append-crawl", action="store_true", help="Append mode for crawler")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawling step")
    parser.add_argument("--skip-hotel-clean", action="store_true", help="Skip hotel detail clean step")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocess step")
    parser.add_argument("--skip-index", action="store_true", help="Skip BM25/vector indexing steps")
    parser.add_argument("--load-mongo", action="store_true", help="Load into MongoDB in preprocess step")
    parser.add_argument("--smoke-query", type=str, default="khách sạn view biển đẹp ở phú quốc")
    parser.add_argument("--smoke-top-k", type=int, default=10)
    args = parser.parse_args()

    if not args.skip_crawl:
        crawl_cmd = [sys.executable, "crawler/traveloka_crawler.py"]
        if args.cities.strip():
            crawl_cmd.extend(["--cities", args.cities.strip()])
        if args.append_crawl:
            crawl_cmd.append("--append")
        run_step("Crawler reviews", crawl_cmd)

    if not args.skip_hotel_clean:
        run_step("Clean hotel detail data", [sys.executable, "scripts/process_hotel_detail_clean.py"])

    if not args.skip_preprocess:
        preprocess_cmd = [sys.executable, "main.py"]
        if args.load_mongo:
            preprocess_cmd.append("--load-mongo")
        run_step("Preprocess reviews", preprocess_cmd)

    if not args.skip_index:
        run_step("Build BM25 index", [sys.executable, "indexing/build_bm25_index.py"])
        run_step("Build vector index", [sys.executable, "indexing/build_vector_index.py"])

    smoke_test(query=args.smoke_query, top_k=args.smoke_top_k)
    print("\nDone end-to-end.")


if __name__ == "__main__":
    main()
