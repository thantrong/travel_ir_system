import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from multiprocessing import cpu_count

from nlp.normalization import normalize_text
from nlp.stopwords import load_stopwords, remove_stopwords
from nlp.tokenizer import tokenize_vi
from preprocessing.clean_text import clean_review_text
from preprocessing.language_filter import is_vietnamese_text
from preprocessing.remove_spam import is_spam_review
from preprocessing.review_tagger import tag_record


class ReviewDataset(Dataset):
    def __init__(self, records: list[dict], stopwords: set):
        self.records = records
        self.stopwords = stopwords

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        text = clean_review_text(row.get("review_text", ""))
        if not text or is_spam_review(text):
            return None
        if not is_vietnamese_text(text):
            return None

        clean = normalize_text(text)
        tokens = remove_stopwords(tokenize_vi(clean), self.stopwords)
        if not tokens:
            return None

        transformed = dict(row)
        transformed["clean_text"] = clean
        transformed["tokens"] = tokens
        transformed = tag_record(transformed)
        return transformed


def read_records(raw_json: Path, raw_jsonl: Path) -> list[dict]:
    if raw_json.exists():
        data = json.loads(raw_json.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data
    if raw_jsonl.exists():
        rows = []
        for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    return []


def collate_fn(batch):
    return [item for item in batch if item is not None]


def process_records_dataloader(records: list[dict], stopwords_path: Path, batch_size: int = 64, num_workers: int = 4) -> list[dict]:
    stopwords = load_stopwords(stopwords_path)
    dataset = ReviewDataset(records, stopwords)
    
    # num_workers = min(num_workers, cpu_count())
    actual_workers = min(num_workers, cpu_count())
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    output = []
    for batch in dataloader:
        output.extend(batch)
    
    return output


def process_records(records: list[dict], stopwords_path: Path) -> list[dict]:
    stopwords = load_stopwords(stopwords_path)
    output = []
    for row in records:
        text = clean_review_text(row.get("review_text", ""))
        if not text or is_spam_review(text):
            continue
        if not is_vietnamese_text(text):
            continue

        clean = normalize_text(text)
        tokens = remove_stopwords(tokenize_vi(clean), stopwords)
        if not tokens:
            continue

        transformed = dict(row)
        transformed["clean_text"] = clean
        transformed["tokens"] = tokens
        transformed = tag_record(transformed)
        output.append(transformed)
    return output


def main():
    parser = argparse.ArgumentParser(description="Offline preprocessing pipeline")
    parser.add_argument("--load-mongo", action="store_true", help="Load processed data into MongoDB")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--no-dataloader", action="store_true", help="Use old single-thread processing")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    raw_json = project_root / "data" / "raw" / "traveloka_raw_final.json"
    raw_jsonl = project_root / "data" / "raw" / "traveloka_checkpoint.jsonl"
    processed_path = project_root / "data" / "processed" / "reviews_processed.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    stopwords_path = project_root / "config" / "stopwords.txt"

    records = read_records(raw_json, raw_jsonl)
    print("Note: external datasets should be ingested via dataset_pipeline.py (config/dataset_sources.yaml).")
    print(f"Total records: {len(records)}")

    if args.no_dataloader:
        print("Using single-thread processing (no DataLoader)...")
        processed = process_records(records, stopwords_path)
    else:
        print(f"Using DataLoader: batch_size={args.batch_size}, num_workers={args.num_workers}...")
        processed = process_records_dataloader(records, stopwords_path, batch_size=args.batch_size, num_workers=args.num_workers)

    processed_path.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed records: {len(processed)}")
    print(f"Saved: {processed_path}")

    if args.load_mongo:
        from database.data_loader import load_reviews

        place_count, review_count = load_reviews(processed)
        print(f"Mongo upserted places: {place_count}, reviews: {review_count}")


if __name__ == "__main__":
    main()