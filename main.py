import argparse
import json
from pathlib import Path
import gc

from torch.utils.data import DataLoader, Dataset
from multiprocessing import cpu_count
import torch
from nlp.normalization import normalize_text
from nlp.stopwords import load_stopwords, remove_stopwords
from nlp.tokenizer import tokenize_vi
from preprocessing.clean_text import clean_review_text
from preprocessing.language_filter import is_vietnamese_text
from preprocessing.remove_spam import is_spam_review
from tqdm import tqdm
from preprocessing.review_tagger import tag_record, tag_records_batch

PROJECT_ROOT = Path(__file__).resolve().parent

print(f"Using PyTorch with {torch.__version__}, CPU count: {cpu_count()}, torch available: {torch.cuda.is_available()},GPU devices: {torch.cuda.device_count()} ")

class ReviewDataset(Dataset):
    def __init__(self, records: list[dict], stopwords: set, use_batch_tagging: bool = False):
        self.records = records
        self.stopwords = stopwords
        self.use_batch_tagging = use_batch_tagging

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


def process_records_dataloader(records: list[dict], stopwords_path: Path, batch_size: int = 64, num_workers: int = 4, phobert_batch_size: int = 64, mongo_batch_size: int = 10000, load_mongo: bool = False) -> list[dict]:
    """Xử lý records với DataLoader + batch PhoBERT inference + load theo lô."""
    import time
    
    total = len(records)
    print(f"\n{'='*60}")
    print(f"PREPROCESSING PIPELINE ({total} records)")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
    print(f"PhoBERT batch_size: {phobert_batch_size} (FP16)")
    print(f"MongoDB batch_size: {mongo_batch_size}")
    print(f"{'='*60}\n")
    
    stopwords = load_stopwords(stopwords_path)
    
    # Chia records thành các lô để xử lý
    total_tagged_count = 0
    total_batches = (total + mongo_batch_size - 1) // mongo_batch_size
    
    for batch_idx in range(total_batches):
        start = batch_idx * mongo_batch_size
        end = min(start + mongo_batch_size, total)
        batch_records = records[start:end]
        
        print(f"\n{'─'*60}")
        print(f"[Batch {batch_idx+1}/{total_batches}] Records {start}-{end-1}")
        print(f"{'─'*60}")
        
        # Bước 1: Lọc và tokenize
        print(f"[Step 1/3] Filtering and tokenizing {len(batch_records)} records...")
        start_time = time.time()
        valid_records = []
        for row in tqdm(batch_records, desc="  Filtering", unit="rec", total=len(batch_records)):
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
            valid_records.append(transformed)
        
        elapsed = time.time() - start_time
        print(f"  ✓ {len(valid_records)}/{len(batch_records)} valid in {elapsed:.1f}s")
        
        if not valid_records:
            print(f"  ⊘ No valid records in this batch, skipping...")
            continue
        
        # Bước 2: Batch PhoBERT tagging
        tagged = tag_records_batch(valid_records, phobert_batch_size=phobert_batch_size)
        total_tagged_count += len(tagged)
        
        # Bước 3: Save batch
        processed_json = PROJECT_ROOT / "data" / "processed" / "reviews_processed.json"
        processed_jsonl = PROJECT_ROOT / "data" / "processed" / "reviews_processed.jsonl"
        processed_json.parent.mkdir(parents=True, exist_ok=True)

        # Duy trì file JSON cho tương thích cũ + thêm JSONL append để tránh O(n^2) I/O.
        if batch_idx == 0:
            processed_json.write_text(json.dumps(tagged, ensure_ascii=False, indent=2), encoding="utf-8")
            with processed_jsonl.open("w", encoding="utf-8") as jf:
                for item in tagged:
                    jf.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with processed_jsonl.open("a", encoding="utf-8") as jf:
                for item in tagged:
                    jf.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved batch {batch_idx+1} ({len(tagged)} records)")
        
        # Load lên MongoDB nếu được yêu cầu
        if load_mongo and tagged:
            print(f"[MongoDB] Loading batch {batch_idx+1}...")
            try:
                from database.data_loader import load_reviews
                place_count, review_count = load_reviews(tagged)
                print(f"  ✓ MongoDB upserted: {place_count} places, {review_count} reviews")
            except Exception as e:
                print(f"  ✗ MongoDB error: {e}")
                print(f"  → Data saved to JSON, you can retry later with:")
                print(f"    python scripts/import_batch_mongo.py")

        # Giảm peak memory khi chạy lô lớn liên tục.
        del valid_records
        del tagged
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"[Summary] Done. {total_tagged_count} records processed.")
    print(f"{'='*60}\n")
    
    return []


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
    parser.add_argument("--load-mongo", action="store_true", help="Load processed data into MongoDB (batch by batch)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--phobert-batch-size", type=int, default=64, help="PhoBERT inference batch size (FP16)")
    parser.add_argument("--mongo-batch-size", type=int, default=10000, help="Records per batch for MongoDB loading")
    parser.add_argument("--no-dataloader", action="store_true", help="Use old single-thread processing")
    args = parser.parse_args()

    raw_json = PROJECT_ROOT / "data" / "raw" / "traveloka_raw_final.json"
    raw_jsonl = PROJECT_ROOT / "data" / "raw" / "traveloka_checkpoint.jsonl"

    stopwords_path = PROJECT_ROOT / "config" / "stopwords.txt"

    records = read_records(raw_json, raw_jsonl)
    print("Note: external datasets should be ingested via dataset_pipeline.py (config/dataset_sources.yaml).")
    print(f"Total records: {len(records)}")

    if args.no_dataloader:
        print("Using single-thread processing (no DataLoader)...")
        processed = process_records(records, stopwords_path)
        processed_path = PROJECT_ROOT / "data" / "processed" / "reviews_processed.json"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Processed records: {len(processed)}")
        print(f"Saved: {processed_path}")
        
        if args.load_mongo:
            from database.data_loader import load_reviews
            place_count, review_count = load_reviews(processed)
            print(f"Mongo upserted places: {place_count}, reviews: {review_count}")
    else:
        processed = process_records_dataloader(
            records, stopwords_path, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            phobert_batch_size=args.phobert_batch_size,
            mongo_batch_size=args.mongo_batch_size,
            load_mongo=args.load_mongo
        )


if __name__ == "__main__":
    main()