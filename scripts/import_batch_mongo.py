"""Import dữ liệu từ file processed JSON vào MongoDB theo batch."""
import sys
import json
import time
import argparse
from pathlib import Path

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def import_batches(processed_path: Path, batch_size: int = 100, start_from: int = 0):
    """Import data vào MongoDB theo từng batch."""
    from database.data_loader import load_reviews

    if not processed_path.exists():
        print(f"File không tồn tại: {processed_path}")
        return
    
    print(f"Đang load data từ: {processed_path}")
    data = json.loads(processed_path.read_text(encoding="utf-8"))
    total = len(data)
    print(f"Total records: {total}")
    
    if start_from > 0:
        data = data[start_from:]
        print(f"Skipping first {start_from}, starting from index {start_from}")
    
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    print(f"\n{'='*60}")
    print(f"BATCH IMPORT SETUP")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Retry on failure: 3 times per batch")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_batches = []
    
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(data))
        batch = data[start:end]
        
        current = start_from + start
        progress = (batch_idx + 1) / total_batches * 100
        print(f"\n[Batch {batch_idx+1}/{total_batches}] Records {start_from+start}-{start_from+end-1} ({progress:.0f}%)...")
        
        # Retry up to 3 times
        for attempt in range(1, 4):
            try:
                place_count, review_count = load_reviews(batch)
                success_count += len(batch)
                print(f"  ✓ OK - {review_count} reviews, {place_count} places added/updated")
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)[:100]
                print(f"  ✗ Attempt {attempt} failed: {error_msg}")
                if attempt >= 3:
                    print(f"  ✗ FAILED permanently - batch {batch_idx+1}")
                    failed_batches.append((batch_idx+1, start+start_from, end+start_from-1))
                else:
                    wait = attempt * 2  # 2s, 4s, 6s
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
        
        # Small delay between batches to avoid rate limiting
        if batch_idx < total_batches - 1 and success_count > 0:
            time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"IMPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {success_count}/{total}")
    
    if failed_batches:
        print(f"\nFailed batches:")
        for fb, start, end in failed_batches:
            print(f"  - Batch {fb}: records {start}-{end}")
    
    print(f"{'='*60}")
    
    if not failed_batches:
        print("✓ All batches imported successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/processed/reviews_processed.json")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of records per batch")
    parser.add_argument("--start-from", type=int, default=0, help="Skip N records (resume from)")
    args = parser.parse_args()
    import_batches(Path(args.file), batch_size=args.batch_size, start_from=args.start_from)