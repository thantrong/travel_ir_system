"""Import dữ liệu từ file processed JSON vào MongoDB mà không cần chạy lại main.py."""
import sys
import json
import argparse
from pathlib import Path

# Thêm project root vào sys.path để import được database module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def import_to_mongo(processed_path: Path):
    if not processed_path.exists():
        print(f"File không tồn tại: {processed_path}")
        return
    
    print(f"Đang load data từ: {processed_path}")
    data = json.loads(processed_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} records")
    
    from database.data_loader import load_reviews
    place_count, review_count = load_reviews(data)
    print(f"✓ MongoDB upserted places: {place_count}, reviews: {review_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/processed/reviews_processed.json",
                       help="Path to processed JSON file")
    args = parser.parse_args()
    import_to_mongo(Path(args.file))