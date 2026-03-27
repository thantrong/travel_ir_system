from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.mongo_connection import get_database
from preprocessing.review_tagger import tag_review


FIELDS_TO_UPDATE = ["category_tags", "descriptor_tags", "hotel_type_tags"]


def _build_tag_doc(review: dict) -> dict:
    res = tag_review(
        str(review.get("review_text", "")),
        hotel_name=str(review.get("hotel_name", "")),
        location=str(review.get("location", "")),
    )
    return {
        "category_tags": res.category_tags,
        "descriptor_tags": res.descriptor_tags,
        "hotel_type_tags": res.hotel_type_tags,
    }


def backfill(limit: int = 0, dry_run: bool = False) -> tuple[int, int]:
    db = get_database()
    reviews_col = db["reviews"]

    query = {
        "$or": [
            {"category_tags": {"$exists": False}},
            {"descriptor_tags": {"$exists": False}},
            {"hotel_type_tags": {"$exists": False}},
        ]
    }
    cursor = reviews_col.find(query)
    if limit > 0:
        cursor = cursor.limit(limit)

    scanned = 0
    updated = 0

    for review in cursor:
        scanned += 1
        review_id = review.get("_id") or review.get("review_id")
        if not review_id:
            continue

        tag_doc = _build_tag_doc(review)
        if dry_run:
            print(f"[DRY RUN] {review_id}: {tag_doc}")
            updated += 1
            continue

        reviews_col.update_one({"_id": review_id}, {"$set": tag_doc})
        updated += 1

    return scanned, updated


def main():
    parser = argparse.ArgumentParser(description="Backfill review tags into MongoDB reviews collection")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N reviews (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Print tag updates without writing to MongoDB")
    args = parser.parse_args()

    scanned, updated = backfill(limit=args.limit, dry_run=args.dry_run)
    print(f"Scanned reviews: {scanned}")
    print(f"Updated reviews: {updated}")


if __name__ == "__main__":
    main()
