import csv
import hashlib
from pathlib import Path


def _to_rating_string(value: str) -> str:
    if not value:
        return ""
    try:
        score = float(str(value).strip())
        return str(round(score, 1))
    except Exception:
        return ""


def load_tripadvisor_csv(csv_path: Path) -> list[dict]:
    """Load Tripadvisor CSV and map to unified review schema."""
    if not csv_path.exists():
        return []

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            review_text = str(row.get("Review", "")).strip()
            if not review_text:
                continue

            rating = _to_rating_string(str(row.get("Rating", "")).strip())
            source_hotel_id = "tripadvisor_vnhotel"
            source_review_id = str(idx)
            digest = hashlib.sha1(f"{review_text}|{rating}".encode("utf-8")).hexdigest()[:12]
            if not source_review_id:
                source_review_id = digest

            rows.append(
                {
                    "review_id": f"tripadvisor_{source_hotel_id}_{source_review_id}",
                    "source_review_id": source_review_id,
                    "source_hotel_id": source_hotel_id,
                    "hotel_name": "Tripadvisor VN Hotel Dataset",
                    "location": "Vietnam",
                    "rating": "",
                    "review_text": review_text,
                    "review_date": "",
                    "review_rating": rating,
                    "source": "tripadvisor",
                }
            )
    return rows
