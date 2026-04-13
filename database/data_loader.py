from pymongo import UpdateOne

from database.mongo_connection import get_collection_names, get_database


def load_reviews(records: list[dict]) -> tuple[int, int]:
    """Upsert places and reviews into MongoDB, in batches."""
    db = get_database()
    collections = get_collection_names()
    places_col = db[collections["places"]]
    reviews_col = db[collections["reviews"]]

    place_ops = []
    review_ops = []
    skipped_no_review_id = 0

    for row in records:
        source_hotel_id = str(row.get("source_hotel_id", "")).strip()
        review_id = str(row.get("review_id", "")).strip()

        if not review_id:
            skipped_no_review_id += 1
            continue

        if source_hotel_id:
            place_types = row.get("place_types") or row.get("types") or ["hotel"]
            if isinstance(place_types, str):
                place_types = [place_types]
            if not isinstance(place_types, list):
                place_types = ["hotel"]
            normalized_types = []
            for t in place_types:
                vt = str(t).strip().lower()
                if vt and vt not in normalized_types:
                    normalized_types.append(vt)
            if not normalized_types:
                normalized_types = ["hotel"]

            place_ops.append(
                UpdateOne(
                    {"_id": source_hotel_id},
                    {
                        "$set": {
                            "_id": source_hotel_id,
                            "types": normalized_types,
                            "location": row.get("location", ""),
                            "rating": row.get("rating", ""),
                            "hotel_name": row.get("hotel_name", ""),
                        }
                    },
                    upsert=True,
                )
            )

        if not review_id:
            continue

        review_doc = dict(row)
        # Chỉ giữ _id, không lưu review_id lặp lại nữa.
        review_doc["_id"] = review_id
        review_doc.pop("review_id", None)
        # Loại trường thuộc hotel khỏi review để tránh dư thừa.
        review_doc.pop("hotel_name", None)
        review_doc.pop("place_name", None)
        review_doc.pop("location", None)
        review_doc.pop("rating", None)
        review_doc.pop("place_types", None)
        review_doc.pop("place_type_source", None)
        review_doc.pop("types", None)
        review_doc.pop("matched_phrases", None)
        review_doc.pop("contexts", None)
        review_doc.pop("descriptor_polarity", None)

        review_ops.append(
            UpdateOne(
                {"_id": review_id},
                {"$set": review_doc},
                upsert=True,
            )
        )

    place_count = len(place_ops)
    review_count = len(review_ops)

    if place_ops:
        places_col.bulk_write(place_ops, ordered=False)
    if review_ops:
        reviews_col.bulk_write(review_ops, ordered=False)

    if skipped_no_review_id > 0:
        print(f"  ⚠ Skipped {skipped_no_review_id} records (no review_id)")

    return place_count, review_count
