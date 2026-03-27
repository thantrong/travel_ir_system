from pymongo import UpdateOne

from database.mongo_connection import get_database


def load_reviews(records: list[dict]) -> tuple[int, int]:
    """Upsert places and reviews into MongoDB, in batches."""
    db = get_database()
    places_col = db["places"]
    reviews_col = db["reviews"]

    place_ops = []
    review_ops = []

    for row in records:
        source = str(row.get("source", "")).strip().lower()
        source_hotel_id = str(row.get("source_hotel_id", "")).strip()
        review_id = str(row.get("review_id", "")).strip()

        if source and source_hotel_id:
            place_pk = f"{source}_{source_hotel_id}"
            place_ops.append(
                UpdateOne(
                    {"_id": place_pk},
                    {
                        "$set": {
                            "_id": place_pk,
                            "name": row.get("hotel_name", row.get("place_name", "")),
                            "type": "hotel",
                            "location": row.get("location", ""),
                            "rating": row.get("rating", ""),
                            "source": source,
                            "source_hotel_id": source_hotel_id,
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

    return place_count, review_count
