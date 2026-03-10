from database.mongo_connection import get_database


def load_reviews(records: list[dict]) -> tuple[int, int]:
    """Upsert places and reviews into MongoDB."""
    db = get_database()
    places_col = db["places"]
    reviews_col = db["reviews"]

    place_count = 0
    review_count = 0

    for row in records:
        source = str(row.get("source", "")).strip().lower()
        source_hotel_id = str(row.get("source_hotel_id", "")).strip()
        review_id = str(row.get("review_id", "")).strip()

        if source and source_hotel_id:
            place_pk = f"{source}_{source_hotel_id}"
            places_col.update_one(
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
            place_count += 1

        if not review_id:
            continue

        review_doc = dict(row)
        review_doc["_id"] = review_id
        # Loại trường thuộc hotel khỏi review để tránh dư thừa.
        review_doc.pop("hotel_name", None)
        review_doc.pop("place_name", None)
        review_doc.pop("location", None)
        review_doc.pop("rating", None)

        reviews_col.update_one(
            {"_id": review_id},
            {"$set": review_doc},
            upsert=True,
        )
        review_count += 1

    return place_count, review_count
