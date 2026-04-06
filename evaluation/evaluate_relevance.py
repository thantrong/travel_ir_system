"""
Auto-evaluate relevance (0/1) cho pool results.
Dùng rules + semantic matching để đánh giá tự động.
"""
import sys
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

POOL_JSON = PROJECT_ROOT / "data" / "evaluation" / "pool_results.json"
POOL_CSV = PROJECT_ROOT / "data" / "evaluation" / "pool_results.csv"
EVALUATED_JSON = PROJECT_ROOT / "data" / "evaluation" / "pool_evaluated.json"
EVALUATED_CSV = PROJECT_ROOT / "data" / "evaluation" / "pool_evaluated.csv"

# Keywords mapping cho từng loại query
TYPE_KEYWORDS = {
    "khách sạn": ["khách sạn", "hotel"],
    "resort": ["resort", "khu nghỉ dưỡng"],
    "homestay": ["homestay", "nhà dân", "farmstay"],
    "villa": ["villa", "biệt thự", "bungalow"],
    "nhà nghỉ": ["nhà nghỉ", "motel", "guesthouse"],
}

LOCATION_KEYWORDS = {
    "đà nẵng": ["đà nẵng", "danang", "đà nẵng"],
    "phú quốc": ["phú quốc", "phu quoc"],
    "nha trang": ["nha trang", "nhatrang"],
    "đà lạt": ["đà lạt", "dalat"],
    "sapa": ["sapa", "sa pa"],
    "hà nội": ["hà nội", "hanoi"],
    "hồ chí minh": ["hồ chí minh", "hcm", "sài gòn", "saigon"],
    "vũng tàu": ["vũng tàu", "vungtau"],
    "hội an": ["hội an", "hoian"],
    "hạ long": ["hạ long", "halong"],
    "huế": ["huế", "hue"],
    "quy nhơn": ["quy nhơn", "quynhon"],
    "phan thiết": ["phan thiết", "phanthiet"],
    "cần thơ": ["cần thơ", "cantho"],
    "hải phòng": ["hải phòng", "haiphong"],
}

AMENITY_KEYWORDS = {
    "biển": ["biển", "beach", "ven biển", "sát biển", "bãi biển"],
    "hồ bơi": ["hồ bơi", "bể bơi", "pool", "bơi"],
    "spa": ["spa", "massage", "xông hơi", "sauna"],
    "gym": ["gym", "phòng tập", "thể dục", "fitness"],
    "ăn sáng": ["ăn sáng", "buffet", "breakfast"],
    "wifi": ["wifi", "internet", "mạng"],
    "bếp": ["bếp", "nấu ăn", "kitchen"],
    "ban công": ["ban công", "balcony", "sân thượng"],
    "view": ["view", "nhìn ra", "hướng", "ngắm"],
    "yên tĩnh": ["yên tĩnh", "yên bình", "nghỉ dưỡng", "tĩnh lặng"],
    "giá rẻ": ["giá rẻ", "rẻ", "bình dân", "tiết kiệm", "hợp lý"],
    "5 sao": ["5 sao", "5*", "sang trọng", "cao cấp", "luxury"],
    "4 sao": ["4 sao", "4*"],
    "3 sao": ["3 sao", "3*"],
    "gia đình": ["gia đình", "trẻ em", "gia đình", "con cái"],
    "công tác": ["công tác", "business", "work"],
    "tuần trăng mật": ["tuần trăng mật", "honeymoon", "lãng mạn", "cặp đôi"],
}

def evaluate_relevance(row: dict) -> int:
    """Đánh giá relevance 0/1 cho 1 row."""
    query = row.get("query", "").lower()
    hotel_name = row.get("hotel_name", "").lower()
    review_text = " ".join([
        row.get("top_review_1", ""),
        row.get("top_review_2", ""),
        row.get("top_review_3", ""),
    ]).lower()
    hybrid_rank = row.get("hybrid_rank", 99)
    
    # Score dựa trên matching
    match_score = 0
    
    # 1. Type matching (query type vs hotel name/reviews)
    for type_name, keywords in TYPE_KEYWORDS.items():
        if type_name in query:
            if any(kw in hotel_name for kw in keywords):
                match_score += 3
            elif any(kw in review_text for kw in keywords):
                match_score += 1
    
    # 2. Location matching
    for loc_name, keywords in LOCATION_KEYWORDS.items():
        if loc_name in query:
            # Check trong hotel name và review
            if any(kw in hotel_name for kw in keywords):
                match_score += 3
            elif any(kw in review_text for kw in keywords):
                match_score += 1
    
    # 3. Amenity matching
    for amenity, keywords in AMENITY_KEYWORDS.items():
        if amenity in query:
            if any(kw in review_text for kw in keywords):
                match_score += 2
            elif any(kw in hotel_name for kw in keywords):
                match_score += 1
    
    # 4. Rank bonus (rank cao → khả năng relevant cao hơn)
    if hybrid_rank <= 3:
        match_score += 2
    elif hybrid_rank <= 5:
        match_score += 1
    
    # 5. Generic queries (chỉ có location hoặc type) → dễ relevant hơn
    query_words = set(query.split())
    if len(query_words) <= 3:
        match_score += 1
    
    # Decision: threshold
    return 1 if match_score >= 2 else 0

def evaluate_all():
    """Load pool results và đánh giá tất cả."""
    if not POOL_JSON.exists():
        print(f"Error: {POOL_JSON} not found")
        print("Run: python evaluation/generate_pool.py first")
        return
    
    # Load JSON
    data = json.loads(POOL_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} (query, hotel) pairs")
    
    # Evaluate
    relevant_count = 0
    for row in data:
        row["relevant"] = evaluate_relevance(row)
        if row["relevant"] == 1:
            relevant_count += 1
    
    # Save evaluated JSON
    EVALUATED_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVALUATED_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # Save evaluated CSV
    if data:
        fieldnames = list(data[0].keys())
        with open(EVALUATED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total pairs: {len(data)}")
    print(f"Relevant (1): {relevant_count} ({relevant_count/len(data)*100:.1f}%)")
    print(f"Not relevant (0): {len(data) - relevant_count}")
    print(f"Saved JSON: {EVALUATED_JSON}")
    print(f"Saved CSV:  {EVALUATED_CSV}")

if __name__ == "__main__":
    evaluate_all()