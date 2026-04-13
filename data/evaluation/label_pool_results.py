"""
Script gán nhãn relevant cho pool_results.csv
- Đọc file CSV
- Phân tích query và hotel reviews
- Gán nhãn 0 (not relevant) hoặc 1 (relevant)
- Lưu ra file copy mới
"""

import csv
import re
from pathlib import Path

# Đường dẫn file
INPUT_FILE = Path(__file__).parent / "pool_results.csv"
OUTPUT_FILE = Path(__file__).parent / "pool_results_labeled.csv"

# Keywords cho từng loại yêu cầu
LOCATION_KEYWORDS = {
    "phú quốc": ["phú quốc", "phu quoc"],
    "đà lạt": ["đà lạt", "da lat", "lâm đồng"],
    "nha trang": ["nha trang", "khánh hòa"],
    "đà nẵng": ["đà nẵng", "da nang"],
    "vũng tàu": ["vũng tàu", "ba rịa"],
    "hội an": ["hội an", "hoi an"],
    "sapa": ["sapa", "sa pa", "lào cai"],
    "huế": ["huế", "thành phố huế"],
    "hà nội": ["hà nội", "ha noi"],
    "hồ chí minh": ["hồ chí minh", "tp hồ chí minh", "sài gòn"],
    "quy nhơn": ["quy nhơn", "bình định"],
    "phan thiết": ["phan thiết", "mũi né", "mui ne"],
    "hạ long": ["hạ long"],
    "côn đảo": ["côn đảo"],
    "ninh bình": ["ninh bình"],
    "hải phòng": ["hải phòng"],
    "cần thơ": ["cần thơ"],
    "quảng bình": ["quảng bình"],
    "bắc ninh": ["bắc ninh"],
    "buôn mê thuộc": ["buôn mê thuộc", "đắk lắk"],
}

TYPE_KEYWORDS = {
    "resort": ["resort", "khu nghỉ dưỡng"],
    "homestay": ["homestay", "home"],
    "khách_sạn": ["khách sạn", "hotel", "ks"],
    "villa": ["villa", "biệt thự"],
    "nhà_nghỉ": ["nhà nghỉ", "motel"],
}

FEATURE_KEYWORDS = {
    "biển": ["biển", "bãi biển", "ven biển", "sát biển"],
    "hồ_bơi": ["hồ bơi", "bể bơi", "pool"],
    "gym": ["gym", "phòng tập", "fitness"],
    "bếp": ["bếp", "nấu ăn", "bbq"],
    "spa": ["spa", "massage", "xông hơi"],
    "trẻ_em": ["trẻ em", "gia đình", "câu lạc bộ trẻ em"],
    "thú_cưng": ["thú cưng", "pet", "chó", "mèo"],
    "yên_tĩnh": ["yên tĩnh", "yên bình", "không ồn"],
    "trung_tâm": ["trung tâm", "thành phố", "near center"],
    "sân_bay": ["sân bay", "airport"],
    "ban_công": ["ban công", "view"],
    "bufê_sáng": ["bufê", "ăn sáng", "buffet"],
    "karaoke": ["karaoke"],
    "xe_điện": ["xe điện"],
}


def extract_query_features(query):
    """Trích xuất các yêu cầu từ query"""
    features = {}
    query_lower = query.lower()
    
    # Check location
    for loc, keywords in LOCATION_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                features["location"] = loc
                break
        if "location" in features:
            break
    
    # Check type
    for type_, keywords in TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                features.setdefault("type", []).append(type_)
    
    # Check features
    for feat, keywords in FEATURE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                features.setdefault("features", []).append(feat)
    
    return features


def check_hotel_relevance(query, query_features, hotel_name, location, top_reviews):
    """Kiểm tra hotel có relevant với query không"""
    score = 0
    query_lower = query.lower()
    hotel_name_lower = hotel_name.lower()
    location_lower = location.lower()
    
    # Combine all text for matching
    all_text = f"{hotel_name_lower} {location_lower} {' '.join([r.lower() for r in top_reviews if r])}"
    
    # 1. Check location match
    if "location" in query_features:
        target_loc = query_features["location"]
        loc_keywords = LOCATION_KEYWORDS.get(target_loc, [])
        loc_match = any(kw in location_lower or kw in hotel_name_lower for kw in loc_keywords)
        if loc_match:
            score += 2
        else:
            # Check if reviews mention location
            for r in top_reviews:
                r_lower = r.lower() if r else ""
                if any(kw in r_lower for kw in loc_keywords):
                    score += 1
                    break
    
    # 2. Check type match
    if "type" in query_features:
        for type_ in query_features["type"]:
            type_keywords = TYPE_KEYWORDS.get(type_, [])
            if any(kw in hotel_name_lower for kw in type_keywords):
                score += 1
    
    # 3. Check feature requirements
    if "features" in query_features:
        for feat in query_features["features"]:
            feat_keywords = FEATURE_KEYWORDS.get(feat, [])
            # Check hotel name first
            if any(kw in hotel_name_lower for kw in feat_keywords):
                score += 1
                continue
            # Then check reviews
            for r in top_reviews:
                r_lower = r.lower() if r else ""
                if any(kw in r_lower for kw in feat_keywords):
                    score += 0.5
                    break
    
    # 4. Check negative signals (keywords in reviews that contradict query)
    # e.g., query wants "yên tĩnh" but reviews say "ồn ào"
    if "yên_tĩnh" in query_features.get("features", []):
        negative_words = ["ồn ào", "ồn", "tiếng ồn", "cách âm kém", "noise"]
        for r in top_reviews:
            r_lower = r.lower() if r else ""
            if any(nw in r_lower for nw in negative_words):
                score -= 1
                break
    
    if "thú_cưng" in query_features.get("features", []):
        # Check if reviews mention pet-friendly
        found_pet = False
        for r in top_reviews:
            r_lower = r.lower() if r else ""
            if "pet" in r_lower or "thú cưng" in r_lower or "chó" in r_lower or "mèo" in r_lower:
                found_pet = True
                break
        if not found_pet and not any(kw in hotel_name_lower for kw in ["pet", "thú cưng"]):
            score -= 0.5
    
    # 5. Check strong relevance: location exact match in query
    # Long queries with specific requirements
    if len(query.split()) > 10:  # Long query
        # Check if hotel name matches query intent
        query_words = set(query_lower.split())
        hotel_words = set(re.findall(r'\w+', hotel_name_lower))
        common = query_words & hotel_words
        if len(common) >= 2:
            score += 1
    
    return 1 if score >= 1 else 0


def process_csv():
    """Xử lý file CSV và gán nhãn"""
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return
    
    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip BOM from keys
            cleaned_row = {k.lstrip('\ufeff'): v for k, v in row.items()}
            rows.append(cleaned_row)
    
    print(f"Loaded {len(rows)} rows from {INPUT_FILE}")
    
    # Process
    labeled_count = 0
    relevant_count = 0
    not_relevant_count = 0
    
    for row in rows:
        query = row.get("query", "")
        hotel_name = row.get("hotel_name", "")
        location = row.get("hotel_location", "")
        top_review_1 = row.get("top_review_1", "")
        top_review_2 = row.get("top_review_2", "")
        top_review_3 = row.get("top_review_3", "")
        
        # Extract query features
        query_features = extract_query_features(query)
        
        # Check relevance
        relevant = check_hotel_relevance(
            query, query_features, hotel_name, location,
            [top_review_1, top_review_2, top_review_3]
        )
        
        row["relevant"] = str(relevant)
        labeled_count += 1
        if relevant == 1:
            relevant_count += 1
        else:
            not_relevant_count += 1
    
    # Write output
    fieldnames = ["bucket_id", "query_id", "query", "hotel_id", "hotel_name", 
                  "hotel_location", "bm25_rank", "vector_rank", "hybrid_rank",
                  "top_review_1", "top_review_2", "top_review_3", "relevant"]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n=== LABELING SUMMARY ===")
    print(f"Total labeled: {labeled_count}")
    print(f"Relevant (1): {relevant_count}")
    print(f"Not relevant (0): {not_relevant_count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_csv()