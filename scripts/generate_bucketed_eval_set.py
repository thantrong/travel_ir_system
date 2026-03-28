from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "evaluation" / "test_queries_200_bucketed.json"

BUCKETS = [
    {
        "bucket_id": "bucket_1_short",
        "bucket_name": "short_1_2_attributes",
        "bucket_description": "Truy vấn ngắn, thường chỉ 1-2 thuộc tính, ví dụ loại chỗ ở + địa điểm hoặc thêm 1 tiêu chí cơ bản.",
        "locations": ["đà nẵng", "sapa", "phú quốc", "nha trang"],
        "patterns": [
            "khách sạn 4 sao {loc}",
            "homestay {loc} giá rẻ",
            "resort {loc} có bể bơi",
            "villa {loc}",
            "hostel {loc}",
            "nhà nghỉ {loc} gần biển",
            "khách sạn {loc} gần sân bay",
            "khách sạn {loc} trung tâm",
            "homestay {loc} view đẹp",
            "resort {loc} sát biển",
        ],
    },
    {
        "bucket_id": "bucket_2_long_context",
        "bucket_name": "long_context_rich",
        "bucket_description": "Mô tả dài, có ngữ cảnh, nhiều điều kiện đồng thời như nhóm đi, ngân sách, tiện ích, mục đích chuyến đi.",
        "locations": ["hà nội", "đà lạt", "huế", "quảng bình"],
        "patterns": [
            "Tôi muốn tìm một chỗ nghỉ thoải mái, yên tĩnh, có bữa sáng và wifi mạnh cho chuyến công tác ở {loc}",
            "Cần homestay lãng mạn cho hai người để trốn cuối tuần, gần điểm ngắm bình minh ở {loc}, không quá xa trung tâm",
            "Nhóm 6 người cần thuê villa hoặc biệt thự có bếp nấu ăn và khu BBQ ở {loc}, ngân sách vừa phải",
            "Tìm resort gia đình ở {loc} có khu vui chơi trẻ em và hồ bơi cho trẻ em",
            "Muốn khách sạn ngay trung tâm {loc}, tiện đi bộ đến nhà hàng, an toàn cho đi một mình",
            "Cần khách sạn ở {loc} có phòng họp nhỏ cho 8 người, wifi mạnh, gần trung tâm triển lãm",
            "Tôi cần chỗ ở gần sân bay {loc}, có shuttle và có chính sách nhận phòng sớm",
            "Tìm place phù hợp để nghỉ dưỡng, có spa, hồ bơi vô cực, view biển ở {loc}",
            "Muốn homestay có không gian vintage, yên tĩnh, để đọc sách ở {loc}, giá phải chăng",
            "Cần khách sạn cho tuần trăng mật, phòng có ban công view biển, phục vụ ăn sáng riêng cho cặp đôi ở {loc}",
        ],
    },
    {
        "bucket_id": "bucket_3_geo_diverse",
        "bucket_name": "geo_diverse_priority_minor_provinces",
        "bucket_description": "Đa dạng vị trí, ưu tiên tỉnh/thành ít phổ biến hoặc ngoài các điểm du lịch lớn để kiểm tra khả năng hiểu địa điểm hiếm.",
        "locations": [
            "lạng sơn", "bắc kạn", "cao bằng", "hà giang", "lai châu", "điện biên", "sơn la", "yên bái",
            "quảng ngãi", "phú yên", "ninh thuận", "bình thuận", "bến tre", "sóc trăng", "bạc liêu", "cà mau",
            "kon tum", "gia lai", "đắk lắk", "đắk nông", "lâm đồng", "quảng trị", "quảng bình", "hà tĩnh",
            "nghệ an", "thanh hóa", "ninh bình", "thừa thiên huế", "hậu giang", "trà vinh", "an giang", "kiên giang",
            "đồng tháp", "tây ninh", "bình phước", "bình dương", "long an", "vĩnh long", "phú thọ", "hòa bình",
        ],
        "patterns": [
            "khách sạn ở {loc}",
            "homestay {loc} gần chợ",
            "resort {loc}",
            "nhà nghỉ {loc} cạnh sông",
            "khách sạn {loc} ven biển",
        ],
    },
    {
        "bucket_id": "bucket_4_natural_semantics",
        "bucket_name": "natural_language_semantic_queries",
        "bucket_description": "Câu truy vấn gần ngôn ngữ tự nhiên, diễn đạt mục tiêu/ngữ nghĩa thay vì chỉ nêu từ khóa.",
        "locations": ["đà lạt", "hà nội", "phú quốc", "huế"],
        "patterns": [
            "Tôi muốn một chỗ ở yên tĩnh, cách ồn ào thành phố, có phòng đọc sách và cà phê gần đó ở {loc}",
            "Muốn tìm khách sạn cho chuyến công tác 2 ngày ở {loc}, gần trung tâm có bữa sáng và wifi ổn định",
            "Cần chọn chỗ ở cho nhóm bạn thích party ban đêm nhưng cần nơi ngủ yên ở {loc}",
            "Tìm nơi phù hợp để tổ chức team building nhỏ ở {loc}, có sân vườn và chỗ BBQ gần trung tâm",
            "Muốn homestay phong cách tối giản, có cây xanh và không gian làm việc ở {loc}",
            "Tìm khách sạn gần bến tàu đến đảo ở {loc}, tiện cho đi lại và có chỗ gửi hành lý",
            "Cần khách sạn thân thiện cho gia đình có trẻ nhỏ ở {loc}, có cũi hoặc giường em bé",
            "Muốn resort spa, yên tĩnh, thích hợp cho tuần trăng mật ở {loc}",
            "Tìm chỗ ở cạnh khu ẩm thực ở {loc}, thích ăn uống địa phương và đi bộ được",
            "Cần khách sạn có dịch vụ đưa đón sân bay và giữ hành lý ở {loc}",
        ],
    },
    {
        "bucket_id": "bucket_5_random_mix",
        "bucket_name": "random_mix_robustness",
        "bucket_description": "Các truy vấn hỗn hợp/ngẫu nhiên, có thể ngắn, mơ hồ, thiếu thuộc tính hoặc pha nhiều tín hiệu khác nhau để kiểm tra độ bền của mô hình.",
        "locations": ["hcm", "đà lạt", "nha trang", "phú quốc"],
        "patterns": [
            "khách sạn {loc}",
            "homestay {loc}",
            "resort {loc}",
            "khách sạn gần sân bay {loc}",
            "villa {loc}",
            "hostel {loc}",
            "khách sạn giá rẻ {loc}",
            "khách sạn sang trọng {loc}",
            "ở {loc}",
            "chỗ nghỉ gần biển {loc}",
        ],
    },
]

records = []
qid = 1
target_per_bucket = 40
for bucket in BUCKETS:
    bucket_records = []
    locs = bucket["locations"]
    patterns = bucket["patterns"]
    for loc in locs:
        for pattern in patterns:
            if len(bucket_records) >= target_per_bucket:
                break
            bucket_records.append(
                {
                    "query_id": f"Q{qid:03d}",
                    "bucket_id": bucket["bucket_id"],
                    "bucket_name": bucket["bucket_name"],
                    "bucket_description": bucket["bucket_description"],
                    "bucket_order": BUCKETS.index(bucket) + 1,
                    "query": pattern.format(loc=loc),
                }
            )
            qid += 1
        if len(bucket_records) >= target_per_bucket:
            break
    records.extend(bucket_records)

# Safety: if any bucket produced fewer than 40 queries because of template count,
# pad that specific bucket with extra variants to reach the target while keeping
# the total at 200.
if len(records) < target_per_bucket * len(BUCKETS):
    extras = {
        "bucket_1_short": [
            ("short_1_2_attributes", "Truy vấn ngắn, thường chỉ 1-2 thuộc tính, ví dụ loại chỗ ở + địa điểm hoặc thêm 1 tiêu chí cơ bản.", "khách sạn {loc} gần trung tâm", "đà nẵng"),
            ("short_1_2_attributes", "Truy vấn ngắn, thường chỉ 1-2 thuộc tính, ví dụ loại chỗ ở + địa điểm hoặc thêm 1 tiêu chí cơ bản.", "homestay {loc} có bếp", "sapa"),
        ],
        "bucket_2_long_context": [
            ("long_context_rich", "Mô tả dài, có ngữ cảnh, nhiều điều kiện đồng thời như nhóm đi, ngân sách, tiện ích, mục đích chuyến đi.", "Tôi cần chỗ ở cho chuyến du lịch cuối tuần ở {loc}, ưu tiên phòng sạch, gần quán ăn và có chỗ đậu xe", "đà lạt"),
        ],
        "bucket_3_geo_diverse": [
            ("geo_diverse_priority_minor_provinces", "Đa dạng vị trí, ưu tiên tỉnh/thành ít phổ biến hoặc ngoài các điểm du lịch lớn để kiểm tra khả năng hiểu địa điểm hiếm.", "khách sạn {loc}", "sơn la"),
        ],
        "bucket_4_natural_semantics": [
            ("natural_language_semantic_queries", "Câu truy vấn gần ngôn ngữ tự nhiên, diễn đạt mục tiêu/ngữ nghĩa thay vì chỉ nêu từ khóa.", "Tôi muốn một nơi nghỉ vừa đủ tiện nghi để làm việc từ xa ở {loc}, không quá đông và có wifi ổn định", "hà nội"),
        ],
        "bucket_5_random_mix": [
            ("random_mix_robustness", "Các truy vấn hỗn hợp/ngẫu nhiên, có thể ngắn, mơ hồ, thiếu thuộc tính hoặc pha nhiều tín hiệu khác nhau để kiểm tra độ bền của mô hình.", "khách sạn {loc} gần biển", "nha trang"),
        ],
    }
    by_bucket = {b["bucket_id"]: [r for r in records if r["bucket_id"] == b["bucket_id"]] for b in BUCKETS}
    for bucket in BUCKETS:
        bid = bucket["bucket_id"]
        while len(by_bucket[bid]) < target_per_bucket:
            pad_list = extras.get(bid, [])
            if not pad_list:
                break
            pad_idx = len(by_bucket[bid]) % len(pad_list)
            _bname, _bdesc, pattern, loc = pad_list[pad_idx]
            by_bucket[bid].append(
                {
                    "query_id": f"Q{qid:03d}",
                    "bucket_id": bid,
                    "bucket_name": bucket["bucket_name"],
                    "bucket_description": bucket["bucket_description"],
                    "bucket_order": BUCKETS.index(bucket) + 1,
                    "query": pattern.format(loc=loc),
                }
            )
            qid += 1
    records = []
    for bucket in BUCKETS:
        records.extend(by_bucket[bucket["bucket_id"]])

# Re-number query IDs sequentially after padding/regrouping.
for idx, item in enumerate(records, start=1):
    item["query_id"] = f"Q{idx:03d}"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(records)} queries to {OUT}")
