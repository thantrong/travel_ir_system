from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.review_tagger import tag_review


def main():
    samples = [
        {
            "hotel_name": "Muong Thanh Grand Tuyen Quang Hotel",
            "location": "Tuyên Quang",
            "review_text": "Ăn sáng ngon. Đồ ăn đa dạng. Thích nhất là quầy bánh cuốn nóng. Vị trí trung tâm. Nói chung là ok",
        },
        {
            "hotel_name": "Sea View Resort",
            "location": "Phú Quốc",
            "review_text": "Hồ bơi đẹp, view biển rất thích, sát biển và nhân viên thân thiện",
        },
        {
            "hotel_name": "Home Stay Đà Lạt",
            "location": "Đà Lạt",
            "review_text": "Phòng sạch sẽ, yên tĩnh, rộng rãi và giá khá hợp lý",
        },
    ]

    for i, sample in enumerate(samples, 1):
        res = tag_review(
            sample["review_text"],
            hotel_name=sample["hotel_name"],
            location=sample["location"],
        )
        print(f"\n=== Sample {i} ===")
        print("Hotel:", sample["hotel_name"])
        print("Location:", sample["location"])
        print("Review:", sample["review_text"])
        print("Category tags:", res.category_tags)
        print("Descriptor tags:", res.descriptor_tags)
        print("Matched phrases:", res.matched_phrases)


if __name__ == "__main__":
    main()
