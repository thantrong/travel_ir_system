import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_HOTEL_ROOT = PROJECT_ROOT / "data" / "raw" / "hotel_detail"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "hotel_detail_clean"


AMENITY_KEYWORDS = [
    "wifi",
    "chỗ đậu xe",
    "đậu xe",
    "lễ tân 24 giờ",
    "hồ bơi",
    "nhà hàng",
    "quầy bar",
    "điều hòa",
    "thang máy",
    "gym",
    "spa",
    "đưa đón sân bay",
    "giặt ủi",
    "phòng gia đình",
    "ban công",
    "cho phép thú cưng",
    "không hút thuốc",
    "bữa sáng",
]

PROPERTY_TYPE_MAP = {
    "homestay": "homestay",
    "resort": "resort",
    "villa": "villa",
    "hostel": "hostel",
    "boutique": "boutique_hotel",
    "apartment": "aparthotel",
    "hotel": "hotel",
}


def normalize_space(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def parse_jsonld_objects(soup: BeautifulSoup) -> list[dict]:
    objs = []
    for script in soup.select("script[type='application/ld+json']"):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    objs.append(item)
        elif isinstance(payload, dict):
            objs.append(payload)
    return objs


def pick_hotel_jsonld(objs: list[dict]) -> dict:
    for obj in objs:
        t = str(obj.get("@type", "")).lower()
        if "hotel" in t:
            return obj
    return {}


def extract_address(hotel_obj: dict) -> str:
    addr = hotel_obj.get("address", {})
    if isinstance(addr, dict):
        # Theo yêu cầu: chỉ lấy địa chỉ chính xác từ streetAddress.
        return normalize_space(str(addr.get("streetAddress", "")))
    if isinstance(addr, str):
        return normalize_space(addr)
    return ""


def infer_property_type(name: str, title_text: str, hotel_obj: dict) -> str:
    t = str(hotel_obj.get("@type", "")).lower()
    hay = f"{name} {title_text} {t}".lower()
    for k, v in PROPERTY_TYPE_MAP.items():
        if k in hay:
            return v
    return "hotel"


def infer_hotel_category_tags(name: str, title_text: str, property_type: str) -> list[str]:
    hay = f"{name} {title_text}".lower()
    tags = [property_type]
    if "3 sao" in hay or "3-star" in hay:
        tags.append("3_star")
    if "4 sao" in hay or "4-star" in hay:
        tags.append("4_star")
    if "5 sao" in hay or "5-star" in hay:
        tags.append("5_star")
    if "gần trung tâm" in hay:
        tags.append("near_center")
    return list(dict.fromkeys(tags))


def extract_star_rating(hotel_obj: dict, title_text: str) -> str:
    rating = ""
    for key_path in [
        ("starRating", "ratingValue"),
        ("aggregateRating", "ratingValue"),
    ]:
        cur = hotel_obj
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            rating = normalize_space(str(cur))
            break

    if not rating:
        m = re.search(r"([1-5])\s*(?:sao|star)", title_text.lower())
        if m:
            rating = m.group(1)
    return rating


def extract_images(hotel_obj: dict) -> list[str]:
    images = hotel_obj.get("image", [])
    out = []
    if isinstance(images, str):
        images = [images]
    if isinstance(images, list):
        for img in images:
            v = normalize_space(str(img))
            if v.startswith("http") and v not in out:
                out.append(v)
    return out


def extract_images_from_html(html: str, soup: BeautifulSoup) -> list[str]:
    """
    Bổ sung ảnh từ HTML thực tế để không bị thiếu gallery.
    Ưu tiên URL ảnh khách sạn từ imagekit/traveloka.
    """
    urls = []

    # 1) Từ thẻ img
    for img in soup.find_all("img"):
        src = normalize_space(str(img.get("src", "")))
        if src.startswith("http"):
            urls.append(src)

    # 2) Quét trực tiếp trong HTML các URL ảnh phổ biến
    # Giữ logic rộng vừa phải để bắt gallery ở các blob/script render.
    pattern = r"https?://[^\"'\\s>]+(?:imagekit|traveloka)[^\"'\\s>]+(?:\\.jpg|\\.jpeg|\\.png|\\.webp)(?:\\?[^\"'\\s>]*)?"
    urls.extend(re.findall(pattern, html, flags=re.IGNORECASE))

    # Dedupe + lọc rác
    cleaned = []
    seen_keys = set()
    for u in urls:
        v = normalize_space(u)
        if not v.startswith("http"):
            continue
        low = v.lower()
        if "avatar" in low:
            continue
        # Bỏ icon/logo/svg/tiny thumb
        if low.endswith(".svg") or ".svg?" in low:
            continue
        if "logo" in low or "icon" in low or "/favicon" in low:
            continue
        # Ưu tiên asset ảnh khách sạn thực tế
        if "hotel/asset/" not in low and "hotel/" not in low:
            continue
        # Dedupe theo "asset gốc", bỏ khác biệt query resize/crop.
        key = re.sub(r"\?.*$", "", v)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if v not in cleaned:
            cleaned.append(v)
    return cleaned


def extract_asset_filenames_from_html(html: str) -> list[str]:
    """
    Bắt các filename dạng hotel/asset/<id-hash>.<ext> xuất hiện trong HTML/script,
    kể cả khi URL bị escape.
    """
    if not html:
        return []
    matches = re.findall(
        r"hotel/asset/([0-9]+-[0-9a-f]{32}\.(?:jpeg|jpg|png|webp))",
        html,
        flags=re.IGNORECASE,
    )
    out = []
    for m in matches:
        v = normalize_space(m)
        if v and v not in out:
            out.append(v)
    return out


def build_image_urls_from_assets(asset_files: list[str], seed_images: list[str]) -> list[str]:
    if not asset_files or not seed_images:
        return []
    # Lấy prefix từ ảnh có sẵn (trước /hotel/asset/)
    prefix = ""
    for u in seed_images:
        if "/hotel/asset/" in u:
            prefix = u.split("/hotel/asset/")[0]
            break
    if not prefix:
        return []

    out = []
    for fname in asset_files:
        # URL canonical để hiển thị/parse ổn định
        url = f"{prefix}/hotel/asset/{fname}?tr=q-80,c-at_max,w-1280,h-720&_src=imagekit"
        if url not in out:
            out.append(url)
    return out


def infer_amenities(page_text: str) -> list[str]:
    txt = page_text.lower()
    found = []
    for kw in AMENITY_KEYWORDS:
        if kw in txt:
            found.append(kw)
    return list(dict.fromkeys(found))


def extract_policies(page_text: str) -> dict:
    txt = normalize_space(page_text)
    lowered = txt.lower()
    policies = {}

    # check-in/out
    ci = re.search(r"(nhận phòng|check[- ]?in)\s*(từ|from)?\s*([0-2]?\d[:h][0-5]?\d?)", lowered)
    co = re.search(r"(trả phòng|check[- ]?out)\s*(trước|before|đến)?\s*([0-2]?\d[:h][0-5]?\d?)", lowered)
    if ci:
        policies["check_in"] = ci.group(3)
    if co:
        policies["check_out"] = co.group(3)

    # simple booleans by phrases
    if "không hút thuốc" in lowered:
        policies["smoking"] = "no_smoking"
    elif "hút thuốc" in lowered:
        policies["smoking"] = "smoking_allowed_or_partial"

    if "cho phép thú cưng" in lowered:
        policies["pets"] = "allowed"
    elif "không cho phép thú cưng" in lowered:
        policies["pets"] = "not_allowed"

    if "trẻ em" in lowered:
        policies["child_policy"] = "available"

    return policies


def extract_checkin_checkout(page_text: str) -> tuple[str, str]:
    txt = normalize_space(page_text).lower()
    # Hỗ trợ các dạng 14:00, 14h00, 14h
    time_pattern = r"([0-2]?\d(?::|h)[0-5]?\d?)"

    ci = ""
    co = ""

    m_ci = re.search(rf"(nhận phòng|check[- ]?in).{{0,40}}?{time_pattern}", txt)
    if m_ci:
        ci = normalize_space(m_ci.group(2))

    m_co = re.search(rf"(trả phòng|check[- ]?out).{{0,40}}?{time_pattern}", txt)
    if m_co:
        co = normalize_space(m_co.group(2))

    return ci, co


def extract_checkin_checkout_from_faq(faq_items: list[dict]) -> tuple[str, str]:
    ci = ""
    co = ""
    for faq in faq_items:
        q = normalize_space(str(faq.get("question", ""))).lower()
        a = normalize_space(str(faq.get("answer", ""))).lower()
        if ("nhận phòng" in q or "check in" in q or "check-in" in q) and ("trả phòng" in q or "check out" in q or "check-out" in q):
            m_ci = re.search(r"(từ|from)\s*([0-2]?\d(?::|h)[0-5]?\d?)", a)
            m_co = re.search(r"(trước|before)\s*([0-2]?\d(?::|h)[0-5]?\d?)", a)
            if m_ci:
                ci = normalize_space(m_ci.group(2))
            if m_co:
                co = normalize_space(m_co.group(2))
            break
    return ci, co


def extract_city(hotel_obj: dict, address: str) -> str:
    addr = hotel_obj.get("address", {})
    if isinstance(addr, dict):
        city = normalize_space(str(addr.get("addressLocality", "")))
        if city:
            return city
        region = normalize_space(str(addr.get("addressRegion", "")))
        if region:
            return region
    if address:
        parts = [normalize_space(p) for p in address.split(",") if normalize_space(p)]
        if len(parts) >= 2:
            return parts[-2]
        if parts:
            return parts[-1]
    return ""


def extract_city_from_jsonld(objs: list[dict], address: str) -> str:
    # 1) Ưu tiên breadcrumb city-level
    for obj in objs:
        if str(obj.get("@type", "")).lower() != "breadcrumblist":
            continue
        elements = obj.get("itemListElement", [])
        if not isinstance(elements, list):
            continue
        for el in elements:
            if not isinstance(el, dict):
                continue
            item = el.get("item", {})
            name = ""
            item_id = ""
            if isinstance(item, dict):
                name = normalize_space(str(item.get("name", "")))
                item_id = normalize_space(str(item.get("@id", ""))).lower()
            if "/city/" in item_id and name:
                return name

    # 2) Heuristic từ address text
    known_cities = [
        "Đà Lạt",
        "Phú Quốc",
        "Hội An",
        "Đà Nẵng",
        "Nha Trang",
        "Hồ Chí Minh",
        "Hà Nội",
    ]
    lowered_addr = normalize_space(address).lower()
    for city in known_cities:
        if city.lower() in lowered_addr:
            return city
    return ""


def extract_accommodation_policy_modal_text(soup: BeautifulSoup) -> str:
    """
    Văn trong modal 'Chính sách lưu trú' ([role=dialog]) — chỉ có sau khi crawler
    đã bấm 'Đọc tất cả' và lưu HTML.
    """
    dialogs = soup.select('[role="dialog"]')
    if not dialogs:
        return ""
    best = ""
    for d in dialogs:
        t = normalize_space(d.get_text(" ", strip=True))
        if len(t) > len(best):
            best = t
    return best


def extract_summary_description_from_html(soup: BeautifulSoup) -> str:
    node = soup.select_one("article#summary-description[role='article']")
    if node:
        text = normalize_space(node.get_text(" ", strip=True))
        if text:
            return text
    return ""


def extract_faq_items(objs: list[dict]) -> list[dict]:
    faq_rows = []
    for obj in objs:
        if str(obj.get("@type", "")).lower() != "faqpage":
            continue
        entities = obj.get("mainEntity", [])
        if not isinstance(entities, list):
            continue
        for idx, item in enumerate(entities, start=1):
            if not isinstance(item, dict):
                continue
            q = normalize_space(str(item.get("name", "")))
            answer_obj = item.get("acceptedAnswer", {})
            a = ""
            if isinstance(answer_obj, dict):
                a = normalize_space(str(answer_obj.get("text", "")))
            if not q or not a:
                continue
            faq_rows.append(
                {
                    "question": q,
                    "answer": a,
                }
            )
    return faq_rows


def process_hotel_folder(hotel_dir: Path) -> tuple[dict | None, list[dict]]:
    hotel_id = hotel_dir.name
    html_candidates = sorted(hotel_dir.glob("*.html"), reverse=True)
    if not html_candidates:
        return None, []

    html_path = html_candidates[0]
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    objs = parse_jsonld_objects(soup)
    hotel_obj = pick_hotel_jsonld(objs)

    title_text = normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")
    name = normalize_space(str(hotel_obj.get("name", ""))) or title_text.split(",")[0].strip()
    description = extract_summary_description_from_html(soup) or normalize_space(str(hotel_obj.get("description", "")))
    address = extract_address(hotel_obj)
    city = extract_city_from_jsonld(objs, address) or extract_city(hotel_obj, address)
    property_type = infer_property_type(name, title_text, hotel_obj)
    star_rating = extract_star_rating(hotel_obj, title_text)
    category_tags = infer_hotel_category_tags(name, title_text, property_type)
    images = extract_images(hotel_obj)
    images_html = extract_images_from_html(html, soup)
    asset_files = extract_asset_filenames_from_html(html)
    images_from_assets = build_image_urls_from_assets(asset_files, images + images_html)
    merged_images = images + images_html + images_from_assets
    images = []
    seen_img_keys = set()
    for url in merged_images:
        v = normalize_space(url)
        if not v:
            continue
        key = re.sub(r"\?.*$", "", v)
        if key in seen_img_keys:
            continue
        seen_img_keys.add(key)
        images.append(v)

    page_text = normalize_space(soup.get_text(" ", strip=True))
    amenities = infer_amenities(page_text)
    policies = extract_policies(page_text)
    policy_modal_text = extract_accommodation_policy_modal_text(soup)
    faq_items = extract_faq_items(objs)
    check_in_time, check_out_time = extract_checkin_checkout_from_faq(faq_items)
    if not check_in_time and not check_out_time:
        check_in_time, check_out_time = extract_checkin_checkout(page_text)

    hotel_doc = {
        "_id": hotel_id,
        "hotel_id": hotel_id,
        "name": name,
        "address": address,
        "city": city,
        "star_rating": star_rating,
        "property_type": property_type,
        "hotel_category_tags": category_tags,
        "amenities": amenities,
        "policies": policies,
        "policy_modal_text": policy_modal_text,
        "check_in_time": check_in_time,
        "check_out_time": check_out_time,
        "images": images,
        "description": description,
        "source": "traveloka",
    }
    return hotel_doc, faq_items


def main():
    parser = argparse.ArgumentParser(description="Process raw hotel detail HTML to clean schema JSON")
    parser.add_argument("--input-dir", type=str, default=str(RAW_HOTEL_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hotel_clean_dir = output_dir / "hotel_clean"
    faq_dir = output_dir / "faq"
    hotel_clean_dir.mkdir(parents=True, exist_ok=True)
    faq_dir.mkdir(parents=True, exist_ok=True)

    hotels = []
    faq_map = {}
    hotel_dirs = [p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    hotel_dirs.sort(key=lambda p: p.name)

    for hotel_dir in hotel_dirs:
        row, faq_items = process_hotel_folder(hotel_dir)
        if not row:
            continue
        hotels.append(row)
        hid = row["hotel_id"]
        faq_map[hid] = {
            "_id": hid,
            "hotel_id": hid,
            "hotel_name": row["name"],
            "faqs": faq_items,
            "source": "traveloka",
        }
        (hotel_clean_dir / f"{row['hotel_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for hid, faq_doc in faq_map.items():
        (faq_dir / f"{hid}.json").write_text(
            json.dumps(faq_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Xóa file tổng hợp cũ (nếu có) để tránh nhầm luồng sử dụng.
    legacy_hotels = output_dir / "hotels_clean.json"
    legacy_faq = output_dir / "hotel_faq_clean.json"
    if legacy_hotels.exists():
        legacy_hotels.unlink()
    if legacy_faq.exists():
        legacy_faq.unlink()

    print(f"Processed hotels: {len(hotels)}")
    print(f"Processed hotel FAQ docs: {len(faq_map)}")
    print(f"Hotel clean dir: {hotel_clean_dir}")
    print(f"FAQ dir: {faq_dir}")


if __name__ == "__main__":
    main()

