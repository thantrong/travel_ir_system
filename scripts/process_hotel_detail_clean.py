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


def extract_accommodation_policy_inline_text(soup: BeautifulSoup) -> str:
    """
    Fallback khi không có modal 'Đọc tất cả':
    lấy policy hiển thị inline trong section/tab Chính sách.
    """
    candidates = []
    # Ưu tiên section có id/testid chứa policy
    for node in soup.select(
        "section[id*='policy' i],"
        "div[id*='policy' i],"
        "section[data-testid*='policy' i],"
        "div[data-testid*='policy' i]"
    ):
        txt = normalize_space(node.get_text(" ", strip=True))
        if txt:
            candidates.append(txt)

    # Fallback text-anchor quanh tiêu đề "Chính sách"
    if not candidates:
        for tag in soup.find_all(string=re.compile(r"Chính\s*Sách|Chính sách", re.IGNORECASE)):
            parent = tag.parent
            if not parent:
                continue
            block = parent.find_parent(["section", "article", "div"]) or parent
            txt = normalize_space(block.get_text(" ", strip=True))
            if txt:
                candidates.append(txt)

    if not candidates:
        return ""

    # Chọn block dài nhất có dấu hiệu policy thực
    best = ""
    for c in candidates:
        low = c.lower()
        if ("nhận phòng" in low or "check-in" in low or "trả phòng" in low or "chính sách" in low) and len(c) > len(best):
            best = c
    if best:
        return best
    return max(candidates, key=len)


def _extract_money_vnd(text: str) -> str:
    m = re.search(r"([0-9][0-9\.,]*)\s*VND|VND\s*([0-9][0-9\.,]*)", text, flags=re.IGNORECASE)
    if not m:
        return ""
    raw = m.group(1) or m.group(2) or ""
    return normalize_space(raw.replace(".", ","))


def extract_policy_structured(policy_text: str) -> dict:
    """
    Chuẩn hóa policy từ modal để tránh lộn xộn giữa các block VI/EN.
    Ưu tiên block "Chính Sách Bổ Sung" (thường là bản cập nhật mới hơn).
    """
    if not policy_text:
        return {}

    txt = normalize_space(policy_text)
    out = {}

    ci = re.search(r"Giờ nhận phòng:\s*Từ\s*([0-2]?\d:[0-5]\d)", txt, flags=re.IGNORECASE)
    co = re.search(r"Giờ trả phòng:\s*Trước\s*([0-2]?\d:[0-5]\d)", txt, flags=re.IGNORECASE)
    if ci:
        out["check_in_time"] = ci.group(1)
    if co:
        out["check_out_time"] = co.group(1)

    m_breakfast = re.search(
        r"Bữa sáng bổ sung.*?(VND\s*[0-9][0-9\.,]*|[0-9][0-9\.,]*\s*VND)",
        txt,
        flags=re.IGNORECASE,
    )
    if m_breakfast:
        out["breakfast_extra_fee_vnd"] = _extract_money_vnd(m_breakfast.group(0))

    m_airport = re.search(
        r"Đưa đón sân bay.*?(VND\s*[0-9][0-9\.,]*|[0-9][0-9\.,]*\s*VND)",
        txt,
        flags=re.IGNORECASE,
    )
    if m_airport:
        out["airport_transfer_fee_vnd"] = _extract_money_vnd(m_airport.group(0))

    # Ưu tiên block "Chính Sách Bổ Sung" để tránh mâu thuẫn mốc tuổi từ block cũ.
    child_block = ""
    m_child_new = re.search(
        r"Chính Sách Bổ Sung(.*?)(Child's Policy|Chính sách giường phụ|Extra bed Policy|Đưa đón sân bay|$)",
        txt,
        flags=re.IGNORECASE,
    )
    if m_child_new:
        child_block = normalize_space(m_child_new.group(1))
    if not child_block:
        m_child_old = re.search(
            r"Hướng Dẫn Nhận Phòng Chung(.*?)(Bữa sáng bổ sung|Chính Sách Bổ Sung|$)",
            txt,
            flags=re.IGNORECASE,
        )
        if m_child_old:
            child_block = normalize_space(m_child_old.group(1))
    if child_block:
        out["child_policy_text"] = child_block

    m_extra_bed_vi = re.search(
        r"Chính sách giường phụ:\s*(.*?)(Extra bed Policy|Đưa đón sân bay|$)",
        txt,
        flags=re.IGNORECASE,
    )
    if m_extra_bed_vi:
        out["extra_bed_policy_text"] = normalize_space(m_extra_bed_vi.group(1))
    else:
        m_extra_bed_en = re.search(
            r"Extra bed Policy:\s*(.*?)(Đưa đón sân bay|$)",
            txt,
            flags=re.IGNORECASE,
        )
        if m_extra_bed_en:
            out["extra_bed_policy_text"] = normalize_space(m_extra_bed_en.group(1))

    return out


def extract_nearby_attractions(page_text: str, policy_text: str) -> list[dict]:
    """
    Trích các địa điểm gần khách sạn cùng khoảng cách (km) theo best-effort.
    """
    source = normalize_space(f"{page_text} {policy_text}")
    if not source:
        return []

    pattern = re.compile(
        r"([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9'’\-/\s]{2,48}?)\s+([0-9]+(?:[.,][0-9]+)?)\s*km\b",
        flags=re.IGNORECASE,
    )
    out = []
    seen = set()
    for m in pattern.finditer(source):
        name = normalize_space(m.group(1))
        dist_raw = normalize_space(m.group(2)).replace(",", ".")
        try:
            dist = float(dist_raw)
        except Exception:
            continue

        low = name.lower()
        noise = (
            "đánh giá",
            "số lượng",
            "xếp hạng",
            "nhận phòng",
            "trả phòng",
            "vnd",
            "wifi",
            "bữa sáng",
            "chính sách",
        )
        if any(n in low for n in noise):
            continue
        if len(name) < 3 or len(name) > 48:
            continue
        # Loại các cụm bị dính / không phải tên địa điểm đơn.
        if " m " in low or "  " in name:
            continue
        # Loại tên mở đầu bằng cụm mô tả không phải điểm đến cụ thể.
        bad_prefixes = ("gần ", "trung tâm ", "khoảng cách ", "khác ", "ơi ", "ảo ", "cách ")
        if low.startswith(bad_prefixes):
            continue
        # Loại cụm mô tả khoảng cách chung, không phải điểm đến.
        if low.endswith("khoảng") or " khoảng " in low:
            continue
        if not name[0].isalpha() or not name[0].isupper():
            continue

        key = (name.lower(), round(dist, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "distance_km": dist})

    out.sort(key=lambda x: x.get("distance_km", 10**9))
    return out


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


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        v = str(value).strip().replace(",", ".")
        if not v:
            return None
        return float(v)
    except Exception:
        return None


def _normalize_price_payload(rate_display: dict) -> dict:
    if not isinstance(rate_display, dict):
        return {}
    total = (rate_display.get("totalFare") or {}).get("amount")
    base = (rate_display.get("baseFare") or {}).get("amount")
    taxes = (rate_display.get("taxes") or {}).get("amount")
    currency = (rate_display.get("totalFare") or {}).get("currency") or (rate_display.get("baseFare") or {}).get("currency")
    return {
        "currency": normalize_space(str(currency)) if currency else "",
        "total_fare": _to_int_or_none(total),
        "base_fare": _to_int_or_none(base),
        "taxes": _to_int_or_none(taxes),
    }


def process_room_types_folder(hotel_dir: Path) -> dict | None:
    hotel_id = hotel_dir.name
    room_candidates = sorted(hotel_dir.glob("rooms_*.json"), reverse=True)
    if not room_candidates:
        return None
    room_path = room_candidates[0]
    try:
        raw = json.loads(room_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    responses = raw.get("responses") or []
    room_map = {}
    for resp in responses:
        payload = (resp or {}).get("payload") or {}
        data = payload.get("data") or {}
        entries = data.get("recommendedEntries") or []
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rid = normalize_space(str(entry.get("hotelRoomId", "")))
            if not rid:
                continue
            name = normalize_space(str(entry.get("name", "")))
            desc = normalize_space(str(entry.get("description", "")))
            room_images = []
            for u in entry.get("roomImages") or []:
                uv = normalize_space(str(u))
                if uv and uv not in room_images:
                    room_images.append(uv)

            amenity_names = []
            abc = entry.get("amenitiesByCategory") or {}
            if isinstance(abc, dict):
                for _, arr in abc.items():
                    if not isinstance(arr, list):
                        continue
                    for item in arr:
                        if not isinstance(item, dict):
                            continue
                        nm = normalize_space(str(item.get("name", "")))
                        if nm and nm not in amenity_names:
                            amenity_names.append(nm)

            room_obj = room_map.setdefault(
                rid,
                {
                    "room_id": rid,
                    "room_name": name,
                    "description": desc,
                    # room_size chỉ là số (m2), không lưu đơn vị.
                    "room_size": None,
                    "base_occupancy": entry.get("baseOccupancy"),
                    "images": room_images,
                    "amenities": amenity_names,
                    "rate_options": [],
                },
            )
            if not room_obj.get("room_name") and name:
                room_obj["room_name"] = name
            if not room_obj.get("description") and desc:
                room_obj["description"] = desc
            for img in room_images:
                if img not in room_obj["images"]:
                    room_obj["images"].append(img)
            for am in amenity_names:
                if am not in room_obj["amenities"]:
                    room_obj["amenities"].append(am)

            # Kích thước phòng
            size_display = entry.get("hotelRoomSizeDisplay") or {}
            if isinstance(size_display, dict):
                room_obj["room_size"] = _to_float_or_none(size_display.get("value"))

            # Bố trí giường (best-effort)
            bed = entry.get("bedArrangements")
            bed_obj = {}
            if isinstance(bed, dict):
                for k in ("displayBedType", "displayName", "bedType", "name"):
                    vv = normalize_space(str(bed.get(k, "")))
                    if vv:
                        bed_obj[k] = vv
            if bed_obj:
                room_obj["bed_arrangements"] = bed_obj

            inv_list = entry.get("hotelRoomInventoryList") or []
            if not isinstance(inv_list, list):
                continue
            for inv in inv_list:
                if not isinstance(inv, dict):
                    continue
                final_price = (inv.get("finalPrice") or {}).get("totalPriceRateDisplay") or {}
                rate = {
                    "inventory_id": normalize_space(str(inv.get("hotelRoomInventoryId", ""))),
                    "inventory_name": normalize_space(str(inv.get("inventoryName", ""))),
                    "is_breakfast_included": bool(inv.get("isBreakfastIncluded", False)),
                    "is_refundable": bool(inv.get("isRefundable", False)),
                    "num_remaining_rooms": _to_int_or_none(inv.get("numRemainingRooms")),
                    "price": _normalize_price_payload(final_price),
                }
                key = (
                    rate.get("inventory_id", ""),
                    rate.get("inventory_name", ""),
                    rate.get("price", {}).get("total_fare"),
                )
                if key not in {
                    (
                        r.get("inventory_id", ""),
                        r.get("inventory_name", ""),
                        (r.get("price") or {}).get("total_fare"),
                    )
                    for r in room_obj["rate_options"]
                }:
                    room_obj["rate_options"].append(rate)

    room_types = list(room_map.values())
    for room in room_types:
        # Ước tính số lượng phòng theo yêu cầu: max số phòng còn của các rate option.
        remaining = [
            _to_int_or_none((opt or {}).get("num_remaining_rooms"))
            for opt in (room.get("rate_options") or [])
        ]
        remaining = [x for x in remaining if x is not None]
        room["estimated_total_rooms"] = max(remaining) if remaining else None

        # Không để field rỗng trong dữ liệu sạch.
        if not room.get("bed_arrangements"):
            room.pop("bed_arrangements", None)

    return {
        "_id": hotel_id,
        "hotel_id": hotel_id,
        "room_types": room_types,
        "source": "traveloka",
        "raw_room_file": str(room_path),
    }


def process_hotel_folder(hotel_dir: Path) -> tuple[dict | None, list[dict], dict]:
    hotel_id = hotel_dir.name
    html_candidates = sorted(hotel_dir.glob("*.html"), reverse=True)
    if not html_candidates:
        return None, [], {}
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
    if not policy_modal_text:
        policy_modal_text = extract_accommodation_policy_inline_text(soup)
    policy_structured = extract_policy_structured(policy_modal_text)
    faq_items = extract_faq_items(objs)
    nearby_attractions = extract_nearby_attractions(page_text, policy_modal_text)
    check_in_time, check_out_time = extract_checkin_checkout_from_faq(faq_items)
    if not check_in_time:
        check_in_time = normalize_space(str(policy_structured.get("check_in_time", "")))
    if not check_out_time:
        check_out_time = normalize_space(str(policy_structured.get("check_out_time", "")))
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
        "nearby_attractions": nearby_attractions,
        "check_in_time": check_in_time,
        "check_out_time": check_out_time,
        "images": images,
        "description": description,
        "source": "traveloka",
    }
    policy_payload = {
        "policies": policies,
        "policy_text": policy_modal_text,
        "policy_structured": policy_structured,
    }
    return hotel_doc, faq_items, policy_payload


def main():
    parser = argparse.ArgumentParser(description="Process raw hotel detail HTML to clean schema JSON")
    parser.add_argument("--input-dir", type=str, default=str(RAW_HOTEL_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hotel_clean_dir = output_dir / "hotel_clean"
    policy_dir = output_dir / "policy_fag"
    room_type_dir = output_dir / "room_types"
    hotel_clean_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    room_type_dir.mkdir(parents=True, exist_ok=True)

    hotels = []
    policy_map = {}
    room_type_map = {}
    hotel_dirs = [p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    hotel_dirs.sort(key=lambda p: p.name)

    for hotel_dir in hotel_dirs:
        row, faq_items, policy_payload = process_hotel_folder(hotel_dir)
        if not row:
            continue
        hotels.append(row)
        hid = row["hotel_id"]
        policy_map[hid] = {
            "_id": hid,
            "hotel_id": hid,
            "hotel_name": row["name"],
            "policy_text": policy_payload.get("policy_text", ""),
            "policy_structured": policy_payload.get("policy_structured", {}),
            "policies": policy_payload.get("policies", {}),
            # Gộp toàn bộ FAQ vào policy doc để tránh tách dữ liệu trùng.
            "policy_related_faqs": faq_items,
            "source": "traveloka",
        }
        room_doc = process_room_types_folder(hotel_dir)
        if room_doc:
            room_type_map[hid] = room_doc
        (hotel_clean_dir / f"{row['hotel_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for hid, policy_doc in policy_map.items():
        (policy_dir / f"{hid}.json").write_text(
            json.dumps(policy_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for hid, room_doc in room_type_map.items():
        (room_type_dir / f"{hid}.json").write_text(
            json.dumps(room_doc, ensure_ascii=False, indent=2),
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
    print(f"Processed hotel policy docs: {len(policy_map)}")
    print(f"Processed hotel room-type docs: {len(room_type_map)}")
    print(f"Hotel clean dir: {hotel_clean_dir}")
    print(f"Policy dir: {policy_dir}")
    print(f"Room type dir: {room_type_dir}")


if __name__ == "__main__":
    main()

