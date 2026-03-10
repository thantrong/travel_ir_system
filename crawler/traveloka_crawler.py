"""
Traveloka Hotel Review Crawler v3
Fix: Hotel cards không có <a> tag → phải click vào div card để navigate.
Fix: URL encoding cho tên thành phố tiếng Việt.
"""
import asyncio
import hashlib
import json
import yaml
import random
import logging
import re
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
try:
    from playwright_stealth import stealth_async
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

# ============================================================
# SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cfg = config.get("crawler", {}).get("traveloka", {})
DELAY = cfg.get("delay", 3)
HEADLESS = cfg.get("headless", False)
TIMEOUT = cfg.get("timeout", 60000)
MAX_HOTELS = max(1, int(cfg.get("max_hotels_per_city", 100)))
MAX_REVIEWS = max(1, int(cfg.get("max_reviews_per_hotel", 100)))
MAX_CONCURRENT_HOTELS = max(1, int(cfg.get("max_concurrent_hotels", 2)))
MAX_CONCURRENT_CITIES = max(1, int(cfg.get("max_concurrent_cities", 2)))
BLOCK_RESOURCE_TYPES = set(cfg.get("block_resource_types", ["image", "media", "font"]))
COOLDOWN_EVERY_HOTELS = max(0, int(cfg.get("cooldown_every_hotels", 0)))
COOLDOWN_SECONDS = max(0, int(cfg.get("cooldown_seconds", 0)))
DETAIL_TASK_TIMEOUT_SECONDS = max(20, int(cfg.get("detail_task_timeout_seconds", 90)))
CITY_TASK_TIMEOUT_SECONDS = max(60, int(cfg.get("city_task_timeout_seconds", 600)))
CITIES_FILE = cfg.get("cities_file", "config/cities.yaml")
PROXY_CFG = cfg.get("proxy", {}) if isinstance(cfg.get("proxy", {}), dict) else {}
FAST_CRAWL = bool(cfg.get("fast_crawl", False))
FAST_MIN_DELAY_SECONDS = max(0.0, float(cfg.get("fast_min_delay_seconds", 0.05)))
FAST_MAX_DELAY_SECONDS = max(FAST_MIN_DELAY_SECONDS, float(cfg.get("fast_max_delay_seconds", 0.25)))
SKIP_IDLE_SECONDS = max(0.0, float(cfg.get("skip_idle_seconds", 20.0)))
CITIES = []

THIRD_PARTY_TRACKING_PATTERNS = (
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "googleadservices.com",
    "analytics.google.com",
    "amplitude.com",
    "pinterest.com",
    "spotify.com",
    "reddit.com",
    "datadoghq.com",
    "braze.com",
    "daum.net",
    "slim02.jp",
)

NON_ESSENTIAL_SAME_ORIGIN_API_PATTERNS = (
    "/api/ttfr-tti-report",
    "/api/v1/tvlk/events",
    "/api/v1/metrics",
    "/api/sen/i",
    "/api/private/hotel-search-seo-view",
    "/api/private/hotel/detail/seoview",
    "/api/private/hotel/breadcrumb",
)

ESSENTIAL_REVIEW_API_PATTERNS = (
    "/api/v2/ugc/review/consumption/v2/getreviews",
    "/api/v2/ugc/review/consumption/v2/getreviewfilter",
    "/api/v2/ugc/review/consumption/v2/getavailablereviewwithaggregate",
    "/api/v2/ugc/review/consumption/v2/getreviewaggregatesummary",
    "/api/v2/ugc/review/consumption/v2/getreviewcount",
    "/api/v2/ugc/review/consumption/v2/getreviewratingtagaggregate",
)

RAW_DIR = Path(__file__).resolve().parent.parent / config.get("data", {}).get("raw_dir", "data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Thư mục lưu Chrome profile (cookie, cache)
CHROME_PROFILE_DIR = Path(__file__).resolve().parent.parent / "chrome_profile"
CHROME_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

def load_city_configs():
    """Load cities từ file riêng, fallback về config.yaml nếu file không có."""
    project_root = Path(__file__).resolve().parent.parent
    cities_file = project_root / CITIES_FILE
    cities = []

    if cities_file.exists():
        try:
            payload = yaml.safe_load(cities_file.read_text(encoding="utf-8")) or {}
            values = payload.get("cities", [])
            if isinstance(values, list):
                cities = values
        except Exception as e:
            logger.warning(f"Failed reading cities file {cities_file}: {e}")

    if not cities:
        values = cfg.get("cities", [])
        if isinstance(values, list):
            cities = values

    normalized = []
    for row in cities:
        if not isinstance(row, dict):
            continue
        if row.get("enabled") is False:
            continue
        geo_id = str(row.get("geo_id", "")).strip()
        name = str(row.get("name", "")).strip()
        code = str(row.get("code", "")).strip() or (name[:2].upper() if name else "")
        if not geo_id or not name:
            logger.warning(f"Skip city config invalid (missing geo_id/name): {row}")
            continue

        city_max_hotels = row.get("max_hotels_per_city", MAX_HOTELS)
        city_max_reviews = row.get("max_reviews_per_hotel", MAX_REVIEWS)
        try:
            city_max_hotels = max(1, int(city_max_hotels))
        except Exception:
            city_max_hotels = MAX_HOTELS
        try:
            city_max_reviews = max(1, int(city_max_reviews))
        except Exception:
            city_max_reviews = MAX_REVIEWS

        normalized.append({
            "geo_id": geo_id,
            "name": name,
            "code": code,
            "region": str(row.get("region", "")).strip().lower(),
            "max_hotels_per_city": city_max_hotels,
            "max_reviews_per_hotel": city_max_reviews,
        })

    return normalized


def build_proxy_pool():
    """Parse proxy config list to Playwright-compatible proxy options."""
    enabled = bool(PROXY_CFG.get("enabled", False))
    if not enabled:
        return [None]

    values = PROXY_CFG.get("list", [])
    if not isinstance(values, list):
        return [None]

    pool = []
    for item in values:
        if not isinstance(item, dict):
            continue
        server = str(item.get("server", "")).strip()
        if not server:
            continue
        proxy = {"server": server}
        username = str(item.get("username", "")).strip()
        password = str(item.get("password", "")).strip()
        if username:
            proxy["username"] = username
        if password:
            proxy["password"] = password
        pool.append(proxy)

    if not pool:
        logger.warning("[Proxy] proxy.enabled=true nhưng danh sách proxy rỗng/không hợp lệ, fallback direct IP.")
        return [None]

    if bool(PROXY_CFG.get("rotate_per_run", True)):
        random.shuffle(pool)
    return pool

CITIES = load_city_configs()

# ============================================================
# UTILS
# ============================================================

# Ký tự dấu đặc trưng tiếng Việt (không có trong tiếng Anh/Hàn/Trung/Nhật)
VIETNAMESE_CHARS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
                       "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ")

def is_vietnamese(text):
    """Kiểm tra text có phải tiếng Việt không (có ký tự dấu tiếng Việt)."""
    if not text:
        return False
    vn_count = sum(1 for c in text if c in VIETNAMESE_CHARS)
    # Cần ít nhất 1 ký tự dấu tiếng Việt cho text dài
    # hoặc text ngắn toàn ASCII nhưng chứa từ Việt
    if vn_count >= 1:
        return True
    # Text ngắn không dấu nhưng chứa từ Việt phổ biến
    vn_words = ["phong", "khach san", "sach", "tot", "dep", "nhan vien", "ok", "tam"]
    if len(text) < 50 and any(w in text.lower() for w in vn_words):
        return True
    return False

async def random_delay(min_s=3, max_s=10):
    if FAST_CRAWL:
        await asyncio.sleep(random.uniform(FAST_MIN_DELAY_SECONDS, FAST_MAX_DELAY_SECONDS))
        return
    # Stable mode: enforce human-like delay range to reduce bot detection.
    min_delay = max(0.0, float(min_s))
    max_delay = max(min_delay, float(max_s))
    if max_delay < min_delay:
        max_delay = min_delay
    await asyncio.sleep(random.uniform(min_delay, max_delay))


async def random_page_delay(min_s=5, max_s=10):
    """Delay for page-level actions (navigation, tab switch)."""
    if FAST_CRAWL:
        await asyncio.sleep(random.uniform(FAST_MIN_DELAY_SECONDS, FAST_MAX_DELAY_SECONDS))
        return
    low = max(0.0, float(min_s))
    high = max(low, float(max_s))
    if high < low:
        high = low
    await asyncio.sleep(random.uniform(low, high))

async def human_scroll(page, times=3):
    for _ in range(times):
        await page.mouse.wheel(0, random.randint(300, 700))
        if FAST_CRAWL:
            await asyncio.sleep(random.uniform(0.05, 0.15))
        else:
            await asyncio.sleep(random.uniform(0.5, 1.2))

def build_search_url(geo_id, city_name):
    checkin = datetime.now() + timedelta(days=1)
    checkout = checkin + timedelta(days=1)
    encoded_name = urllib.parse.quote(city_name)
    spec = f"{checkin.strftime('%d-%m-%Y')}.{checkout.strftime('%d-%m-%Y')}.1.1.HOTEL_GEO.{geo_id}.{encoded_name}.1"
    return f"https://www.traveloka.com/vi-vn/hotel/search?spec={spec}"

def make_absolute_traveloka_url(raw_url):
    if not raw_url:
        return ""
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url
    if raw_url.startswith("/"):
        return f"https://www.traveloka.com{raw_url}"
    return ""


def normalize_name_key(text):
    """Normalize hotel name for fuzzy compare in fallback click flow."""
    value = str(text or "").strip().lower()
    value = value.replace("’", "'").replace("`", "'")
    value = value.replace("&", " and ")
    value = re.sub(r"[^0-9a-zA-ZÀ-ỹđĐ' ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

async def collect_hotel_targets(search_page, hotel_names, max_hotels_for_city):
    """Từ danh sách hotel_name, cố gắng lấy URL detail để crawl song song."""
    targets = []
    seen_urls = set()
    seen_source_ids = set()
    normalized_names = {name.lower(): name for name in hotel_names}

    # Ưu tiên cách 1: lần theo ancestor anchor từ h3.
    for hotel_name in hotel_names:
        try:
            h3 = search_page.locator("h3", has_text=hotel_name).first
            anchor = h3.locator("xpath=ancestor::a[1]")
            if await anchor.count() == 0:
                continue
            href = await anchor.get_attribute("href")
            detail_url = make_absolute_traveloka_url(href or "")
            source_id = extract_source_hotel_id(detail_url)
            if not detail_url or "/hotel/detail" not in detail_url:
                continue
            if detail_url in seen_urls or (source_id and source_id in seen_source_ids):
                continue
            seen_urls.add(detail_url)
            if source_id:
                seen_source_ids.add(source_id)
            targets.append({"hotel_name": hotel_name, "detail_url": detail_url, "source_hotel_id": source_id})
        except Exception:
            continue

    # Cách 2: quét trực tiếp tất cả anchor chứa /hotel/detail (DOM mới của Traveloka).
    if len(targets) < max(3, min(max_hotels_for_city, 5)):
        anchors = search_page.locator("a[href*='/hotel/detail']")
        anchor_count = await anchors.count()
        for idx in range(anchor_count):
            try:
                anchor = anchors.nth(idx)
                href = await anchor.get_attribute("href")
                detail_url = make_absolute_traveloka_url(href or "")
                source_id = extract_source_hotel_id(detail_url)
                if not detail_url or "/hotel/detail" not in detail_url:
                    continue
                if detail_url in seen_urls or (source_id and source_id in seen_source_ids):
                    continue

                raw_text = (await anchor.inner_text() or "").strip()
                first_line = raw_text.splitlines()[0].strip() if raw_text else ""
                hotel_name = normalized_names.get(first_line.lower(), first_line)
                if not hotel_name:
                    url_name_match = re.search(r"HOTEL\.\d+\.(.+?)\.(?:\d+)(?:[&?]|$)", detail_url)
                    url_name = urllib.parse.unquote(url_name_match.group(1)) if url_name_match else "Unknown Hotel"
                    hotel_name = url_name.replace(".", " ")

                seen_urls.add(detail_url)
                if source_id:
                    seen_source_ids.add(source_id)
                targets.append({"hotel_name": hotel_name, "detail_url": detail_url, "source_hotel_id": source_id})
            except Exception:
                continue

    return targets

async def optimize_context_for_mac(context):
    """Giảm tải CPU/GPU/network và chặn fetch tracking không cần thiết."""
    if not BLOCK_RESOURCE_TYPES:
        return

    async def _route_handler(route):
        request = route.request
        resource_type = request.resource_type
        url = request.url.lower()
        if resource_type in BLOCK_RESOURCE_TYPES:
            await route.abort()
            return
        if any(pattern in url for pattern in THIRD_PARTY_TRACKING_PATTERNS):
            await route.abort()
            return
        # Chặn các endpoint telemetry cùng domain không phục vụ crawl review.
        if "traveloka.com/api/" in url and any(pattern in url for pattern in NON_ESSENTIAL_SAME_ORIGIN_API_PATTERNS):
            await route.abort()
            return
        await route.continue_()

    await context.route("**/*", _route_handler)

def extract_source_hotel_id(detail_url):
    """Lấy hotel_id gốc từ URL Traveloka."""
    if not detail_url:
        return ""
    match = re.search(r"HOTEL\.(\d{5,})", detail_url)
    if match:
        return match.group(1)
    match = re.search(r"-(\d{5,})(?:\?|$)", detail_url)
    return match.group(1) if match else ""

def normalize_score_to_5(score_raw):
    """Chuẩn hóa điểm review về thang 5."""
    if score_raw in (None, ""):
        return ""
    try:
        score = float(str(score_raw).replace(",", "."))
        if score > 5:
            score = score / 2.0
        return str(round(score, 1))
    except (ValueError, TypeError):
        return ""

def to_iso_timestamp(timestamp_raw):
    """Chuyển timestamp API về ISO8601 UTC."""
    if timestamp_raw in (None, ""):
        return ""
    try:
        ts = int(str(timestamp_raw))
        if ts > 10**12:  # milliseconds
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return str(timestamp_raw)

def normalize_text_spaces(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def is_valid_location_candidate(text, city_name):
    """Lọc candidate địa chỉ, tránh dính nội dung review."""
    value = normalize_text_spaces(text)
    if not value:
        return False
    if len(value) < 10 or len(value) > 160:
        return False

    lowered = value.lower()
    city_lower = city_name.lower()
    if city_lower and city_lower not in lowered:
        return False

    review_like_tokens = [
        "dịch vụ",
        "nhân viên",
        "sạch sẽ",
        "lần sau",
        "quay lại",
        "view đẹp",
        "thân thiện",
        "hỗ trợ",
        "❤️",
        "🥲",
    ]
    if any(token in lowered for token in review_like_tokens):
        return False

    # Địa chỉ thường có dấu phẩy/ số nhà/ đơn vị hành chính.
    has_address_signal = (
        "," in value
        or any(ch.isdigit() for ch in value)
        or "phường" in lowered
        or "quận" in lowered
        or "huyện" in lowered
        or "xã" in lowered
        or "thành phố" in lowered
        or "việt nam" in lowered
    )
    if not has_address_signal:
        return False
    return True

def _extract_address_from_jsonld_node(node):
    if isinstance(node, dict):
        address = node.get("address")
        if isinstance(address, dict):
            parts = [
                address.get("streetAddress", ""),
                address.get("addressLocality", ""),
                address.get("addressRegion", ""),
                address.get("postalCode", ""),
                address.get("addressCountry", ""),
            ]
            combined = ", ".join([normalize_text_spaces(p) for p in parts if normalize_text_spaces(p)])
            if combined:
                return combined
        for value in node.values():
            result = _extract_address_from_jsonld_node(value)
            if result:
                return result
    elif isinstance(node, list):
        for item in node:
            result = _extract_address_from_jsonld_node(item)
            if result:
                return result
    return ""

def extract_location_from_soup(soup, city_name, fallback_city):
    """Ưu tiên dữ liệu cấu trúc và selector địa chỉ rõ ràng."""
    # 1) JSON-LD thường chứa address chuẩn.
    for script in soup.select("script[type='application/ld+json']"):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        candidate = _extract_address_from_jsonld_node(payload)
        if is_valid_location_candidate(candidate, city_name):
            return normalize_text_spaces(candidate)

    # 2) Selector chứa keyword address/location.
    css_candidates = [
        "[data-testid*='address']",
        "[data-testid*='location']",
        "[class*='address']",
        "[class*='location']",
        "[itemprop='address']",
    ]
    for selector in css_candidates:
        for elem in soup.select(selector):
            candidate = normalize_text_spaces(elem.get_text(" ", strip=True))
            if is_valid_location_candidate(candidate, city_name):
                return candidate

    # 3) Fallback cuối: quét span/div nhưng lọc chặt để không dính review text.
    for elem in soup.find_all(["span", "div"]):
        candidate = normalize_text_spaces(elem.get_text(" ", strip=True))
        if is_valid_location_candidate(candidate, city_name):
            return candidate

    return fallback_city


def normalize_location_to_city(location_raw, city_name):
    """Chuẩn hóa location về tên tỉnh/thành để dễ query."""
    city = normalize_text_spaces(city_name or "")
    if city:
        # Với crawler theo city config, ưu tiên nhãn city đã chọn để tránh lưu địa chỉ chi tiết.
        return city
    return normalize_text_spaces(location_raw or "")

def build_review_id(source_hotel_id, source_review_id, review_text, review_rating, review_timestamp):
    """Sinh review_id chuẩn: traveloka_<source_hotel_id>_<source_review_id>."""
    if source_review_id:
        if source_hotel_id:
            return f"traveloka_{source_hotel_id}_{source_review_id}"
        return f"traveloka_{source_review_id}"
    seed = f"{source_hotel_id}|{review_text.strip().lower()}|{review_rating}|{review_timestamp}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"traveloka_{source_hotel_id}_{digest}" if source_hotel_id else f"traveloka_{digest}"


def should_skip_source_review_id(source_review_id):
    """Loại các review ID bên thứ ba (ví dụ EXPEDIA-...) để tránh rối ID chuẩn."""
    value = str(source_review_id or "").strip()
    if not value:
        return False
    upper = value.upper()
    return upper.startswith("EXPEDIA-")

def extract_next_cursor(payload):
    """Tìm cursor phân trang trong response API."""
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    candidates = [
        data.get("nextCursor"),
        data.get("next_cursor"),
        data.get("cursor"),
        data.get("next"),
    ]
    for value in candidates:
        if value:
            return value
    return ""

def set_cursor_in_payload(payload, next_cursor):
    """Cập nhật cursor cho payload request tiếp theo."""
    updated = False

    def _walk(node):
        nonlocal updated
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = key.lower()
                if "cursor" in key_lower and (value is None or isinstance(value, str)):
                    node[key] = next_cursor
                    updated = True
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return updated

def build_get_reviews_payload(source_hotel_id, skip, limit):
    return {
        "fields": [],
        "data": {
            "objectId": source_hotel_id,
            "productType": "HOTEL",
            "configId": "REV_CONSV2_HOTEL_GENERAL_V1",
            "filter": {
                "format": "FORMAT_VALUE_TEXT",
                "rating": "RATING_VALUE_ALL",
                "language": "LANGUAGE_VALUE_ALL",
                "travelPurpose": "TRAVEL_PURPOSE_VALUE_ALL",
            },
            "sort": "SORT_CREATED_DESCENDING",
            "origin": "TRAVELOKA",
            "ratingTagSet": [],
            "limit": str(limit),
            "skip": str(skip),
        },
        "clientInterface": "desktop",
    }

async def build_dynamic_ugc_headers(page, detail_url):
    """Tạo header UGC động từ phiên browser hiện tại (không cần copy fetch thủ công)."""
    route_prefix = "vi-vn"
    if "/vi-vn/" in detail_url:
        route_prefix = "vi-vn"

    local_info = await page.evaluate("""() => ({
        did: localStorage.getItem('tvlk-did') || '',
        locale: (window.__NEXT_DATA__ && window.__NEXT_DATA__.locale) || 'vi-vn'
    })""")
    did = (local_info or {}).get("did", "")
    locale = (local_info or {}).get("locale", "vi-vn")

    cookies = await page.context.cookies("https://www.traveloka.com")
    cookie_map = {c.get("name", ""): c.get("value", "") for c in cookies}
    mcc_id = cookie_map.get("tv_mcc_id", "")
    client_session = cookie_map.get("clientSessionId", "")

    headers = {
        "accept": "*/*",
        "accept-language": "vi,en;q=0.9",
        "content-type": "application/json",
        "x-client-interface": "desktop",
        "x-domain": "ugcReview",
        "x-route-prefix": route_prefix,
        "tv-country": "VN",
        "tv-currency": "VND",
        "tv-language": "vi_VN",
        "referer": detail_url,
    }
    if mcc_id:
        headers["tv-mcc-id"] = mcc_id
    if client_session:
        headers["tv-clientsessionid"] = client_session
    if did:
        headers["x-did"] = did
    if locale and isinstance(locale, str):
        headers["x-route-prefix"] = locale.lower().replace("_", "-")
    return headers

# ============================================================
# STEP 1: Click từng hotel card trên search results
# ============================================================

async def crawl_hotels_in_city(context, city, claimed_hotel_ids=None, claimed_hotel_lock=None):
    """Load search page, click từng hotel card, scrape reviews từ tab mới."""
    geo_id = city["geo_id"]
    city_name = city["name"]
    city_code = city.get("code", city_name[:2].upper())
    city_max_hotels = max(1, int(city.get("max_hotels_per_city", MAX_HOTELS)))
    city_max_reviews = max(1, int(city.get("max_reviews_per_hotel", MAX_REVIEWS)))
    logger.info(
        f"[CityConfig] {city_name}: max_hotels={city_max_hotels}, max_reviews_per_hotel={city_max_reviews}"
    )
    url = build_search_url(geo_id, city_name)

    search_page = await context.new_page()
    all_reviews = []
    hotels_scraped = 0

    try:
        if HAS_STEALTH:
            await stealth_async(search_page)
        logger.info(f"[Search] Loading: {url}")
        await search_page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
        await random_page_delay(5, 10)
        await human_scroll(search_page, times=4)

        for retry in range(3):
            h3_count_check = await search_page.locator("h3").count()
            if h3_count_check > 2:
                break
            logger.info(f"[Search] Retry {retry+1}: Only {h3_count_check} h3 found. Waiting + scrolling more...")
            try:
                body_text = await search_page.locator("body").inner_text()
                if "Tìm kiếm lại" in body_text:
                    retry_btn = search_page.get_by_text("Tìm kiếm lại", exact=True)
                    if await retry_btn.count() > 0:
                        await retry_btn.first.click(timeout=5000)
                        logger.info("[Search] Clicked 'Tìm kiếm lại' to refresh results")
                        await random_page_delay(5, 10)
            except Exception as e:
                logger.debug(f"[Search] Could not click retry button: {e}")
            await random_delay(5, 8)
            await human_scroll(search_page, times=5)

        async def extract_hotel_names_from_page() -> list[str]:
            names = []
            seen = set()
            h3_elements = search_page.locator("h3")
            h3_count = await h3_elements.count()
            for i in range(h3_count):
                try:
                    text = await h3_elements.nth(i).text_content()
                    if not text or len(text.strip()) <= 3:
                        continue
                    cleaned = text.strip()
                    skip_texts = [
                        city_name, "khách sạn", "xem bộ sưu tập", "tìm kiếm",
                        "xếp theo", "lọc", "hiển thị", "top ",
                    ]
                    if any(skip.lower() in cleaned.lower() for skip in skip_texts):
                        continue
                    key = cleaned.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    names.append(cleaned)
                except Exception:
                    continue
            return names

        async def extract_targets_from_page() -> list[dict]:
            """Collect visible /hotel/detail anchors from current lazy-loading window."""
            results = []
            anchors = search_page.locator("a[href*='/hotel/detail']")
            anchor_count = await anchors.count()
            for idx in range(anchor_count):
                try:
                    anchor = anchors.nth(idx)
                    href = await anchor.get_attribute("href")
                    detail_url = make_absolute_traveloka_url(href or "")
                    source_id = extract_source_hotel_id(detail_url)
                    if not detail_url or "/hotel/detail" not in detail_url:
                        continue
                    raw_text = (await anchor.inner_text() or "").strip()
                    first_line = raw_text.splitlines()[0].strip() if raw_text else ""
                    hotel_name = first_line
                    if not hotel_name:
                        url_name_match = re.search(r"HOTEL\.\d+\.(.+?)\.(?:\d+)(?:[&?]|$)", detail_url)
                        url_name = urllib.parse.unquote(url_name_match.group(1)) if url_name_match else "Unknown Hotel"
                        hotel_name = url_name.replace(".", " ")
                    results.append(
                        {"hotel_name": hotel_name, "detail_url": detail_url, "source_hotel_id": source_id}
                    )
                except Exception:
                    continue
            return results

        hotel_names_acc = {}
        hotel_targets_acc = {}
        stale_rounds = 0
        # Lazy-loading kiểu virtualization cần tích lũy qua nhiều vòng, không dùng snapshot cuối.
        max_scroll_rounds = max(8, min(city_max_hotels // 4 + 10, 32))
        for round_idx in range(max_scroll_rounds):
            current_names = await extract_hotel_names_from_page()
            current_targets = await extract_targets_from_page()

            for name in current_names:
                key = str(name).strip().lower()
                if key and key not in hotel_names_acc:
                    hotel_names_acc[key] = name

            for target in current_targets:
                source_hotel_id = str(target.get("source_hotel_id", "")).strip()
                detail_url = str(target.get("detail_url", "")).strip()
                dedup_key = source_hotel_id or detail_url
                if not dedup_key:
                    continue
                if dedup_key not in hotel_targets_acc:
                    hotel_targets_acc[dedup_key] = target

            current_total_names = len(hotel_names_acc)
            current_total_targets = len(hotel_targets_acc)
            logger.info(
                f"[Search] Round {round_idx + 1}/{max_scroll_rounds}: "
                f"visible_names={len(current_names)}, visible_targets={len(current_targets)}, "
                f"acc_names={current_total_names}, acc_targets={current_total_targets}"
            )

            if round_idx == 0:
                prev_score = -1
            score = current_total_names + current_total_targets
            if score > prev_score:
                stale_rounds = 0
            else:
                stale_rounds += 1
            prev_score = score

            if current_total_targets >= city_max_hotels or stale_rounds >= 4:
                break
            await human_scroll(search_page, times=3)
            await random_delay(0.4, 0.9)

        target_hotels = list(hotel_names_acc.values())
        hotel_targets = list(hotel_targets_acc.values())

        if len(hotel_targets) < max(3, min(city_max_hotels, 8)):
            # Fallback cũ: cố gom thêm từ h3->anchor nếu lazy list trả về quá ít.
            fallback_targets = await collect_hotel_targets(search_page, target_hotels, city_max_hotels)
            for target in fallback_targets:
                source_hotel_id = str(target.get("source_hotel_id", "")).strip()
                detail_url = str(target.get("detail_url", "")).strip()
                dedup_key = source_hotel_id or detail_url
                if dedup_key and dedup_key not in hotel_targets_acc:
                    hotel_targets_acc[dedup_key] = target
            hotel_targets = list(hotel_targets_acc.values())

        logger.info(f"[Search] Final accumulated hotel names: {len(target_hotels)}")
        logger.info(f"[Search] Final accumulated detail targets: {len(hotel_targets)}")
        logger.info(f"[Search] Sample hotel names: {target_hotels[:10]}")
        fallback_click_flow = False
        if not hotel_targets:
            if target_hotels:
                logger.warning(
                    "[Search] Found hotel names but no detail URLs yet. Retrying short lazy-load scan..."
                )
                for _ in range(3):
                    await human_scroll(search_page, times=2)
                    await search_page.wait_for_timeout(350)
                    retry_targets = await extract_targets_from_page()
                    for target in retry_targets:
                        source_hotel_id = str(target.get("source_hotel_id", "")).strip()
                        detail_url = str(target.get("detail_url", "")).strip()
                        dedup_key = source_hotel_id or detail_url
                        if dedup_key and dedup_key not in hotel_targets_acc:
                            hotel_targets_acc[dedup_key] = target
                    if hotel_targets_acc:
                        break
                hotel_targets = list(hotel_targets_acc.values())
                if not hotel_targets:
                    logger.warning(
                        "[Search] Still no detail URLs. Switching to click fallback to avoid skipping city."
                    )
                    fallback_click_flow = True
            else:
                logger.warning("[Search] No hotel targets found. Skip city.")
                if SKIP_IDLE_SECONDS > 0:
                    await asyncio.sleep(SKIP_IDLE_SECONDS)
                return all_reviews

        async def claim_source_hotel_id(source_hotel_id):
            """Claim hotel ID globally to avoid duplicated crawl across cities/workers."""
            if not source_hotel_id or claimed_hotel_ids is None:
                return True
            if claimed_hotel_lock:
                async with claimed_hotel_lock:
                    if source_hotel_id in claimed_hotel_ids:
                        return False
                    claimed_hotel_ids.add(source_hotel_id)
                    return True
            if source_hotel_id in claimed_hotel_ids:
                return False
            claimed_hotel_ids.add(source_hotel_id)
            return True

        async def scrape_hotel_target(idx, target, sem):
            hotel_name = target["hotel_name"]
            detail_url = target["detail_url"]
            async with sem:
                detail_page = None
                try:
                    detail_page = await context.new_page()
                    if HAS_STEALTH:
                        await stealth_async(detail_page)
                    logger.info(f"\n--- [Parallel Attempt {idx}] Opening: {hotel_name} ---")
                    await detail_page.goto(detail_url, timeout=TIMEOUT, wait_until="domcontentloaded")
                    await random_page_delay(5, 10)
                    reviews = await asyncio.wait_for(
                        scrape_reviews_from_detail(
                            detail_page, hotel_name, city_name, city_code, detail_url, city_max_reviews
                        ),
                        timeout=DETAIL_TASK_TIMEOUT_SECONDS,
                    )
                    return {"hotel_name": hotel_name, "reviews": reviews, "error": ""}
                except asyncio.TimeoutError:
                    return {
                        "hotel_name": hotel_name,
                        "reviews": [],
                        "error": f"detail_timeout_{DETAIL_TASK_TIMEOUT_SECONDS}s",
                    }
                except Exception as e:
                    return {"hotel_name": hotel_name, "reviews": [], "error": str(e)}
                finally:
                    if detail_page:
                        try:
                            await detail_page.close()
                        except Exception:
                            pass

        if hotel_targets:
            # Quét dư vừa phải để bù hotel lỗi, tránh mở quá nhiều target gây nặng tab.
            max_parallel_targets = min(len(hotel_targets), city_max_hotels + max(4, city_max_hotels // 4))
            sem = asyncio.Semaphore(MAX_CONCURRENT_HOTELS)
            selected_targets = []
            for target in hotel_targets[:max_parallel_targets]:
                source_hotel_id = str(target.get("source_hotel_id", "")).strip()
                can_crawl = await claim_source_hotel_id(source_hotel_id)
                if not can_crawl:
                    logger.info(
                        f"[Skip] Duplicate source_hotel_id already claimed: {source_hotel_id} ({target.get('hotel_name', '')})"
                    )
                    continue
                selected_targets.append(target)
            if not selected_targets:
                if target_hotels:
                    logger.warning(
                        "[Search] All URL targets already claimed. Switching to click fallback by hotel name."
                    )
                    fallback_click_flow = True
                else:
                    logger.warning("[Search] All targets were crawled before. Skip city.")
                    if SKIP_IDLE_SECONDS > 0:
                        await asyncio.sleep(SKIP_IDLE_SECONDS)
                    return all_reviews
            tasks = [
                asyncio.create_task(scrape_hotel_target(idx + 1, selected_targets[idx], sem))
                for idx in range(len(selected_targets))
            ]
            checkpoint_file = RAW_DIR / "traveloka_checkpoint.jsonl"
            for done in asyncio.as_completed(tasks):
                result = await done
                if result["error"]:
                    logger.error(f"[Error] Failed for {result['hotel_name']}: {result['error']}")
                    continue
                reviews = result["reviews"]
                if not reviews:
                    logger.info(f"[Skip] No Vietnamese reviews for {result['hotel_name']}, skipping...")
                    continue
                if hotels_scraped >= city_max_hotels:
                    continue
                all_reviews.extend(reviews)
                hotels_scraped += 1
                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    for r in reviews:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                logger.info(f"[Progress] Hotels: {hotels_scraped}, Total reviews: {len(all_reviews)}")
                if COOLDOWN_EVERY_HOTELS and COOLDOWN_SECONDS and hotels_scraped % COOLDOWN_EVERY_HOTELS == 0:
                    logger.info(f"[Cooldown] Sleep {COOLDOWN_SECONDS}s to reduce temperature")
                    await asyncio.sleep(COOLDOWN_SECONDS)
                if hotels_scraped >= city_max_hotels:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
        if fallback_click_flow and target_hotels and hotels_scraped < city_max_hotels:
            logger.info("[Fallback] Start click-by-name flow for remaining hotels.")
            checkpoint_file = RAW_DIR / "traveloka_checkpoint.jsonl"
            attempted = 0
            for hotel_name in target_hotels:
                if hotels_scraped >= city_max_hotels:
                    break
                attempted += 1
                detail_page = None
                try:
                    logger.info(f"\n--- [Fallback Attempt {attempted}] Clicking: {hotel_name} ---")
                    matched_h3 = None
                    wanted = normalize_name_key(hotel_name)
                    for _ in range(3):
                        h3_nodes = search_page.locator("h3")
                        h3_count = await h3_nodes.count()
                        best_idx = -1
                        best_score = -1
                        for idx in range(h3_count):
                            try:
                                txt = (await h3_nodes.nth(idx).text_content() or "").strip()
                                if not txt:
                                    continue
                                got = normalize_name_key(txt)
                                score = 0
                                if got == wanted:
                                    score = 3
                                elif wanted and got and (wanted in got or got in wanted):
                                    score = 2
                                elif wanted and got:
                                    overlap = set(wanted.split()) & set(got.split())
                                    if overlap:
                                        score = 1
                                if score > best_score:
                                    best_score = score
                                    best_idx = idx
                            except Exception:
                                continue
                        if best_idx >= 0 and best_score >= 1:
                            matched_h3 = h3_nodes.nth(best_idx)
                            break
                        await human_scroll(search_page, times=1)
                        await search_page.wait_for_timeout(250)

                    if not matched_h3:
                        logger.warning(f"[Fallback-Skip] Cannot find clickable h3 for: {hotel_name}")
                        continue

                    async with context.expect_page(timeout=8000) as new_page_info:
                        await matched_h3.click(timeout=4000)
                    detail_page = await new_page_info.value
                    await detail_page.wait_for_load_state("domcontentloaded")
                    await random_delay(0.05, 0.2)
                    detail_url = detail_page.url
                    source_hotel_id = extract_source_hotel_id(detail_url)
                    if not source_hotel_id:
                        logger.info(f"[Fallback-Skip] Missing source_hotel_id for {hotel_name}")
                        continue
                    can_crawl = await claim_source_hotel_id(source_hotel_id)
                    if not can_crawl:
                        logger.info(
                            f"[Fallback-Skip] Duplicate source_hotel_id already claimed: {source_hotel_id}"
                        )
                        continue
                    reviews = await asyncio.wait_for(
                        scrape_reviews_from_detail(
                            detail_page, hotel_name, city_name, city_code, detail_url, city_max_reviews
                        ),
                        timeout=DETAIL_TASK_TIMEOUT_SECONDS,
                    )
                    if not reviews:
                        continue
                    all_reviews.extend(reviews)
                    hotels_scraped += 1
                    with open(checkpoint_file, "a", encoding="utf-8") as f:
                        for r in reviews:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    logger.info(f"[Fallback-Progress] Hotels: {hotels_scraped}, Total reviews: {len(all_reviews)}")
                except Exception as e:
                    logger.error(f"[Fallback-Error] Failed for {hotel_name}: {e}")
                finally:
                    if detail_page:
                        try:
                            await detail_page.close()
                        except Exception:
                            pass
        return all_reviews
    finally:
        try:
            await search_page.close()
        except Exception:
            pass

# ============================================================
# STEP 2: Scrape reviews từ hotel detail page
# ============================================================

def parse_reviews_from_api_payload(
    payload,
    source_hotel_id,
    hotel_name,
    location,
    overall_rating,
    seen_review_ids,
):
    """Parse response JSON từ getReviews API thành schema crawler."""
    parsed = []
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    review_items = data.get("reviews", [])
    if not isinstance(review_items, list):
        review_items = []

    for item in review_items:
        if not isinstance(item, dict):
            continue
        # Rule B: chỉ giữ bản gốc (không lấy bản dịch máy).
        translation_status = str(item.get("translationStatus", "")).upper()
        if translation_status != "ORIGINAL":
            continue
        # Theo yêu cầu dữ liệu: chỉ lấy review gốc của khách.
        review_text = str(item.get("reviewOriginalText") or "").strip()
        if len(review_text) <= 10:
            continue
        if not is_vietnamese(review_text):
            continue
        review_reply = item.get("reviewReply", {})
        if isinstance(review_reply, dict):
            reply_text = str(
                review_reply.get("replyOriginalText")
                or review_reply.get("replyContentText")
                or review_reply.get("replyText")
                or ""
            ).strip()
            if reply_text and review_text == reply_text:
                continue

        source_review_id = str(item.get("reviewId", "")).strip()
        if should_skip_source_review_id(source_review_id):
            continue
        review_score_raw = item.get("reviewScore", "")
        review_timestamp_raw = item.get("reviewTimestamp", "")
        review_rating = normalize_score_to_5(review_score_raw)
        review_date = to_iso_timestamp(review_timestamp_raw)
        review_id = build_review_id(
            source_hotel_id,
            source_review_id,
            review_text,
            review_rating,
            review_timestamp_raw,
        )
        if review_id in seen_review_ids:
            continue
        seen_review_ids.add(review_id)

        parsed.append({
            "review_id": review_id,
            "source_review_id": source_review_id,
            "source_hotel_id": source_hotel_id,
            "hotel_name": hotel_name,
            "location": location,
            "rating": overall_rating,
            "review_text": review_text,
            "review_date": review_date,
            "review_rating": review_rating,
            "source": "traveloka",
        })

    has_next = bool(data.get("hasNext"))
    next_cursor = extract_next_cursor(payload)
    return parsed, has_next, next_cursor

async def fetch_reviews_via_api(
    context,
    source_hotel_id,
    hotel_name,
    location,
    overall_rating,
    seed_response=None,
    seed_headers=None,
    detail_url="",
    max_reviews=100,
):
    """API-first: parse response đã capture + replay request thật để phân trang."""
    if not seed_response and not seed_headers:
        logger.warning("[Detail] No captured getReviews response or UGC headers")
        return []

    replay_headers = {}
    allowed_headers = {
        "accept",
        "content-type",
        "accept-language",
        "sec-ch-ua",
        "sec-ch-ua-mobile",
        "sec-ch-ua-platform",
        "sec-fetch-dest",
        "sec-fetch-mode",
        "sec-fetch-site",
        "priority",
        "referer",
        "origin",
        "t-a-v",
        "tv-clientsessionid",
        "tv-country",
        "tv-currency",
        "tv-language",
        "tv-mcc-id",
        "www-app-version",
        "x-client-interface",
        "x-did",
        "x-domain",
        "x-route-prefix",
    }
    api_url = "https://www.traveloka.com/api/v2/ugc/review/consumption/v2/getReviews"
    base_request_payload = build_get_reviews_payload(source_hotel_id, 0, min(20, max_reviews))
    skip = 0
    limit = min(20, max_reviews)

    if seed_response:
        try:
            first_payload = await seed_response.json()
        except Exception as e:
            logger.warning(f"[Detail] Failed parsing captured getReviews response: {e}")
            first_payload = None

        seed_request = seed_response.request
        api_url = seed_response.url
        post_data = seed_request.post_data or "{}"
        try:
            base_request_payload = json.loads(post_data)
        except json.JSONDecodeError:
            logger.warning("[Detail] Captured request payload is not valid JSON, using default payload")
            base_request_payload = build_get_reviews_payload(source_hotel_id, 0, min(20, max_reviews))
        # Chốt objectId theo đúng hotel hiện tại, tránh dính payload từ tab khác.
        if not isinstance(base_request_payload, dict):
            base_request_payload = build_get_reviews_payload(source_hotel_id, 0, min(20, max_reviews))
        if not isinstance(base_request_payload.get("data"), dict):
            base_request_payload["data"] = {}
        base_request_payload["data"]["objectId"] = source_hotel_id

        data_obj = base_request_payload.get("data", {}) if isinstance(base_request_payload, dict) else {}
        try:
            skip = int(str(data_obj.get("skip", "0")))
        except ValueError:
            skip = 0
        try:
            limit = int(str(data_obj.get("limit", "20")))
        except ValueError:
            limit = 20
        limit = max(1, min(limit, max_reviews))

        req_headers = seed_request.headers or {}
        for key, value in req_headers.items():
            if key.lower() in allowed_headers:
                replay_headers[key] = value
    else:
        first_payload = None
        req_headers = seed_headers or {}
        for key, value in req_headers.items():
            if key.lower() in allowed_headers:
                replay_headers[key] = value

    if "x-route-prefix" not in {k.lower() for k in replay_headers.keys()}:
        replay_headers["x-route-prefix"] = "vi-vn"
    if "x-domain" not in {k.lower() for k in replay_headers.keys()}:
        replay_headers["x-domain"] = "ugcReview"
    if "referer" not in {k.lower() for k in replay_headers.keys()} and detail_url:
        replay_headers["referer"] = detail_url
    if "content-type" not in {k.lower() for k in replay_headers.keys()}:
        replay_headers["content-type"] = "application/json"
    if "accept" not in {k.lower() for k in replay_headers.keys()}:
        replay_headers["accept"] = "*/*"
    if "x-client-interface" not in {k.lower() for k in replay_headers.keys()}:
        replay_headers["x-client-interface"] = "desktop"
    if "x-domain" not in {k.lower() for k in replay_headers.keys()}:
        replay_headers["x-domain"] = "ugcReview"

    reviews = []
    seen_review_ids = set()
    next_cursor = ""
    if first_payload:
        parsed, has_next, next_cursor = parse_reviews_from_api_payload(
            first_payload,
            source_hotel_id,
            hotel_name,
            location,
            overall_rating,
            seen_review_ids,
        )
        reviews.extend(parsed)
    else:
        has_next = True

    if first_payload:
        skip += limit

    while has_next and len(reviews) < max_reviews:
        request_payload = json.loads(json.dumps(base_request_payload))
        if not isinstance(request_payload, dict):
            break
        if not isinstance(request_payload.get("data"), dict):
            request_payload["data"] = {}
        request_payload["data"]["limit"] = str(limit)

        # Ưu tiên cursor pagination; fallback về skip/limit nếu API không có cursor.
        if next_cursor:
            cursor_updated = set_cursor_in_payload(request_payload, next_cursor)
            if not cursor_updated:
                request_payload["data"]["cursor"] = next_cursor
        else:
            request_payload["data"]["skip"] = str(skip)

        try:
            response = await context.request.post(
                api_url,
                data=request_payload,
                headers=replay_headers if replay_headers else None,
                timeout=TIMEOUT,
            )
            if not response.ok:
                logger.warning(f"[Detail] Replay getReviews failed: HTTP_{response.status}")
                break
            payload = await response.json()
        except Exception as e:
            logger.warning(f"[Detail] Replay getReviews error: {e}")
            break

        parsed, has_next, parsed_next_cursor = parse_reviews_from_api_payload(
            payload,
            source_hotel_id,
            hotel_name,
            location,
            overall_rating,
            seen_review_ids,
        )
        reviews.extend(parsed)
        if parsed_next_cursor:
            next_cursor = parsed_next_cursor
        elif not parsed:
            next_cursor = ""
        skip += limit
        await random_delay(0.2, 0.6)

    return reviews[:max_reviews]

async def scrape_reviews_from_detail(page, hotel_name, city_name, city_code, detail_url, max_reviews):
    """Ưu tiên lấy review từ API getReviews, fallback DOM nếu cần."""
    captured_review_responses = []
    ugc_seed_headers = {}
    context = page.context
    detail_url_snapshot = detail_url
    source_hotel_id = extract_source_hotel_id(detail_url)
    if not source_hotel_id:
        logger.warning(f"[Detail] Skip hotel because missing source_hotel_id from URL: {detail_url}")
        return []

    def on_response(response):
        url_lower = response.url.lower()
        req = response.request
        referer = (req.headers or {}).get("referer", "")
        referer_lower = referer.lower()
        detail_url_lower = detail_url_snapshot.lower()

        # Chỉ nhận network phát sinh từ đúng detail page hiện tại.
        if detail_url_lower and detail_url_lower not in referer_lower:
            return

        is_essential_review_api = req.method == "POST" and any(
            pattern in url_lower for pattern in ESSENTIAL_REVIEW_API_PATTERNS
        )
        if not is_essential_review_api:
            return

        # Bắt ở context-level vì getReviews có thể phát sinh từ worker/frame.
        if "/api/v2/ugc/review/consumption/v2/" in url_lower:
            # Lưu headers UGC làm seed để replay kể cả khi miss getReviews.
            if not ugc_seed_headers:
                ugc_seed_headers.update(req.headers or {})
        if "getreviews" in url_lower and req.method == "POST":
            post_data = req.post_data or ""
            object_id = ""
            if post_data:
                try:
                    parsed_post = json.loads(post_data)
                    object_id = str((parsed_post.get("data", {}) or {}).get("objectId", "")).strip()
                except Exception:
                    object_id = ""
            # Chỉ bắt response đúng objectId của hotel đang crawl.
            if source_hotel_id and object_id and object_id != source_hotel_id:
                return
            captured_review_responses.append(response)

    context.on("response", on_response)
    await human_scroll(page, times=2)

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one("h1")
    hotel_name_extracted = h1.get_text(strip=True) if h1 else hotel_name
    overall_rating = ""
    rating_match = re.search(r'(\d+[\.,]\d+)\s*/\s*10', soup.get_text())
    if rating_match:
        overall_rating = normalize_score_to_5(rating_match.group(1))

    location_raw = extract_location_from_soup(soup, city_name, city_name)
    location = normalize_location_to_city(location_raw, city_name)

    logger.info(f"[Detail] Source hotel ID: {source_hotel_id or 'N/A'}")
    logger.info(f"[Detail] Name: {hotel_name_extracted} | Rating: {overall_rating}/5 | Location: {location}")

    try:
        await page.get_by_text("Đánh giá", exact=True).first.click()
        await random_delay(2, 4)
        await human_scroll(page, times=2)
        await asyncio.sleep(1.5)
    except Exception as e:
        logger.warning(f"[Detail] Could not capture getReviews request after clicking tab: {e}")
    finally:
        context.remove_listener("response", on_response)

    if captured_review_responses:
        logger.info(f"[Detail] Captured {len(captured_review_responses)} getReviews response(s)")

    seed_response = captured_review_responses[0] if captured_review_responses else None
    if not ugc_seed_headers:
        try:
            ugc_seed_headers = await build_dynamic_ugc_headers(page, detail_url)
            logger.info("[Detail] Built dynamic UGC headers from session")
        except Exception as e:
            logger.warning(f"[Detail] Could not build dynamic UGC headers: {e}")

    api_reviews = await fetch_reviews_via_api(
        context,
        source_hotel_id,
        hotel_name_extracted,
        location,
        overall_rating,
        seed_response=seed_response,
        seed_headers=ugc_seed_headers if ugc_seed_headers else None,
        detail_url=detail_url,
        max_reviews=max_reviews,
    )
    if api_reviews:
        logger.info(f"[Detail] Extracted {len(api_reviews)} reviews via API for {hotel_name_extracted}")
        return api_reviews

    logger.warning(
        f"[Detail] Skipping {hotel_name_extracted}: no API reviews. "
        "DOM fallback is disabled to avoid mixing reviewReply."
    )
    return []


def canonicalize_and_deduplicate_reviews(reviews):
    """Deduplicate by review_id and unify hotel_name/location per source_hotel_id."""
    if not reviews:
        return []

    location_counter = {}
    hotel_name_counter = {}
    for row in reviews:
        source_hotel_id = str(row.get("source_hotel_id", "")).strip()
        if not source_hotel_id:
            continue
        location = str(row.get("location", "")).strip()
        hotel_name = str(row.get("hotel_name", "")).strip()
        if source_hotel_id not in location_counter:
            location_counter[source_hotel_id] = {}
        if source_hotel_id not in hotel_name_counter:
            hotel_name_counter[source_hotel_id] = {}
        if location:
            location_counter[source_hotel_id][location] = location_counter[source_hotel_id].get(location, 0) + 1
        if hotel_name:
            hotel_name_counter[source_hotel_id][hotel_name] = hotel_name_counter[source_hotel_id].get(hotel_name, 0) + 1

    canonical_location = {
        hid: max(counts.items(), key=lambda kv: kv[1])[0] for hid, counts in location_counter.items() if counts
    }
    canonical_hotel_name = {
        hid: max(counts.items(), key=lambda kv: kv[1])[0] for hid, counts in hotel_name_counter.items() if counts
    }

    dedup_map = {}
    for row in reviews:
        review_id = str(row.get("review_id", "")).strip()
        if not review_id:
            continue
        normalized = dict(row)
        source_hotel_id = str(normalized.get("source_hotel_id", "")).strip()
        if source_hotel_id:
            if source_hotel_id in canonical_location:
                normalized["location"] = canonical_location[source_hotel_id]
            if source_hotel_id in canonical_hotel_name:
                normalized["hotel_name"] = canonical_hotel_name[source_hotel_id]
        dedup_map[review_id] = normalized
    return list(dedup_map.values())

# ============================================================
# MAIN
# ============================================================

async def run_crawler():
    if not CITIES:
        logger.error("No cities in config!")
        return

    logger.info("=" * 60)
    logger.info("STARTING TRAVELOKA CRAWLER v3")
    logger.info(f"Cities: {[c['name'] for c in CITIES]}")
    logger.info(f"Max hotels/city: {MAX_HOTELS}, Max reviews/hotel: {MAX_REVIEWS}")
    logger.info(f"Max concurrent hotels: {MAX_CONCURRENT_HOTELS}")
    logger.info(f"Max concurrent cities: {MAX_CONCURRENT_CITIES}")
    logger.info("=" * 60)

    # Clear old checkpoint
    checkpoint = RAW_DIR / "traveloka_checkpoint.jsonl"
    if checkpoint.exists():
        checkpoint.unlink()

    proxy_pool = build_proxy_pool()
    async with async_playwright() as p:
        all_reviews = []
        for attempt_idx, proxy in enumerate(proxy_pool, start=1):
            proxy_label = "direct_ip" if not proxy else proxy.get("server", "unknown_proxy")
            logger.info(f"[Proxy] Attempt {attempt_idx}/{len(proxy_pool)} using: {proxy_label}")

            launch_kwargs = {
                "user_data_dir": str(CHROME_PROFILE_DIR),
                "headless": HEADLESS,
                "viewport": {"width": 1920, "height": 1080},
                "locale": "vi-VN",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
                "ignore_default_args": ["--enable-automation"],
            }
            if proxy:
                launch_kwargs["proxy"] = proxy

            context = await p.chromium.launch_persistent_context(**launch_kwargs)
            try:
                # Apply stealth plugin nếu có
                if HAS_STEALTH:
                    for pg in context.pages:
                        await stealth_async(pg)
                    logger.info("[Stealth] playwright-stealth applied")

                await optimize_context_for_mac(context)
                logger.info(f"[Perf] Blocking resource types: {sorted(BLOCK_RESOURCE_TYPES)}")
                claimed_hotel_ids = set()
                claimed_hotel_lock = asyncio.Lock()

                async def crawl_city_safe(city):
                    logger.info(f"\n{'='*50}")
                    logger.info(f"CRAWLING: {city['name']} (region={city.get('region', 'unknown')})")
                    logger.info(f"{'='*50}")
                    try:
                        return await asyncio.wait_for(
                            crawl_hotels_in_city(context, city, claimed_hotel_ids, claimed_hotel_lock),
                            timeout=CITY_TASK_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[CityTimeout] {city['name']} exceeded {CITY_TASK_TIMEOUT_SECONDS}s, skip city")
                        return []
                    except Exception as e:
                        logger.error(f"[CityError] {city['name']}: {e}")
                        return []

                city_error_count = 0
                logger.info("[CrawlMode] Sequential city mode enabled (no region workers).")
                for city in CITIES:
                    records = await crawl_city_safe(city)
                    if not records:
                        city_error_count += 1
                    all_reviews.extend(records)

                # Nếu dùng proxy mà fail toàn bộ city, thử proxy kế tiếp.
                if proxy and city_error_count == len(CITIES):
                    logger.warning(f"[Proxy] All cities failed with proxy {proxy_label}, trying next proxy...")
                    continue
                break
            finally:
                await context.close()

        all_reviews = canonicalize_and_deduplicate_reviews(all_reviews)

        # Final save
        final_file = RAW_DIR / "traveloka_raw_final.json"
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"CRAWLER FINISHED!")
        logger.info(f"Total reviews: {len(all_reviews)}")
        logger.info(f"Saved to: {final_file}")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(run_crawler())
