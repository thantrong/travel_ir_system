"""
Crawl_Hotel
-----------
Crawler mới để thu thập HTML trang chi tiết khách sạn sau render.

Thiết kế:
- Discovery khách sạn qua response API search list (nhanh, ít click UI).
- Mỗi khách sạn mở detail URL; lưu HTML gốc sau render (raw, không compact)
  để đảm bảo tính toàn vẹn dữ liệu nguồn cho bước clean phía sau.
- "HTML tĩnh" từ server không chứa DOM sau React — ta chụp snapshot sau
  scroll + chờ network + mở modal/expand (xem _materialize_dom_snapshot).
- Không thay đổi crawler IR cũ.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse, urlunparse

import yaml
from bs4 import BeautifulSoup, Comment
from playwright.async_api import async_playwright

try:
    from playwright_stealth import stealth_async
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import crawler.traveloka_crawler as ir_crawler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("crawler_hotel.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"
CFG = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
TRAVELOKA_CFG = (CFG.get("crawler", {}) or {}).get("traveloka", {}) or {}

TIMEOUT = int(TRAVELOKA_CFG.get("timeout", 60000))
HEADLESS = bool(TRAVELOKA_CFG.get("headless", False))
BLOCK_RESOURCE_TYPES = set(TRAVELOKA_CFG.get("block_resource_types", ["media", "font"]))
RAW_DIR = PROJECT_ROOT / (CFG.get("data", {}) or {}).get("raw_dir", "data/raw")
HOTEL_DETAIL_ROOT = RAW_DIR / "hotel_detail"
DISCOVERY_ROOT = RAW_DIR / "discovery"
HOTEL_DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
DISCOVERY_ROOT.mkdir(parents=True, exist_ok=True)
ROOM_API_PATTERNS = (
    "/api/v2/hotel/search/rooms",
    "/api/v2/hotel/room",
    "/api/v2/accom/room",
)

# Ảnh khách sạn: cùng file asset, chỉ khác query resize (imagekit / CDN).
_HOTEL_ASSET_URL_RE = re.compile(
    r"(https?://[^\s\"'<>]+/hotel/asset/[0-9]+-[0-9a-f]{32}\.(?:jpe?g|png|webp))\?[^\s\"'<>]*",
    re.IGNORECASE,
)


def normalize_traveloka_image_url(url: str) -> str:
    """Bỏ query (tr=, w-, h-, _src=…) — giữ một URL gốc cho mỗi asset."""
    u = (url or "").strip()
    if not u.startswith("http") or "/hotel/asset/" not in u.lower():
        return u
    if "?" not in u:
        return u
    parsed = urlparse(u)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def normalize_srcset_attr(value: str) -> str:
    if not (value or "").strip():
        return value
    out = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if parts:
            parts[0] = normalize_traveloka_image_url(parts[0])
            out.append(" ".join(parts))
    return ", ".join(out)


def strip_hotel_asset_query_params(html: str) -> str:
    """Trong toàn bộ HTML (kể cả JSON trong script): gom URL ảnh asset về dạng không query."""
    return _HOTEL_ASSET_URL_RE.sub(r"\1", html)


def minify_next_data_script(soup: BeautifulSoup) -> None:
    """Next.js __NEXT_DATA__: bỏ whitespace thừa trong JSON (giảm mạnh dung lượng)."""
    for script in soup.find_all("script"):
        if str(script.get("id", "") or "").strip() != "__NEXT_DATA__":
            continue
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            script.clear()
            script.append(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass


def hotel_id_from_detail_url(url: str) -> str:
    """Lấy source_hotel_id từ segment cuối URL (vd: .../ht-house-3000010028716)."""
    tail = url.rstrip("/").split("/")[-1]
    if "-" in tail:
        tail = tail.split("-")[-1]
    return tail if tail.isdigit() else ""


def normalize_space(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def parse_args():
    ap = argparse.ArgumentParser(description="Crawl full HTML detail page for hotels")
    ap.add_argument("--cities", type=str, default="", help="City code list, vd: --cities LD,PQ")
    ap.add_argument("--geo-id", type=str, default="", help="Override geo_id, vd: 10010000")
    ap.add_argument("--city-name", type=str, default="", help="Override city name, vd: Lâm Đồng")
    ap.add_argument("--city-code", type=str, default="CUSTOM", help="Override city code")
    ap.add_argument("--max-hotels", type=int, default=2, help="Số khách sạn cần lấy HTML")
    ap.add_argument("--max-workers", type=int, default=2, help="Số tab chạy song song (mặc định 2)")
    ap.add_argument(
        "--detail-url",
        action="append",
        default=[],
        dest="detail_urls",
        help="URL trang detail khách sạn (lặp flag để thêm nhiều URL; bỏ qua discovery).",
    )
    return ap.parse_args()


def load_enabled_cities() -> List[Dict]:
    return ir_crawler.load_city_configs()


def select_cities(args) -> List[Dict]:
    if args.geo_id and args.city_name:
        return [{
            "geo_id": str(args.geo_id).strip(),
            "name": str(args.city_name).strip(),
            "code": str(args.city_code or "CUSTOM").strip().upper(),
        }]

    selected = load_enabled_cities()
    if args.cities:
        wanted = {c.strip().upper() for c in args.cities.split(",") if c.strip()}
        selected = [c for c in selected if str(c.get("code", "")).upper() in wanted]
    return selected


def save_discovery(city: Dict, targets: List[Dict]) -> Path:
    city_code = str(city.get("code", "XX")).strip().lower()
    city_geo = str(city.get("geo_id", "")).strip()
    out = DISCOVERY_ROOT / f"{city_code}_{city_geo}_htmlcrawl.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in targets:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def build_html_path(city: Dict, target: Dict) -> Path:
    source_hotel_id = str(target.get("source_hotel_id", "")).strip() or "unknown"
    hotel_dir = HOTEL_DETAIL_ROOT / source_hotel_id
    hotel_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return hotel_dir / f"{day}.html"


def build_room_api_dir(city: Dict, target: Dict) -> Path:
    source_hotel_id = str(target.get("source_hotel_id", "")).strip() or "unknown"
    out_dir = HOTEL_DETAIL_ROOT / source_hotel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_manifest_path(target: Dict) -> Path:
    source_hotel_id = str(target.get("source_hotel_id", "")).strip() or "unknown"
    hotel_dir = HOTEL_DETAIL_ROOT / source_hotel_id
    hotel_dir.mkdir(parents=True, exist_ok=True)
    return hotel_dir / "manifest.json"


def build_rooms_api_path(target: Dict) -> Path:
    source_hotel_id = str(target.get("source_hotel_id", "")).strip() or "unknown"
    hotel_dir = HOTEL_DETAIL_ROOT / source_hotel_id
    hotel_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return hotel_dir / f"rooms_{day}.json"


def _script_is_hotel_relevant(script_text: str, script_type: str, script_id: str) -> bool:
    """Giữ script có dữ liệu khách sạn / JSON-LD / phòng / gallery; bỏ bundle rác."""
    t = (script_type or "").lower().strip()
    if t in ("application/ld+json", "application/ld+json; charset=utf-8"):
        return True
    if "ld+json" in t:
        return True

    s = script_text or ""
    if len(s) < 40:
        return False

    # Dấu hiệu dữ liệu nghiệp vụ (không dùng chữ "traveloka.com" — quá rộng, giữ nhầm bundle).
    markers = (
        "hotelRoomId",
        "recommendedEntries",
        "hotel/asset/",
        "imageWithCaptions",
        "FAQPage",
        "BreadcrumbList",
        "LodgingBusiness",
        "streetAddress",
        "acceptedAnswer",
        "mainEntity",
        "aggregateRating",
    )
    if any(m in s for m in markers):
        return True
    # Script nhỏ có thể là bootstrap UI — bỏ nếu không khớp marker.
    low = s.lower()
    junk = (
        "googletagmanager",
        "google-analytics",
        "gtag(",
        "fbevents",
        "facebook.net",
        "hotjar",
        "segment.",
    )
    if any(j in low for j in junk):
        return False
    return False


def compact_html_for_storage(html: str) -> str:
    """
    Giảm dung lượng file HTML đã render mà vẫn đủ cho pipeline parse:
    - Giữ toàn bộ body (text amenities/policy + DOM cần select)
    - Giữ [role=dialog] sau khi đã mở modal chính sách (aria-modal giữ tối thiểu)
    - Giữ script có JSON-LD / dữ liệu phòng / asset ảnh / FAQ trong markup
    - Bỏ style, link, meta, iframe, svg, canvas, comment, phần lớn attribute dư
    - Minify khoảng trắng giữa thẻ (không đụng nội dung script JSON)
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        for node in soup.find_all(string=lambda t: isinstance(t, Comment)):
            node.extract()

        page_title = ""
        if soup.title:
            page_title = (soup.title.get_text(" ", strip=True) or "").strip()

        for tag in soup.find_all(["style", "link", "meta", "noscript", "iframe", "svg", "canvas", "template"]):
            tag.decompose()

        for script in list(soup.find_all("script")):
            st = str(script.get("type", "") or "")
            sid = str(script.get("id", "") or "")
            body = script.string or script.get_text() or ""
            if _script_is_hotel_relevant(body, st, sid):
                continue
            script.decompose()

        # Giữ aria-modal / aria-labelledby cho [role=dialog] (modal chính sách).
        attr_whitelist = {
            "id", "role", "href", "src", "alt", "title", "type", "itemprop", "name", "content", "srcset",
            "aria-modal", "aria-labelledby", "aria-describedby",
        }
        for tag in soup.find_all(True):
            for attr in list(tag.attrs.keys()):
                al = attr.lower()
                if al == "style":
                    del tag.attrs[attr]
                    continue
                role_val = str(tag.get("role", "") or "").lower()
                if role_val == "dialog" and al in ("aria-modal", "aria-labelledby", "aria-describedby"):
                    continue
                if al in attr_whitelist:
                    continue
                if al.startswith("data-"):
                    del tag.attrs[attr]
                    continue
                if al.startswith("aria-"):
                    del tag.attrs[attr]
                    continue
                del tag.attrs[attr]

        for tag in soup.find_all("img"):
            src = tag.get("src")
            if src:
                tag["src"] = normalize_traveloka_image_url(str(src))
        for tag in soup.find_all(True):
            if tag.get("srcset"):
                tag["srcset"] = normalize_srcset_attr(str(tag.get("srcset")))

        if soup.html:
            soup.html.attrs = {}

        # JSON-LD / script dữ liệu thường nằm trong <head>; clear head sẽ xóa mất.
        if soup.head and soup.body:
            for script in list(soup.head.find_all("script")):
                soup.body.append(script)

        if soup.head:
            soup.head.clear()
            if page_title:
                t = soup.new_tag("title")
                t.string = page_title
                soup.head.append(t)

        minify_next_data_script(soup)

        out = str(soup)
        out = strip_hotel_asset_query_params(out)
        out = re.sub(r">\s+<", "><", out)
        return out.strip()
    except Exception:
        return html


async def collect_targets_from_search(context, city: Dict, max_hotels: int) -> List[Dict]:
    city_name = city["name"]
    geo_id = city["geo_id"]
    search_url = ir_crawler.build_search_url(geo_id, city_name)
    search_page = await context.new_page()
    targets_map = {}
    order = []

    async def _handle_response(resp):
        try:
            if resp.status != 200:
                return
            low = str(resp.url or "").lower()
            if "traveloka.com" not in low or "/api/" not in low:
                return
            if "/api/v2/ugc/review/consumption/v2/" in low:
                return

            ctype = str((resp.headers or {}).get("content-type", "")).lower()
            if "json" not in ctype:
                return
            payload = await resp.json()
            extracted = ir_crawler.extract_hotel_targets_from_payload(payload)
            if not extracted:
                return

            before = len(targets_map)
            for target in extracted:
                sid = str(target.get("source_hotel_id", "")).strip()
                detail_url = str(target.get("detail_url", "")).strip()
                key = sid or detail_url
                if not key or key in targets_map:
                    continue
                targets_map[key] = {
                    "hotel_name": str(target.get("hotel_name", "")).strip() or "Unknown Hotel",
                    "detail_url": detail_url,
                    "source_hotel_id": sid,
                }
                order.append(key)
            if len(targets_map) > before:
                logger.info(
                    "[SearchAPI] +%s targets from %s (total=%s)",
                    len(targets_map) - before,
                    low.split("?", 1)[0],
                    len(targets_map),
                )
        except Exception:
            return

    def _on_response(resp):
        asyncio.create_task(_handle_response(resp))

    search_page.on("response", _on_response)
    try:
        logger.info("[Search] Loading: %s", search_url)
        await search_page.goto(search_url, timeout=TIMEOUT, wait_until="domcontentloaded")
        await asyncio.sleep(7)
        for _ in range(6):
            if len(targets_map) >= max_hotels:
                break
            await ir_crawler.human_scroll(search_page, times=3)
            await asyncio.sleep(2)
        targets = [targets_map[k] for k in order][:max_hotels]
        logger.info("[Search] Final targets: %s", len(targets))
        logger.info("[Search] Sample: %s", targets[:2])
        return targets
    finally:
        await search_page.close()


async def capture_full_html(context, city: Dict, targets: List[Dict], max_workers: int) -> List[Dict]:
    results = []
    max_workers = max(1, int(max_workers or 1))

    async def _wait_for_room_api(page, captured_room_responses, timeout_seconds: float) -> bool:
        """Chờ đến khi có room API hoặc hết timeout; trả về True nếu đã bắt được."""
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        # Poll ngắn để thoát sớm ngay khi đã bắt được response.
        while asyncio.get_event_loop().time() < deadline:
            if captured_room_responses:
                return True
            await asyncio.sleep(0.25)
        return bool(captured_room_responses)

    async def _trigger_room_ui(page):
        """Kích hoạt section phòng để frontend có khả năng bắn rooms API."""
        try:
            await page.mouse.wheel(0, 1200)
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 1800)
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 2400)
        except Exception:
            pass

        # Thử click các nút/link thường dùng để mở danh sách phòng.
        candidate_texts = [
            "Xem phòng",
            "Chọn phòng",
            "Xem tất cả phòng",
            "Room",
            "Rooms",
            "Select Room",
        ]
        for text in candidate_texts:
            try:
                locator = page.get_by_text(text, exact=False).first
                if await locator.count() > 0:
                    await locator.scroll_into_view_if_needed(timeout=2000)
                    await locator.click(timeout=2000)
                    await asyncio.sleep(0.8)
                    return
            except Exception:
                continue

    async def _wait_network_idle_soft(page) -> None:
        """SPA thường không bao giờ 'idle' hẳn — chờ tối đa ~12s, lỗi thì bỏ qua."""
        try:
            await page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass

    async def _scroll_full_page_for_lazy(page) -> None:
        """Scroll từ đầu đến cuối để trigger lazy render (ảnh, section, intersection observer)."""
        try:
            await page.evaluate(
                """
                async () => {
                    const delay = (ms) => new Promise((r) => setTimeout(r, ms));
                    const h = Math.max(
                        document.documentElement.scrollHeight,
                        document.body ? document.body.scrollHeight : 0,
                        0
                    );
                    for (let y = 0; y < h; y += 550) {
                        window.scrollTo(0, y);
                        await delay(55);
                    }
                    window.scrollTo(0, h);
                    await delay(180);
                    window.scrollTo(0, 0);
                    await delay(120);
                }
                """
            )
        except Exception:
            pass

    async def _materialize_dom_snapshot(page) -> bool:
        """
        Để chuỗi HTML lưu ra *gần* với DOM đầy đủ sau render:
        1) chờ mạng tương đối ổn định
        2) mở tab chính sách + modal "Đọc tất cả" (nếu có)

        Không thể đảm bảo toàn bộ tab/modal (cần click từng loại UI nếu cần).
        """
        await _wait_network_idle_soft(page)
        await asyncio.sleep(0.35)
        return await _expand_accommodation_policy(page)

    async def _scroll_policy_dialog_content(page) -> None:
        """Cuộn trong modal để nội dung dài (cọc, trẻ em) được mount đầy đủ."""
        dlg = page.locator('[role="dialog"]').first
        try:
            if await dlg.count() == 0:
                return
            await dlg.evaluate(
                """
                (el) => {
                    for (let i = 0; i < 10; i++) {
                        el.scrollTop = el.scrollHeight;
                    }
                }
            """
            )
            await asyncio.sleep(0.35)
        except Exception:
            pass

    async def _expand_accommodation_policy(page) -> bool:
        """
        Chính sách lưu trú: scroll tới khối 'Chính sách', bấm 'Đọc tất cả',
        cuộn trong modal — DOM đầy đủ mới serialize vào HTML lưu file.
        """
        async def _click_policy_tab_once() -> bool:
            tab_candidates = [
                # Selector ổn định do người dùng cung cấp.
                page.locator("[data-testid='link-POLICY']").first,
                page.get_by_role("tab", name=re.compile(r"^\s*Chính\s*sách\s*$", re.I)).first,
                page.get_by_role("link", name=re.compile(r"^\s*Chính\s*sách\s*$", re.I)).first,
                page.get_by_role("button", name=re.compile(r"^\s*Chính\s*sách\s*$", re.I)).first,
                page.locator("a,button,div[role='button']", has_text=re.compile(r"^\s*Chính\s*sách\s*$", re.I)).first,
            ]
            for tab in tab_candidates:
                try:
                    if await tab.count() == 0:
                        continue
                    await tab.scroll_into_view_if_needed(timeout=5000)
                    await asyncio.sleep(0.2)
                    await tab.click(timeout=5000, force=True)
                    await asyncio.sleep(1.0)
                    logger.info("[Policy] Đã click tab/menu 'Chính sách'")
                    return True
                except Exception:
                    continue
            return False

        headings = (
            "Chính sách lưu trú",
            "Chính sách khách sạn",
            "Chính sách nhận phòng",
            "Accommodation policy",
            "Hotel policy",
            "Chính sách",
        )
        tab_clicked = await _click_policy_tab_once()
        if not tab_clicked:
            logger.info("[Policy] Chưa click được tab/menu 'Chính sách' ở lượt chính")

        await _wait_network_idle_soft(page)
        await asyncio.sleep(0.4)

        # Một lượt chính: không lặp nhiều vòng/reload để giảm tín hiệu anti-bot.
        for heading in headings:
            try:
                h = page.get_by_text(heading, exact=False).first
                if await h.count() == 0:
                    continue
                await h.scroll_into_view_if_needed(timeout=6000)
                await asyncio.sleep(0.5)
                logger.info("[Policy] Đã scroll tới khu vực %r", heading)
                # Chỉ click đúng nút "Đọc tất cả" trong hoặc ngay sau khu vực chính sách.
                read_btn = h.locator(
                    "xpath=ancestor::*[self::section or self::article or self::div][1]"
                    "//a[contains(normalize-space(.), 'Đọc tất cả')] | "
                    "ancestor::*[self::section or self::article or self::div][1]"
                    "//button[contains(normalize-space(.), 'Đọc tất cả')] | "
                    "ancestor::*[self::section or self::article or self::div][1]"
                    "//div[@role='button' and contains(normalize-space(.), 'Đọc tất cả')]"
                ).first
                if await read_btn.count() == 0:
                    read_btn = h.locator(
                        "xpath=ancestor::*[self::section or self::article or self::div][1]"
                        "/following-sibling::*[1]//a[contains(normalize-space(.), 'Đọc tất cả')] | "
                        "ancestor::*[self::section or self::article or self::div][1]"
                        "/following-sibling::*[1]//button[contains(normalize-space(.), 'Đọc tất cả')] | "
                        "ancestor::*[self::section or self::article or self::div][1]"
                        "/following-sibling::*[1]//div[@role='button' and contains(normalize-space(.), 'Đọc tất cả')]"
                    ).first
                # Fallback theo DOM thực tế Traveloka: div aria-live polite + role=button.
                if await read_btn.count() == 0:
                    read_btn = page.locator(
                        "div[role='button'][aria-live='polite']",
                        has_text=re.compile(r"Đọc\s*tất\s*cả", re.I),
                    ).first
                if await read_btn.count() == 0:
                    logger.info("[Policy] Không thấy nút 'Đọc tất cả' gần khu vực %r", heading)
                    continue

                await read_btn.scroll_into_view_if_needed(timeout=4000)
                await asyncio.sleep(0.25)
                await read_btn.click(timeout=5000, force=True)
                await asyncio.sleep(1.0)

                dlg = page.locator('[role="dialog"]')
                if await dlg.count() > 0:
                    try:
                        await dlg.first.wait_for(state="visible", timeout=12000)
                        await _scroll_policy_dialog_content(page)
                        logger.info("[Policy] Modal mở bằng nút 'Đọc tất cả' (%r)", heading)
                        return True
                    except Exception:
                        pass
                for hint in ("nhận phòng", "check-in", "Giấy tờ", "cọc", "deposit"):
                    if await page.get_by_text(hint, exact=False).count() > 0:
                        logger.info("[Policy] Có nội dung chính sách sau khi bấm 'Đọc tất cả' (%r)", heading)
                        return True
            except Exception:
                continue
        # Fallback cuối: một số hotel không có nút "Đọc tất cả", nhưng policy hiển thị inline.
        for heading in headings:
            try:
                h = page.get_by_text(heading, exact=False).first
                if await h.count() == 0:
                    continue
                container = h.locator("xpath=ancestor::*[self::section or self::article or self::div][1]").first
                text = normalize_space(await container.inner_text(timeout=5000))
                if len(text) < 120:
                    continue
                lower = text.lower()
                if ("nhận phòng" in lower or "check-in" in lower) and ("trả phòng" in lower or "check-out" in lower):
                    logger.info("[Policy] Không có 'Đọc tất cả' nhưng đã lấy được policy inline (%r)", heading)
                    return True
            except Exception:
                continue
        logger.info("[Policy] Không mở được modal/expand chính sách (bỏ qua, không fail crawl)")
        return False

    def _extract_room_fallback_from_html(html: str):
        """Fallback: bóc dữ liệu room từ script JSON trong HTML khi network không ổn định."""
        if not html:
            return None

        matches = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)
        candidates = []
        for script_body in matches:
            body = script_body.strip()
            if not body or len(body) < 50:
                continue
            if "hotelRoomId" not in body and "recommendedEntries" not in body:
                continue

            # Tìm các object JSON lớn có dấu hiệu room data.
            start = body.find("{")
            end = body.rfind("}")
            if start == -1 or end == -1 or end <= start:
                continue
            raw_json = body[start:end + 1]
            try:
                payload = json.loads(raw_json)
            except Exception:
                continue

            payload_str = json.dumps(payload, ensure_ascii=False)
            if "hotelRoomId" in payload_str or "recommendedEntries" in payload_str:
                candidates.append(payload)

        if not candidates:
            return None

        # Lấy candidate lớn nhất để tăng xác suất chứa đầy đủ room data.
        best = max(candidates, key=lambda x: len(json.dumps(x, ensure_ascii=False)))
        return {
            "source": "html_script_fallback",
            "status": 200,
            "payload": best,
        }

    async def _scrape_one(idx: int, target: Dict):
        page = await context.new_page()
        captured_room_responses = []

        async def _handle_room_response(resp):
            try:
                if resp.status != 200:
                    return
                low = str(resp.url or "").lower()
                if not any(pattern in low for pattern in ROOM_API_PATTERNS):
                    return
                ctype = str((resp.headers or {}).get("content-type", "")).lower()
                if "json" not in ctype:
                    return
                payload = await resp.json()
                captured_room_responses.append({
                    "url": str(resp.url),
                    "status": resp.status,
                    "payload": payload,
                })
            except Exception:
                return

        def _on_response(resp):
            asyncio.create_task(_handle_room_response(resp))

        page.on("response", _on_response)
        try:
            if HAS_STEALTH:
                await stealth_async(page)

            detail_url = target["detail_url"]
            logger.info("[HTML %s/%s] Opening: %s", idx, len(targets), detail_url)
            await page.goto(detail_url, timeout=TIMEOUT, wait_until="domcontentloaded")

            # Stage 1: trigger nhẹ + chờ ngắn.
            await _trigger_room_ui(page)
            has_room_api = await _wait_for_room_api(page, captured_room_responses, timeout_seconds=4.0)

            # Stage 2: chưa có thì trigger lại sâu hơn và chờ thêm.
            if not has_room_api:
                await _trigger_room_ui(page)
                has_room_api = await _wait_for_room_api(page, captured_room_responses, timeout_seconds=5.0)

            # Fallback theo yêu cầu:
            # nếu chưa có API thì đợi 5s, reload lại trang, rồi đợi thêm tối đa 4s.
            if not has_room_api:
                logger.info(
                    "[RoomAPI] No response yet for hotel_id=%s, wait 5s then reload + wait 4s",
                    target.get("source_hotel_id", ""),
                )
                await asyncio.sleep(5)
                await page.reload(timeout=TIMEOUT, wait_until="domcontentloaded")
                await _trigger_room_ui(page)
                has_room_api = await _wait_for_room_api(page, captured_room_responses, timeout_seconds=4.0)

            # Đưa tối đa nội dung lazy vào DOM trước khi serialize HTML (scroll + modal chính sách).
            policy_ok = await _materialize_dom_snapshot(page)

            raw_html = await page.content()
            # Lưu raw HTML sau render để giữ toàn vẹn dữ liệu nguồn.
            html_path = build_html_path(city, target)
            html_path.write_text(raw_html, encoding="utf-8")

            row = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "city": city.get("name"),
                "city_code": city.get("code"),
                "source_hotel_id": target.get("source_hotel_id"),
                "hotel_name": target.get("hotel_name"),
                "detail_url": detail_url,
                "html_path": str(html_path),
                "html_size_bytes": len(raw_html.encode("utf-8")),
                "html_raw_integrity": True,
                "policy_modal_attempted": True,
                "policy_modal_visible": bool(policy_ok),
                "room_api_files": [],
            }

            room_api_dir = build_room_api_dir(city, target)
            rooms_api_path = build_rooms_api_path(target)
            if captured_room_responses:
                merged = {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "hotel_id": target.get("source_hotel_id") or "",
                    "detail_url": detail_url,
                    "responses": captured_room_responses,
                }
                rooms_api_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                row["room_api_files"].append(str(rooms_api_path))
            else:
                fallback_payload = _extract_room_fallback_from_html(raw_html)
                if fallback_payload:
                    merged = {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "hotel_id": target.get("source_hotel_id") or "",
                        "detail_url": detail_url,
                        "responses": [fallback_payload],
                    }
                    rooms_api_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                    row["room_api_files"].append(str(rooms_api_path))
                    logger.info(
                        "[RoomAPI] Used HTML fallback for hotel_id=%s",
                        target.get("source_hotel_id", ""),
                    )

            results.append(row)
            manifest_path = build_manifest_path(target)
            manifest_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(
                "[HTML] Saved: %s (%s bytes), room_api=%s",
                html_path,
                row["html_size_bytes"],
                len(row["room_api_files"]),
            )
        except Exception as e:
            logger.error("[HTML] Failed %s: %s", target.get("hotel_name"), e)
        finally:
            await page.close()

    sem = asyncio.Semaphore(max_workers)

    async def _runner(idx: int, target: Dict):
        async with sem:
            return await _scrape_one(idx, target)

    tasks = [asyncio.create_task(_runner(i, t)) for i, t in enumerate(targets, start=1)]
    if tasks:
        await asyncio.gather(*tasks)
    return results


async def run():
    args = parse_args()
    cities = select_cities(args)
    if not cities:
        logger.error("No city selected. Dùng --geo-id/--city-name hoặc bật enabled=true trong cities.yaml.")
        return

    max_hotels = max(1, int(args.max_hotels))
    max_workers = max(1, int(getattr(args, "max_workers", 2) or 2))
    logger.info(
        "Starting Crawl_Hotel for cities=%s, max_hotels=%s, max_workers=%s",
        [c["name"] for c in cities],
        max_hotels,
        max_workers,
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            locale="vi-VN",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )

        async def _route_handler(route):
            req = route.request
            if req.resource_type in BLOCK_RESOURCE_TYPES:
                await route.abort()
                return
            await route.continue_()

        await context.route("**/*", _route_handler)

        try:
            for city in cities:
                logger.info("=== CITY: %s (%s) ===", city["name"], city["geo_id"])
                if getattr(args, "detail_urls", None):
                    targets = []
                    for u in args.detail_urls:
                        u = str(u).strip()
                        if not u:
                            continue
                        hid = hotel_id_from_detail_url(u)
                        targets.append(
                            {
                                "hotel_name": f"hotel_{hid}" if hid else "direct",
                                "detail_url": u,
                                "source_hotel_id": hid,
                            }
                        )
                    logger.info("[Targets] Direct URLs: %s", len(targets))
                else:
                    targets = await collect_targets_from_search(context, city, max_hotels)
                if not targets:
                    logger.warning("No targets found for %s", city["name"])
                    continue
                if not getattr(args, "detail_urls", None):
                    discovery_path = save_discovery(city, targets)
                    logger.info("[Discovery] Saved: %s", discovery_path)
                captured = await capture_full_html(context, city, targets, max_workers=max_workers)
                report_path = HOTEL_DETAIL_ROOT / f"{city['code'].lower()}_capture_report.json"
                report_path.write_text(json.dumps(captured, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("[Report] Saved %s rows to %s", len(captured), report_path)
        finally:
            await context.close()
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run())

