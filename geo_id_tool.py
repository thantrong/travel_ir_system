#!/usr/bin/env python3
"""
Tool tìm geo_id Traveloka cho danh sách thành phố trong config/cities.yaml.

Cách dùng:
  ./venv/bin/python geo_id_tool.py
  ./venv/bin/python geo_id_tool.py --apply
  ./venv/bin/python geo_id_tool.py --all --headless false
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata

import yaml
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


AUTOCOMPLETE_API_PATH = "/api/v1/hotel/autocomplete"


@dataclass
class LookupResult:
    city_name: str
    code: str
    current_geo_id: str
    found_geo_id: str
    matched_name: str
    matched_type: str
    matched_country: str
    match_score: int
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find Traveloka geo_id from cities.yaml")
    parser.add_argument(
        "--cities-file",
        default="config/cities.yaml",
        help="Path to cities.yaml",
    )
    parser.add_argument(
        "--output",
        default="config/city_geo_lookup_results.json",
        help="Path to save lookup results JSON",
    )
    parser.add_argument(
        "--headless",
        default="true",
        choices=["true", "false"],
        help="Run browser headless or not",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write found geo_id back to cities.yaml for rows currently missing geo_id",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Lookup all rows (default only rows missing geo_id)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of cities to process (0 = all)",
    )
    parser.add_argument(
        "--type-delay-ms",
        type=int,
        default=80,
        help="Typing delay per character to trigger autocomplete",
    )
    parser.add_argument(
        "--allow-geo-area",
        action="store_true",
        help="Allow geoAreaContent fallback (off by default to avoid wrong district-level matches)",
    )
    parser.add_argument(
        "--min-apply-score",
        type=int,
        default=260,
        help="Minimum match_score to write geo_id when using --apply",
    )
    return parser.parse_args()


def load_cities(cities_file: Path) -> dict[str, Any]:
    if not cities_file.exists():
        raise FileNotFoundError(f"Cities file not found: {cities_file}")
    payload = yaml.safe_load(cities_file.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Invalid cities file format: root must be a mapping")
    if not isinstance(payload.get("cities"), list):
        raise ValueError("Invalid cities file format: key 'cities' must be a list")
    return payload


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def build_query_variants(city_name: str) -> list[str]:
    raw = str(city_name or "").strip()
    if not raw:
        return []
    variants = [raw]
    lowered = raw.lower()
    prefixes = ["tp. ", "tp ", "thành phố "]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            stripped = raw[len(prefix):].strip()
            if stripped:
                variants.append(stripped)
    # remove duplicates while preserving order
    uniq: list[str] = []
    seen = set()
    for v in variants:
        key = normalize_text(v)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    return uniq


def fold_text(value: str) -> str:
    text = normalize_text(value)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def is_vietnam_candidate(row: dict[str, Any]) -> bool:
    country = str(row.get("country", "")).strip().upper()
    name_combo = normalize_text(f"{row.get('name', '')} {row.get('displayName', '')} {row.get('additionalInfo', '')}")
    if country == "VN":
        return True
    if "việt nam" in name_combo or "vietnam" in name_combo:
        return True
    disallow_markers = [
        "trung quốc",
        "china",
        "thái lan",
        "thailand",
        "japan",
        "korea",
        "indonesia",
        "singapore",
        "malaysia",
    ]
    if any(marker in name_combo for marker in disallow_markers):
        return False
    return False


def city_match_score(city_name: str, row: dict[str, Any]) -> int:
    city_fold = fold_text(city_name)
    row_name = str(row.get("displayName", "") or row.get("name", "")).strip()
    row_first_segment = row_name.split(",")[0].strip()
    row_fold = fold_text(row_name)
    first_fold = fold_text(row_first_segment)

    if city_fold == first_fold:
        return 220
    if city_fold == row_fold:
        return 200

    city_tokens = [tok for tok in city_fold.split() if tok]
    row_tokens = [tok for tok in row_fold.split() if tok]
    if city_tokens and row_tokens:
        overlap = sum(1 for tok in city_tokens if tok in row_tokens)
        token_ratio = overlap / max(len(city_tokens), 1)
        if token_ratio >= 1.0:
            return 160
        if token_ratio >= 0.75 and overlap >= 2:
            return 120

    if city_fold and city_fold in row_fold:
        return 90
    return 0


def choose_best_row(city_name: str, data: dict[str, Any], allow_geo_area: bool = False) -> dict[str, Any] | None:
    buckets: list[tuple[str, list[dict[str, Any]]]] = [
        ("geoRegionContent", ((data.get("geoRegionContent") or {}).get("rows") or [])),
        ("geoCityContent", ((data.get("geoCityContent") or {}).get("rows") or [])),
    ]
    if allow_geo_area:
        buckets.append(("geoAreaContent", ((data.get("geoAreaContent") or {}).get("rows") or [])))

    candidates: list[tuple[int, dict[str, Any], str]] = []

    for bucket_name, rows in buckets:
        for row in rows:
            if not is_vietnam_candidate(row):
                continue
            row_name = str(row.get("displayName", "") or row.get("name", "")).strip()
            row_first_segment = row_name.split(",")[0].strip()
            city_fold = fold_text(city_name)
            first_fold = fold_text(row_first_segment)

            # geoArea thường là quận/phường/landmark; chỉ nhận khi segment đầu khớp chính xác tên city.
            if bucket_name == "geoAreaContent" and city_fold != first_fold:
                continue

            score = city_match_score(city_name, row)
            if score < 100:
                # Chặn match mờ (vd Bắc Kạn -> Bắc Kinh).
                continue
            if bucket_name == "geoRegionContent":
                score += 30
            elif bucket_name == "geoCityContent":
                score += 20
            elif bucket_name == "geoAreaContent":
                score += 10
            num_hotels = str(row.get("numHotels", "0"))
            try:
                score += min(int(num_hotels), 200)
            except Exception:
                pass
            candidates.append((score, row, bucket_name))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = dict(candidates[0][1])
    best["_bucket"] = candidates[0][2]
    best["_score"] = candidates[0][0]
    return best


async def lookup_geo_id_for_city(
    page,
    city_name: str,
    type_delay_ms: int,
    allow_geo_area: bool,
) -> tuple[str, str, str, str, int]:
    box = page.get_by_placeholder("Thành phố, khách sạn, điểm đến")
    await box.click(force=True)
    await box.fill("")
    for query_variant in build_query_variants(city_name):
        city_norm = normalize_text(query_variant)

        def _is_target_response(resp) -> bool:
            if AUTOCOMPLETE_API_PATH not in resp.url:
                return False
            if resp.request.method != "POST":
                return False
            try:
                post_data = json.loads(resp.request.post_data or "{}")
                query = normalize_text(((post_data.get("data") or {}).get("query") or ""))
                return query == city_norm
            except Exception:
                return False

        try:
            async with page.expect_response(_is_target_response, timeout=12000) as resp_info:
                await box.fill("")
                await box.type(query_variant, delay=type_delay_ms)
            resp = await resp_info.value
            body = await resp.json()
            data = body.get("data", {}) if isinstance(body, dict) else {}
            best = choose_best_row(query_variant, data, allow_geo_area=allow_geo_area)
            if not best:
                continue
            return (
                str(best.get("id", "")),
                str(best.get("name", "")),
                str(best.get("_bucket", "")),
                str(best.get("country", "")),
                int(best.get("_score", 0)),
            )
        except Exception:
            continue
    return "", "", "", "", 0


async def run_lookup(
    cities: list[dict[str, Any]],
    headless: bool,
    type_delay_ms: int,
    allow_geo_area: bool,
) -> list[LookupResult]:
    results: list[LookupResult] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(locale="vi-VN")
        await page.goto("https://www.traveloka.com/vi-vn/hotel", wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_timeout(3000)

        for idx, city in enumerate(cities, start=1):
            name = str(city.get("name", "")).strip()
            code = str(city.get("code", "")).strip()
            current_geo_id = str(city.get("geo_id", "")).strip()
            if not name:
                continue

            print(f"[{idx}/{len(cities)}] Lookup: {name}")
            try:
                found_geo_id, matched_name, matched_type, matched_country, match_score = await lookup_geo_id_for_city(
                    page,
                    name,
                    type_delay_ms,
                    allow_geo_area,
                )
                if found_geo_id:
                    results.append(
                        LookupResult(
                            city_name=name,
                            code=code,
                            current_geo_id=current_geo_id,
                            found_geo_id=found_geo_id,
                            matched_name=matched_name,
                            matched_type=matched_type,
                            matched_country=matched_country,
                            match_score=match_score,
                            status="found",
                        )
                    )
                    print(f"  -> found geo_id={found_geo_id} ({matched_type}, score={match_score})")
                else:
                    results.append(
                        LookupResult(
                            city_name=name,
                            code=code,
                            current_geo_id=current_geo_id,
                            found_geo_id="",
                            matched_name="",
                            matched_type="",
                            matched_country="",
                            match_score=0,
                            status="not_found",
                            error="No candidate row returned",
                        )
                    )
                    print("  -> not found")
            except PlaywrightTimeoutError:
                results.append(
                    LookupResult(
                        city_name=name,
                        code=code,
                        current_geo_id=current_geo_id,
                        found_geo_id="",
                        matched_name="",
                        matched_type="",
                        matched_country="",
                        match_score=0,
                        status="timeout",
                        error="Timeout waiting autocomplete response",
                    )
                )
                print("  -> timeout")
            except Exception as exc:
                results.append(
                    LookupResult(
                        city_name=name,
                        code=code,
                        current_geo_id=current_geo_id,
                        found_geo_id="",
                        matched_name="",
                        matched_type="",
                        matched_country="",
                        match_score=0,
                        status="error",
                        error=str(exc),
                    )
                )
                print(f"  -> error: {exc}")

        await browser.close()
    return results


def save_results(output_file: Path, results: list[LookupResult]) -> None:
    payload = [
        {
            "city_name": r.city_name,
            "code": r.code,
            "current_geo_id": r.current_geo_id,
            "found_geo_id": r.found_geo_id,
            "matched_name": r.matched_name,
            "matched_type": r.matched_type,
            "matched_country": r.matched_country,
            "match_score": r.match_score,
            "status": r.status,
            "error": r.error,
        }
        for r in results
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_geo_ids(cities_payload: dict[str, Any], results: list[LookupResult], min_apply_score: int) -> int:
    updates = {
        r.city_name: r.found_geo_id
        for r in results
        if r.status == "found" and r.found_geo_id and r.match_score >= min_apply_score
    }
    changed = 0
    for row in cities_payload.get("cities", []):
        name = str(row.get("name", "")).strip()
        current_geo_id = str(row.get("geo_id", "")).strip()
        found_geo_id = updates.get(name, "")
        if not found_geo_id:
            continue
        if current_geo_id == found_geo_id:
            continue
        row["geo_id"] = found_geo_id
        changed += 1
    return changed


async def async_main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cities_file = (project_root / args.cities_file).resolve()
    output_file = (project_root / args.output).resolve()
    headless = args.headless.lower() == "true"

    cities_payload = load_cities(cities_file)
    all_cities = cities_payload["cities"]

    selected: list[dict[str, Any]] = []
    for row in all_cities:
        if not isinstance(row, dict):
            continue
        if args.all:
            selected.append(row)
        else:
            if not str(row.get("geo_id", "")).strip():
                selected.append(row)

    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    if not selected:
        print("No city to process.")
        return

    print(f"Processing {len(selected)} cities...")
    results = await run_lookup(
        selected,
        headless=headless,
        type_delay_ms=args.type_delay_ms,
        allow_geo_area=args.allow_geo_area,
    )
    save_results(output_file, results)
    print(f"Saved results: {output_file}")

    found = sum(1 for r in results if r.status == "found")
    failed = len(results) - found
    print(f"Summary: found={found}, not_found_or_error={failed}")

    if args.apply:
        changed = apply_geo_ids(cities_payload, results, args.min_apply_score)
        cities_file.write_text(
            yaml.safe_dump(cities_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"Applied geo_id updates to cities file: {changed} (min_score={args.min_apply_score})")


def main() -> None:
    import asyncio

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
