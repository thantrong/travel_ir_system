import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from nlp.normalization import normalize_text
from nlp.stopwords import load_stopwords, remove_stopwords
from nlp.tokenizer import tokenize_vi
from preprocessing.clean_text import clean_review_text
from preprocessing.language_filter import is_vietnamese_text
from preprocessing.remove_spam import is_spam_review
from preprocessing.review_tagger import tag_record


def normalize_id_text(value: str) -> str:
    return str(value or "").strip()


def build_review_id(source: str, source_hotel_id: str, source_review_id: str, review_text: str) -> str:
    source_prefix = normalize_id_text(source).lower()
    hotel_id = normalize_id_text(source_hotel_id)
    src_review_id = normalize_id_text(source_review_id)
    if hotel_id and src_review_id:
        return f"{source_prefix}_{hotel_id}_{src_review_id}"
    if src_review_id:
        return f"{source_prefix}_{src_review_id}"
    digest = hashlib.sha1(review_text.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"{source_prefix}_{hotel_id}_{digest}" if hotel_id else f"{source_prefix}_{digest}"


def parse_rating(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(round(float(text), 1))
    except Exception:
        return ""


def normalize_location_label(value: str, fallback: str = "Vietnam") -> str:
    """Coerce location to city/province label (not full address)."""
    raw = str(value or "").strip()
    if not raw:
        return str(fallback or "Vietnam")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return str(fallback or "Vietnam")

    def _clean_admin_prefix(text: str) -> str:
        t = text.strip()
        lowered = t.lower()
        prefixes = ("tỉnh ", "thành phố ", "tp. ", "tp ")
        for pref in prefixes:
            if lowered.startswith(pref):
                return t[len(pref):].strip()
        return t

    # Prefer explicit administrative labels.
    markers = ("tỉnh", "thành phố", "tp ", "tp.", "province", "city")
    for part in reversed(parts):
        lowered = part.lower()
        if any(m in lowered for m in markers):
            return _clean_admin_prefix(part)

    # Fallback: pick right-most meaningful segment (often province/city in VN addresses).
    for part in reversed(parts):
        lowered = part.lower()
        if lowered in {"việt nam", "vietnam"}:
            continue
        if lowered.isdigit():
            continue
        return _clean_admin_prefix(part)

    return str(fallback or "Vietnam")


def get_field(row: dict, field_spec: Any, default: str = "") -> str:
    if not field_spec:
        return default
    if isinstance(field_spec, list):
        for key in field_spec:
            value = str(row.get(str(key), "")).strip()
            if value:
                return value
        return default
    return str(row.get(str(field_spec), default) or default).strip()


def iter_source_rows(input_path: Path, file_type: str) -> list[dict]:
    values: list[dict] = []
    if file_type == "csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                values.append(row)
        return values

    if file_type == "json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    values.append(row)
        return values

    if file_type == "jsonl":
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                values.append(row)
        return values

    raise ValueError(f"Unsupported file type: {file_type}")


def build_lookup_table(lookup_cfg: dict) -> dict[str, dict]:
    input_path = Path(str(lookup_cfg.get("path", "")).strip()).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Lookup file not found: {input_path}")
    lookup_type = str(lookup_cfg.get("type", "csv")).strip().lower()
    key_field = str(lookup_cfg.get("key_field", "")).strip()
    if not key_field:
        raise ValueError("lookup.key_field is required")

    rows = iter_source_rows(input_path, lookup_type)
    table: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get(key_field, "")).strip()
        if not key:
            continue
        table[key] = row
    return table


def enrich_row_with_lookup(row: dict, lookup_cfg: dict, lookup_table: dict[str, dict]) -> dict:
    source_key_field = str(lookup_cfg.get("source_key_field", "")).strip()
    prefix = str(lookup_cfg.get("prefix", "lookup_"))
    if not source_key_field:
        return row

    source_key = str(row.get(source_key_field, "")).strip()
    if not source_key:
        return row
    lookup_row = lookup_table.get(source_key)
    if not lookup_row:
        return row

    merged = dict(row)
    for key, value in lookup_row.items():
        merged[f"{prefix}{key}"] = value
    return merged


def map_rows_to_unified_schema(
    source_name: str,
    rows: list[dict],
    mapping: dict,
    defaults: dict,
    lookup_cfg: dict | None,
    lookup_table: dict[str, dict] | None,
    limit: int,
) -> list[dict]:
    unified: list[dict] = []
    review_text_field = mapping.get("review_text", "")
    if not review_text_field:
        raise ValueError(f"[{source_name}] mapping.review_text is required")

    for idx, row in enumerate(rows, start=1):
        if limit > 0 and len(unified) >= limit:
            break

        effective_row = row
        if lookup_cfg and lookup_table:
            effective_row = enrich_row_with_lookup(row, lookup_cfg, lookup_table)

        review_text = get_field(effective_row, review_text_field, "")
        if not review_text:
            continue

        source_hotel_id = normalize_id_text(get_field(effective_row, mapping.get("source_hotel_id", ""), ""))
        hotel_name_field = mapping.get("hotel_name", mapping.get("place_name", ""))
        hotel_name_default = defaults.get("hotel_name", defaults.get("place_name", f"{source_name}_dataset"))
        hotel_name = get_field(effective_row, hotel_name_field, hotel_name_default)
        location_raw = get_field(effective_row, mapping.get("location", ""), defaults.get("location", "Vietnam"))
        location = normalize_location_label(location_raw, defaults.get("location", "Vietnam"))
        rating = parse_rating(get_field(effective_row, mapping.get("review_rating", ""), ""))
        review_date = get_field(effective_row, mapping.get("review_date", ""), "")
        source_review_id = get_field(effective_row, mapping.get("source_review_id", ""), str(idx))

        if not source_hotel_id:
            source_hotel_id = hashlib.sha1(
                f"{source_name}|{hotel_name.strip().lower()}|{location.strip().lower()}".encode("utf-8")
            ).hexdigest()[:12]
        review_id = build_review_id(source_name, source_hotel_id, source_review_id, review_text)
        record = {
            "review_id": review_id,
            "source_review_id": source_review_id,
            "source_hotel_id": source_hotel_id,
            "hotel_name": hotel_name,
            "location": location,
            "rating": "",
            "review_text": review_text,
            "review_date": review_date,
            "review_rating": rating,
            "source": source_name,
        }
        unified.append(record)
    return unified


def apply_place_average_rating(records: list[dict]) -> list[dict]:
    """Fill place-level rating by averaging review_rating within each source_hotel_id."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in records:
        source_hotel_id = str(row.get("source_hotel_id", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        group_key = f"{source}|{source_hotel_id}"
        if not source_hotel_id:
            continue
        try:
            score = float(str(row.get("review_rating", "")).strip())
        except Exception:
            continue
        sums[group_key] = sums.get(group_key, 0.0) + score
        counts[group_key] = counts.get(group_key, 0) + 1

    avg_map: dict[str, str] = {}
    for group_key, total in sums.items():
        count = counts.get(group_key, 0)
        if count > 0:
            avg_map[group_key] = str(round(total / count, 2))

    for row in records:
        source_hotel_id = str(row.get("source_hotel_id", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        group_key = f"{source}|{source_hotel_id}"
        row["rating"] = avg_map.get(group_key, "")
    return records


def process_records(
    records: list[dict],
    stopwords_path: Path,
    only_vietnamese: bool,
    source_vi_filter: dict[str, bool] | None = None,
) -> list[dict]:
    stopwords = load_stopwords(stopwords_path)
    output = []
    for row in records:
        text = clean_review_text(row.get("review_text", ""))
        if not text or is_spam_review(text):
            continue

        source_name = str(row.get("source", "")).strip().lower()
        enforce_vi = only_vietnamese
        if not enforce_vi and source_vi_filter:
            enforce_vi = bool(source_vi_filter.get(source_name, False))

        if enforce_vi and not is_vietnamese_text(text):
            continue

        clean = normalize_text(text)
        tokens = remove_stopwords(tokenize_vi(clean), stopwords)
        if not tokens:
            continue

        transformed = dict(row)
        transformed["clean_text"] = clean
        transformed["tokens"] = tokens
        transformed = tag_record(transformed)
        output.append(transformed)
    return output


def load_sources_from_config(config_path: Path, only_source: str) -> list[dict]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        return []
    selected = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        if source.get("enabled", True) is False:
            continue
        if only_source and str(source.get("name", "")).strip().lower() != only_source.lower():
            continue
        selected.append(source)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Unified dataset ingestion pipeline")
    parser.add_argument(
        "--sources-config",
        default="config/dataset_sources.yaml",
        help="Path to dataset sources config YAML",
    )
    parser.add_argument("--source-name", default="", help="Process only one source by name")
    parser.add_argument("--limit-per-source", type=int, default=0, help="Read at most N rows/source (0 = all)")
    parser.add_argument("--only-vietnamese", action="store_true", help="Keep only Vietnamese reviews")
    parser.add_argument("--load-mongo", action="store_true", help="Load processed records into MongoDB")
    parser.add_argument(
        "--raw-output",
        default="data/raw/dataset_raw_unified.json",
        help="Output path for unified raw records",
    )
    parser.add_argument(
        "--processed-output",
        default="data/processed/dataset_processed_unified.json",
        help="Output path for processed records",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sources_config_path = (project_root / args.sources_config).resolve()
    if not sources_config_path.exists():
        raise FileNotFoundError(f"Sources config not found: {sources_config_path}")

    stopwords_path = project_root / "config" / "stopwords.txt"
    raw_output = (project_root / args.raw_output).resolve()
    processed_output = (project_root / args.processed_output).resolve()
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    processed_output.parent.mkdir(parents=True, exist_ok=True)

    sources = load_sources_from_config(sources_config_path, args.source_name)
    if not sources:
        raise ValueError("No enabled sources found in dataset config")

    unified_records: list[dict] = []
    source_vi_filter: dict[str, bool] = {}
    for source in sources:
        source_name = str(source.get("name", "")).strip().lower()
        input_path = Path(str(source.get("input_path", "")).strip()).expanduser().resolve()
        file_type = str(source.get("type", "csv")).strip().lower()
        mapping = source.get("mapping", {}) if isinstance(source.get("mapping", {}), dict) else {}
        defaults = source.get("defaults", {}) if isinstance(source.get("defaults", {}), dict) else {}
        lookup_cfg = source.get("lookup", {}) if isinstance(source.get("lookup", {}), dict) else {}
        lookup_table = None
        if not source_name or not input_path.exists():
            print(f"Skip source invalid: name={source_name}, input_path={input_path}")
            continue
        source_vi_filter[source_name] = bool(source.get("only_vietnamese", False))

        if lookup_cfg:
            try:
                lookup_table = build_lookup_table(lookup_cfg)
                print(f"Loaded lookup for '{source_name}': {len(lookup_table)} rows")
            except Exception as exc:
                print(f"Lookup disabled for '{source_name}' due to error: {exc}")
                lookup_cfg = {}
                lookup_table = None

        rows = iter_source_rows(input_path, file_type)
        mapped = map_rows_to_unified_schema(
            source_name=source_name,
            rows=rows,
            mapping=mapping,
            defaults=defaults,
            lookup_cfg=lookup_cfg if lookup_cfg else None,
            lookup_table=lookup_table,
            limit=args.limit_per_source,
        )
        unified_records.extend(mapped)
        print(f"Loaded source '{source_name}': raw_rows={len(rows)}, mapped_records={len(mapped)}")

    unified_records = apply_place_average_rating(unified_records)
    raw_output.write_text(json.dumps(unified_records, ensure_ascii=False, indent=2), encoding="utf-8")

    processed_records = process_records(
        unified_records,
        stopwords_path=stopwords_path,
        only_vietnamese=args.only_vietnamese,
        source_vi_filter=source_vi_filter,
    )
    processed_output.write_text(json.dumps(processed_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Unified raw records: {len(unified_records)}")
    print(f"Processed records: {len(processed_records)}")
    print(f"Saved raw: {raw_output}")
    print(f"Saved processed: {processed_output}")

    if args.load_mongo:
        from database.data_loader import load_reviews

        place_count, review_count = load_reviews(processed_records)
        print(f"Mongo upserted places: {place_count}, reviews: {review_count}")


if __name__ == "__main__":
    main()
