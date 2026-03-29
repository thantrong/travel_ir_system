"""
Query Understanding Layer cho phiên bản Hotel-level.
Refactor:
- Stopwords chung chỉ áp dụng cho review preprocessing.
- Query stopwords tách riêng ra file config/query_stopwords.txt.
- Domain lexicon, synonyms, location aliases và descriptor terms tách ra file config/*.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Iterable

import yaml

from nlp.normalization import normalize_text
from nlp.stopwords import load_stopwords, remove_stopwords
from nlp.tokenizer import tokenize_vi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
QUERY_STOPWORDS_FILE = CONFIG_DIR / "query_stopwords.txt"
QUERY_SYNONYMS_FILE = CONFIG_DIR / "query_synonyms.yaml"
LOCATION_ALIASES_FILE = CONFIG_DIR / "location_aliases.yaml"
DESCRIPTOR_TERMS_FILE = CONFIG_DIR / "descriptor_terms.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _load_term_set(path: Path, key: str) -> frozenset[str]:
    payload = _load_yaml(path)
    values = payload.get(key, [])
    if not isinstance(values, list):
        return frozenset()
    return frozenset(str(v).strip().lower() for v in values if str(v).strip())


def _load_synonym_map(path: Path) -> dict[str, list[str]]:
    payload = _load_yaml(path)
    mapping = payload.get("synonyms", {})
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, values in mapping.items():
        if not isinstance(values, list):
            continue
        normalized = [str(v).strip() for v in values if str(v).strip()]
        if normalized:
            out[str(key).strip().lower()] = normalized
    return out


def _load_location_aliases(path: Path) -> dict[str, str]:
    payload = _load_yaml(path)
    locations = payload.get("locations", {})
    if not isinstance(locations, dict):
        return {}
    out: dict[str, str] = {}
    for canonical, aliases in locations.items():
        canonical_key = str(canonical).strip().lower()
        if not canonical_key:
            continue
        out[canonical_key] = canonical_key
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            alias_key = str(alias).strip().lower().replace(" ", "_")
            if alias_key:
                out[alias_key] = canonical_key
    return out


def _load_location_keywords(path: Path) -> dict[str, list[str]]:
    payload = _load_yaml(path)
    locations = payload.get("locations", {})
    if not isinstance(locations, dict):
        return {}
    out: dict[str, list[str]] = {}
    for canonical, aliases in locations.items():
        canonical_key = str(canonical).strip().lower()
        kws = [canonical_key.replace("_", " ")]
        if isinstance(aliases, list):
            kws.extend(str(a).strip().lower() for a in aliases if str(a).strip())
        out[canonical_key] = list(dict.fromkeys(kws))
    return out


@lru_cache(maxsize=1)
def query_stop_tokens() -> frozenset[str]:
    if not QUERY_STOPWORDS_FILE.exists():
        return frozenset()
    return frozenset(load_stopwords(QUERY_STOPWORDS_FILE))


@lru_cache(maxsize=1)
def descriptor_tokens() -> frozenset[str]:
    return _load_term_set(DESCRIPTOR_TERMS_FILE, "descriptors") if DESCRIPTOR_TERMS_FILE.exists() else frozenset()


@lru_cache(maxsize=1)
def synonym_map() -> dict[str, list[str]]:
    return _load_synonym_map(QUERY_SYNONYMS_FILE)


@lru_cache(maxsize=1)
def location_aliases() -> dict[str, str]:
    return _load_location_aliases(LOCATION_ALIASES_FILE)


@lru_cache(maxsize=1)
def location_keywords() -> dict[str, list[str]]:
    return _load_location_keywords(LOCATION_ALIASES_FILE)


# Category hints used for query-time routing/filtering.
CATEGORY_TOKEN_HINTS: dict[str, set[str]] = {
    "mountain": {"núi", "đồi", "thung_lũng", "cao_nguyên", "săn_mây", "rừng", "view_núi", "hướng_núi"},
    "beach": {"biển", "bãi_biển", "ven_biển", "gần_biển", "đảo", "view_biển"},
    "family": {"gia_đình", "trẻ_em", "kid", "family", "phòng_gia_đình"},
    "budget": {"giá_rẻ", "bình_dân", "tiết_kiệm", "hợp_lý", "rẻ"},
    "luxury": {"sang_trọng", "cao_cấp", "luxury", "5_sao", "resort", "villa"},
    "center": {"trung_tâm", "phố_đi_bộ", "chợ_đêm", "gần_chợ"},
    "amenity_pool": {"hồ_bơi", "bể_bơi", "pool", "vô_cực"},
    "amenity_breakfast": {"ăn_sáng", "buffet", "điểm_tâm"},
    "airport": {"sân_bay", "airport", "đưa_đón", "xe_đón", "gần_ga", "ga_tàu", "metro"},
    "spa_gym": {"spa", "massage", "gym", "fitness", "xông_hơi", "jacuzzi"},
    "photo": {"sống_ảo", "checkin", "check_in", "chụp_hình", "vintage", "view_đẹp"},
    "quiet": {"yên_tĩnh", "nghỉ_dưỡng", "thư_giãn", "đọc_sách", "riêng_tư"},
    "pet": {"pet", "pet_friendly", "thú_cưng", "đưa_thú_cưng"},
    "kitchen": {"bếp_nấu", "bếp_riêng", "bbq", "nướng", "kitchen"},
    "business": {"công_tác", "business", "workation", "phòng_họp", "meeting", "co_working"},
}


def _token_forms(tok: str) -> set[str]:
    out: set[str] = set()
    low = (tok or "").lower().strip()
    if not low:
        return out
    out.add(low)
    if "_" in low:
        out.update(p for p in low.split("_") if p)
    return out


@dataclass
class QueryUnderstandingResult:
    raw_query: str = ""
    core_tokens: list[str] = field(default_factory=list)
    detected_location: str = ""
    descriptor_tokens: list[str] = field(default_factory=list)
    detected_categories: list[str] = field(default_factory=list)
    expanded_tokens: list[str] = field(default_factory=list)


def understand_query(raw_query: str, stopwords_path: Path | None = None) -> QueryUnderstandingResult:
    """Xử lý language, semantic extraction cho search engine."""
    res = QueryUnderstandingResult(raw_query=raw_query)
    if not raw_query or not raw_query.strip():
        return res

    normalized = normalize_text(raw_query)
    tokens = tokenize_vi(normalized)

    # Stopwords review-level: chỉ dùng khi caller truyền vào explicit file.
    if stopwords_path and stopwords_path.exists():
        stopwords = load_stopwords(stopwords_path)
        tokens = remove_stopwords(tokens, stopwords)

    # Query stopwords riêng cho câu truy vấn.
    qstops = query_stop_tokens()
    core = [tok for tok in tokens if tok.lower() not in qstops]

    detected_loc = ""
    loc_remove: set[str] = set()

    # 1) Trigrams
    for i in range(len(core) - 2):
        tg = (core[i].lower(), core[i + 1].lower(), core[i + 2].lower())
        joined = "_".join(tg)
        if joined in location_aliases():
            detected_loc = location_aliases()[joined]
            loc_remove.update({core[i], core[i + 1], core[i + 2]})
            break

    # 2) Bigrams
    if not detected_loc:
        for i in range(len(core) - 1):
            bg = (core[i].lower(), core[i + 1].lower())
            joined = "_".join(bg)
            if joined in location_aliases():
                detected_loc = location_aliases()[joined]
                loc_remove.update({core[i], core[i + 1]})
                break

    # 3) Unigrams / aliases
    if not detected_loc:
        loc_map = location_aliases()
        for t in core:
            key = t.lower().replace(" ", "_")
            if key in loc_map:
                detected_loc = loc_map[key]
                loc_remove.add(t)
                break

    res.detected_location = detected_loc
    core = [t for t in core if t not in loc_remove]
    res.core_tokens = core

    desc_set = descriptor_tokens()
    res.descriptor_tokens = [t for t in core if any(form in desc_set for form in _token_forms(t))]

    syn_map = synonym_map()
    expanded: list[str] = []
    seen: set[str] = set()
    for t in core:
        syns = syn_map.get(t.lower(), [t])
        for syn in syns:
            if syn not in seen:
                seen.add(syn)
                expanded.append(syn)

    if detected_loc and detected_loc not in seen:
        expanded.append(detected_loc)

    detected_categories: list[str] = []
    query_forms: set[str] = set()
    for tok in core + expanded:
        query_forms.update(_token_forms(tok))

    for cat, hints in CATEGORY_TOKEN_HINTS.items():
        if query_forms.intersection(hints):
            detected_categories.append(cat)

    res.detected_categories = detected_categories
    res.expanded_tokens = expanded
    return res


LOCATION_KEYWORDS = location_keywords()


def location_matched(detected_location: str, hotel_location: str) -> bool:
    if not detected_location or not hotel_location:
        return False
    kws = LOCATION_KEYWORDS.get(detected_location, [detected_location.replace("_", " ")])
    hloc_lower = str(hotel_location).lower()
    return any(kw.lower() in hloc_lower for kw in kws)
