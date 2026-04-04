"""
Query Tag Extractor - Dùng FlashText để extract tags từ query
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from flashtext import KeywordProcessor

from nlp.normalization import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULE_FILE = PROJECT_ROOT / "config" / "review_tag_rules.yaml"


@lru_cache(maxsize=1)
def _load_tag_keywords() -> dict[str, list[tuple[str, str]]]:
    """
    Load tag keywords từ review_tag_rules.yaml.
    Returns: {"category": [(keyword, polarity), ...], "descriptor": [...]}
    """
    if not RULE_FILE.exists():
        return {"category": [], "descriptor": []}
    
    payload = yaml.safe_load(RULE_FILE.read_text(encoding="utf-8")) or {}
    
    keywords: dict[str, list[tuple[str, str]]] = {"category": [], "descriptor": []}
    
    # Category tags
    for tag, cfg in payload.get("category_tags", {}).items():
        if not isinstance(cfg, dict):
            continue
        for phrase in cfg.get("positive", []):
            normalized = normalize_text(str(phrase).replace("_", " "))
            if normalized:
                keywords["category"].append((normalized, tag))
        for phrase in cfg.get("negative", []):
            normalized = normalize_text(str(phrase).replace("_", " "))
            if normalized:
                keywords["category"].append((normalized, tag))
    
    # Descriptor tags
    for tag, cfg in payload.get("descriptor_tags", {}).items():
        if not isinstance(cfg, dict):
            continue
        for phrase in cfg.get("positive", []):
            normalized = normalize_text(str(phrase).replace("_", " "))
            if normalized:
                keywords["descriptor"].append((normalized, tag))
        for phrase in cfg.get("negative", []):
            normalized = normalize_text(str(phrase).replace("_", " "))
            if normalized:
                keywords["descriptor"].append((normalized, tag))
    
    return keywords


@lru_cache(maxsize=1)
def _build_keyword_processor() -> KeywordProcessor:
    """Build FlashText processor với tất cả tag keywords."""
    kp = KeywordProcessor(case_sensitive=False)
    keywords = _load_tag_keywords()
    
    for keyword, tag in keywords["category"] + keywords["descriptor"]:
        kp.add_keyword(keyword, tag)
    
    return kp


def extract_query_tags(query: str) -> dict[str, list[str]]:
    """
    Extract tags từ query dùng FlashText.
    
    Returns:
        {
            "category_tags": ["beach", "budget"],
            "descriptor_tags": ["cleanliness", "staff_friendly"],
            "all_tags": ["beach", "budget", "cleanliness", "staff_friendly"]
        }
    """
    kp = _build_keyword_processor()
    keywords_found = kp.extract_keywords(query)
    
    # Load rules để phân loại tag
    keywords = _load_tag_keywords()
    category_tag_names = {tag for _, tag in keywords["category"]}
    descriptor_tag_names = {tag for _, tag in keywords["descriptor"]}
    
    category_tags = []
    descriptor_tags = []
    seen = set()
    
    for tag in keywords_found:
        if tag in seen:
            continue
        seen.add(tag)
        if tag in category_tag_names:
            category_tags.append(tag)
        elif tag in descriptor_tag_names:
            descriptor_tags.append(tag)
    
    return {
        "category_tags": category_tags,
        "descriptor_tags": descriptor_tags,
        "all_tags": category_tags + descriptor_tags,
    }


def get_query_tag_filter(query: str) -> dict | None:
    """
    Tạo MongoDB filter từ query tags.
    Returns None nếu không có tags để filter.
    """
    tags = extract_query_tags(query)
    
    if not tags["all_tags"]:
        return None
    
    # Filter: reviews có ít nhất 1 trong các tags tìm được
    return {"category_tags": {"$in": tags["all_tags"]}}