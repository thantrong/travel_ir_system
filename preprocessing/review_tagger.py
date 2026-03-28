from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi
from preprocessing.clean_text import clean_review_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULE_FILE = PROJECT_ROOT / "config" / "review_tag_rules.yaml"


@dataclass
class TagResult:
    category_tags: list[str] = field(default_factory=list)
    descriptor_tags: list[str] = field(default_factory=list)
    matched_phrases: dict[str, list[str]] = field(default_factory=dict)


def _load_rules() -> dict:
    if not RULE_FILE.exists():
        return {}
    payload = yaml.safe_load(RULE_FILE.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def rules() -> dict:
    return _load_rules()


def _normalize_phrases(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for v in values:
        nv = normalize_text(str(v).replace("_", " "))
        if nv and nv not in seen:
            seen.add(nv)
            out.append(nv)
    return out


def _prepare_text(review_text: str, hotel_name: str = "", location: str = "") -> str:
    parts = [clean_review_text(review_text), hotel_name, location]
    joined = " ".join(p for p in parts if p)
    return normalize_text(joined)


def _score_rule(text: str, positive: list[str], negative: list[str]) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for phrase in positive:
        if phrase and phrase in text:
            score += 1
            matched.append(phrase)
    for phrase in negative:
        if phrase and phrase in text:
            score -= 1
            matched.append(f"!{phrase}")
    return score, matched


def _apply_tag_group(text: str, spec: dict[str, dict]) -> tuple[list[str], dict[str, list[str]]]:
    tags: list[str] = []
    matched_map: dict[str, list[str]] = {}
    for tag, cfg in spec.items():
        if not isinstance(cfg, dict):
            continue
        positive = _normalize_phrases(cfg.get("positive", []))
        negative = _normalize_phrases(cfg.get("negative", []))
        score, matched = _score_rule(text, positive, negative)
        if score > 0:
            tags.append(tag)
            matched_map[tag] = matched
    return tags, matched_map


def tag_review(review_text: str, hotel_name: str = "", location: str = "") -> TagResult:
    """Rule-based tagging for a single review.

    Designed for phase-1 enrichment before MongoDB/indexing.
    """
    payload = rules()
    text = _prepare_text(review_text, hotel_name, location)
    if not text:
        return TagResult()

    category_rules = payload.get("category_tags", {})
    descriptor_rules = payload.get("descriptor_tags", {})
    category_tags, category_matches = _apply_tag_group(text, category_rules)
    descriptor_tags, descriptor_matches = _apply_tag_group(text, descriptor_rules)

    return TagResult(
        category_tags=category_tags,
        descriptor_tags=descriptor_tags,
        matched_phrases={
            "category_tags": category_matches,
            "descriptor_tags": descriptor_matches,
        },
    )


def tag_record(record: dict) -> dict:
    review_text = str(record.get("review_text", ""))
    hotel_name = str(record.get("hotel_name", ""))
    location = str(record.get("location", ""))
    res = tag_review(review_text, hotel_name=hotel_name, location=location)

    tagged = dict(record)
    tagged["category_tags"] = res.category_tags
    tagged["descriptor_tags"] = res.descriptor_tags
    tagged["matched_phrases"] = res.matched_phrases
    return tagged
