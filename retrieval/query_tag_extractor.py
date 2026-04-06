"""
Query Tag Extractor - tạm thời vô hiệu hóa FlashText tagging/filtering.

Giữ API tương thích để hệ thống không vỡ, nhưng không còn tạo tag/filter
để tránh mất thông tin khi search.
"""

from __future__ import annotations


def extract_query_tags(query: str) -> dict[str, list[str]]:
    return {
        "category_tags": [],
        "descriptor_tags": [],
        "all_tags": [],
    }


def get_query_tag_filter(query: str) -> dict | None:
    return None
