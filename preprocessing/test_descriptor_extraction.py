"""
Test script để kiểm tra việc trích xuất descriptor + ngữ cảnh từ review.
Sử dụng token-based matching để tránh lỗi normalize làm sai lệch từ.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nlp.normalization import normalize_text
from nlp.tokenizer import tokenize_vi
import yaml

# Load rules
RULE_FILE = PROJECT_ROOT / "config" / "review_tag_rules.yaml"
rules = yaml.safe_load(RULE_FILE.read_text(encoding="utf-8"))

# Build descriptor phrase map: phrase_tokens -> (tag, is_negative)
descriptor_phrases = {}
for tag, cfg in rules.get("descriptor_tags", {}).items():
    for phrase in cfg.get("positive", []):
        # Tokenize phrase để matching trên token
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            descriptor_phrases[phrase_tokens] = (tag, False)
    for phrase in cfg.get("negative", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            descriptor_phrases[phrase_tokens] = (tag, True)

# Category phrase map
category_phrases = {}
for tag, cfg in rules.get("category_tags", {}).items():
    for phrase in cfg.get("positive", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            category_phrases[phrase_tokens] = tag
    for phrase in cfg.get("negative", []):
        phrase_tokens = tuple(tokenize_vi(normalize_text(phrase.replace("_", " "))))
        if phrase_tokens:
            category_phrases[phrase_tokens] = tag


def find_phrase_in_tokens(tokens: list[str], phrase_tokens: tuple[str, ...]) -> list[int]:
    """Tìm vị trí bắt đầu của phrase_tokens trong tokens list."""
    positions = []
    phrase_len = len(phrase_tokens)
    for i in range(len(tokens) - phrase_len + 1):
        if tuple(tokens[i:i + phrase_len]) == phrase_tokens:
            positions.append(i)
    return positions


def get_token_window(tokens: list[str], center_idx: int, window: int = 5) -> str:
    """Lấy cửa sổ token quanh vị trí trung tâm."""
    start = max(0, center_idx - window)
    end = min(len(tokens), center_idx + window + 1)
    return " ".join(tokens[start:end])


def extract_descriptor_contexts(text: str, window: int = 5) -> list[dict]:
    """
    Trích xuất descriptor + ngữ cảnh từ review text.
    Sử dụng token-based matching để tránh lỗi normalize.
    """
    # Tokenize text gốc (không normalize để giữ nguyên từ)
    tokens = tokenize_vi(text.lower())
    
    # Tìm tất cả descriptor matches
    matches = []  # (start_token_idx, end_token_idx, tag, is_negative, phrase_str)
    
    for phrase_tokens, (tag, is_negative) in descriptor_phrases.items():
        positions = find_phrase_in_tokens(tokens, phrase_tokens)
        for start_idx in positions:
            end_idx = start_idx + len(phrase_tokens)
            phrase_str = " ".join(tokens[start_idx:end_idx])
            matches.append((start_idx, end_idx, tag, is_negative, phrase_str))
    
    # Sort by position
    matches.sort(key=lambda x: x[0])
    
    # Merge overlapping matches (chỉ khi overlap thực sự)
    merged = []
    for match in matches:
        start, end, tag, is_neg, phrase = match
        if merged and start < merged[-1][1]:  # Chỉ merge khi overlap thực sự
            # Merge: mở rộng end, gộp tag
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), prev[2] + "|" + tag, prev[3] or is_neg, prev[4] + " + " + phrase)
        else:
            merged.append(match)
    
    # Extract context for each match
    results = []
    for start, end, tag, is_neg, phrase in merged:
        center_idx = (start + end) // 2
        context = get_token_window(tokens, center_idx, window)
        
        results.append({
            "descriptor": phrase,
            "tag": tag,
            "polarity": "negative" if is_neg else "positive",
            "context": context,
            "token_range": (start, end),
        })
    
    return results


def extract_category_tags(text: str) -> list[str]:
    """Trích xuất category tags từ review text."""
    tokens = tokenize_vi(text.lower())
    tags = set()
    for phrase_tokens, tag in category_phrases.items():
        if find_phrase_in_tokens(tokens, phrase_tokens):
            tags.add(tag)
    return list(tags)


# Test cases
test_reviews = [
    {
        "review_id": "traveloka_9000005501941_1845131484293252412",
        "review_text": "Ưu điểm: Chú bảo vệ quá nhiệt tình. A lễ tân cũng thoải mái, mình trả phòng lúc 12 rưỡi nhưng ko thu thêm tiền phòng hay tiền xe. Ksan mới\nNhược điểm: Ksan mới nên còn thiếu sót, cửa sổ nhỏ, ko kéo rèm lên được, cả tối ko có nước nóng, tắm lạnh quá kinh khủng, ksan ko có thang máy, vị trí xa trung tâm nên đi đâu cũng bất tiện, nhà xe còn ko biết đường vào và ko đồng ý đón ra."
    },
    {
        "review_id": "test_001",
        "review_text": "Phòng sạch sẽ gọn gàng, nhân viên thân thiện nhiệt tình, nhưng giá hơi cao và xa trung tâm."
    },
    {
        "review_id": "test_002",
        "review_text": "Khách sạn mới tinh, đầy đủ tiện nghi. View xịn miễn bàn. Giá hợp lý. Nhân viên thân thiện, nhiệt tình."
    },
]

print("=" * 80)
print("TEST DESCRIPTOR + CONTEXT EXTRACTION (TOKEN-BASED)")
print("=" * 80)

for review in test_reviews:
    print(f"\n{'='*80}")
    print(f"Review ID: {review['review_id']}")
    print(f"Text: {review['review_text'][:100]}...")
    print(f"{'='*80}")
    
    # Extract category tags
    categories = extract_category_tags(review["review_text"])
    print(f"\n📍 Category tags: {categories}")
    
    # Extract descriptor contexts
    contexts = extract_descriptor_contexts(review["review_text"], window=5)
    print(f"\n🏷️ Descriptor contexts:")
    for ctx in contexts:
        print(f"  - Descriptor: {ctx['descriptor']}")
        print(f"    Tag: {ctx['tag']}")
        print(f"    Polarity: {ctx['polarity']}")
        print(f"    Context: '{ctx['context']}'")
        print(f"    Token range: {ctx['token_range']}")
        print()

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)