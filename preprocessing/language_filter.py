VIET_CHARS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩị"
    "òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)

COMMON_VI_WORDS = (
    "khách sạn",
    "nhân viên",
    "phòng",
    "đẹp",
    "sạch",
    "thuận tiện",
    "đà nẵng",
    "phú quốc",
)


def is_vietnamese_text(text: str) -> bool:
    if not text:
        return False
    value = str(text)
    if any(ch in VIET_CHARS for ch in value):
        return True
    lowered = value.lower()
    return any(token in lowered for token in COMMON_VI_WORDS)
