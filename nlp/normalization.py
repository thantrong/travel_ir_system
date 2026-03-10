import re


PUNCT_RE = re.compile(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩị"
                      r"òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    value = text.lower().strip()
    value = PUNCT_RE.sub(" ", value)
    value = WS_RE.sub(" ", value)
    return value.strip()
