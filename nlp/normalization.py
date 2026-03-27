import re


PUNCT_RE = re.compile(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩị"
                      r"òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
# Nén các từ có ký tự lặp kéo dài: ngonnnn -> ngon, đẹpppp -> đẹp
CHAR_REPEAT_RE = re.compile(r"([a-zà-ỹđ])\1{2,}", re.IGNORECASE)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    value = text.lower().strip()
    value = PUNCT_RE.sub(" ", value)
    value = CHAR_REPEAT_RE.sub(r"\1", value)
    value = WS_RE.sub(" ", value)
    return value.strip()
