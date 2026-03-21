"""
Query Understanding Layer cho phiên bản Hotel-level
TASK 3: Query Understanding module
TASK 4: Query Stopwords mở rộng
TASK 5: Query Expansion (synonyms)
TASK 6: Location Detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nlp.normalization import normalize_text
from nlp.stopwords import load_stopwords, remove_stopwords
from nlp.tokenizer import tokenize_vi


# TASK 4: Query Stopwords
QUERY_STOP_TOKENS: frozenset[str] = frozenset([
    "tôi", "mình", "chúng_tôi", "chúng_mình", "ta",
    "muốn", "cần", "tìm", "kiếm", "tìm_kiếm",
    "xin", "hãy", "giúp", "cho", "bạn", "ơi",
    "ở", "tại", "vào", "ra", "và", "hoặc", "hay", "nào",
    "có", "không", "nhé", "thì", "với", "để", "mà",
    "về", "thuộc", "trong", "đây", "đó",
    "nào", "gì", "đâu", "sao", "thế_nào", "bao_nhiêu",
    "tốt", "đẹp", "phù_hợp", "thích_hợp", "hợp",
    "thành", "phố", "thành_phố", "khu", "vực", "khu_vực",
    "trung", "tâm", "nơi", "chỗ", "vùng", "miền",
])

# TASK 6: Location Detection
_LOCATION_ALIASES: dict[str, str] = {
    "phú_quốc": "phú_quốc", "phu_quoc": "phú_quốc",
    "đà_nẵng": "đà_nẵng", "da_nang": "đà_nẵng",
    "nha_trang": "nha_trang", "nhatrang": "nha_trang",
    "hạ_long": "hạ_long", "ha_long": "hạ_long", "halong": "hạ_long",
    "vũng_tàu": "vũng_tàu", "vung_tau": "vũng_tàu",
    "hội_an": "hội_an", "hoi_an": "hội_an",
    "sa_pa": "sapa", "sapa": "sapa",
    "đà_lạt": "đà_lạt", "da_lat": "đà_lạt",
    "hcm": "hồ_chí_minh", "hồ_chí_minh": "hồ_chí_minh",
    "saigon": "hồ_chí_minh", "sài_gòn": "hồ_chí_minh",
    "hà_nội": "hà_nội", "hanoi": "hà_nội",
    "cần_thơ": "cần_thơ", "can_tho": "cần_thơ",
    "huế": "huế", "hue": "huế",
    "quy_nhơn": "quy_nhơn", "quynhon": "quy_nhơn",
    "phan_thiết": "phan_thiết", "phanthiet": "phan_thiết",
    "mũi_né": "mũi_né", "muine": "mũi_né",
    "ninh_binh": "ninh_bình", "ninh_bình": "ninh_bình",
    "buôn_ma_thuột": "buôn_ma_thuột", "bmt": "buôn_ma_thuột",
    "pleiku": "pleiku", "kon_tum": "kon_tum",
    "hải_phòng": "hải_phòng", "haiphong": "hải_phòng",
    "quảng_bình": "quảng_bình", "phong_nha": "phong_nha",
    "đông_hà": "đông_hà", "quảng_trị": "quảng_trị",
    "tuyên_quang": "tuyên_quang", "tuyen_quang": "tuyên_quang",
    "cao_bằng": "cao_bằng", "cao_bang": "cao_bằng",
    "lai_châu": "lai_châu", "lai_chau": "lai_châu",
    "lào_cai": "lào_cai", "lao_cai": "lào_cai",
    "thái_nguyên": "thái_nguyên", "thai_nguyen": "thái_nguyên",
    "điện_biên": "điện_biên", "dien_bien": "điện_biên",
    "lạng_sơn": "lạng_sơn", "lang_son": "lạng_sơn",
    "sơn_la": "sơn_la", "son_la": "sơn_la",
    "phú_thọ": "phú_thọ", "phu_tho": "phú_thọ",
    "bắc_ninh": "bắc_ninh", "bac_ninh": "bắc_ninh",
    "quảng_ninh": "quảng_ninh", "quang_ninh": "quảng_ninh",
    "hưng_yên": "hưng_yên", "hung_yen": "hưng_yên",
    "thanh_hóa": "thanh_hóa", "thanh_hoa": "thanh_hóa",
    "nghệ_an": "nghệ_an", "nghe_an": "nghệ_an",
    "hà_tĩnh": "hà_tĩnh", "ha_tinh": "hà_tĩnh",
    "quảng_ngãi": "quảng_ngãi", "quang_ngai": "quảng_ngãi",
    "gia_lai": "gia_lai", "gialai": "gia_lai",
    "đắk_lắk": "đắk_lắk", "dak_lak": "đắk_lắk",
    "khánh_hòa": "khánh_hòa", "khanh_hoa": "khánh_hòa",
    "lâm_đồng": "lâm_đồng", "lam_dong": "lâm_đồng",
    "đồng_nai": "đồng_nai", "dong_nai": "đồng_nai",
    "đồng_tháp": "đồng_tháp", "dong_thap": "đồng_tháp",
    "an_giang": "an_giang", "angiang": "an_giang",
    "vĩnh_long": "vĩnh_long", "vinh_long": "vĩnh_long",
    "cà_mau": "cà_mau", "ca_mau": "cà_mau",
    "bình_thuận": "bình_thuận", "binh_thuan": "bình_thuận",
    "phú_yên": "phú_yên", "phu_yen": "phú_yên",
    "tp_hồ_chí_minh": "hồ_chí_minh", "tp_hcm": "hồ_chí_minh", "tphcm": "hồ_chí_minh",
    "tp_hà_nội": "hà_nội", "tphn": "hà_nội",
    "tp_hải_phòng": "hải_phòng",
    "tp_cần_thơ": "cần_thơ", "tp_cần thơ": "cần_thơ",
    "bà_rịa_vũng_tàu": "vũng_tàu", "ba_ria_vung_tau": "vũng_tàu",
    "kiên_giang": "phú_quốc", "kien_giang": "phú_quốc",
}

_LOCATION_BIGRAMS: dict[tuple[str, str], str] = {
    ("phú", "quốc"): "phú_quốc",
    ("đà", "nẵng"): "đà_nẵng",
    ("nha", "trang"): "nha_trang",
    ("hạ", "long"): "hạ_long",
    ("vũng", "tàu"): "vũng_tàu",
    ("hội", "an"): "hội_an",
    ("sa", "pa"): "sapa",
    ("đà", "lạt"): "đà_lạt",
    ("hà", "nội"): "hà_nội",
    ("cần", "thơ"): "cần_thơ",
    ("sài", "gòn"): "hồ_chí_minh",
    ("hồ", "chí"): "hồ_chí_minh",
    ("mũi", "né"): "mũi_né",
    ("quy", "nhơn"): "quy_nhơn",
    ("phan", "thiết"): "phan_thiết",
    ("ninh", "bình"): "ninh_bình",
    ("buôn", "ma"): "buôn_ma_thuột",
    ("hải", "phòng"): "hải_phòng",
    ("quảng", "bình"): "quảng_bình",
    ("quảng", "ninh"): "quảng_ninh",
    ("bắc", "ninh"): "bắc_ninh",
    ("quảng_nam"): "quảng_nam",
    ("phong", "nha"): "phong_nha",
    ("bảo", "lộc"): "bảo_lộc",
    ("tây", "ninh"): "tây_ninh",
    ("tuyên", "quang"): "tuyên_quang",
    ("cao", "bằng"): "cao_bằng",
    ("lai", "châu"): "lai_châu",
    ("lào", "cai"): "lào_cai",
    ("thái", "nguyên"): "thái_nguyên",
    ("điện", "biên"): "điện_biên",
    ("lạng", "sơn"): "lạng_sơn",
    ("sơn", "la"): "sơn_la",
    ("phú", "thọ"): "phú_thọ",
    ("hưng", "yên"): "hưng_yên",
    ("thanh", "hóa"): "thanh_hóa",
    ("nghệ", "an"): "nghệ_an",
    ("hà", "tĩnh"): "hà_tĩnh",
    ("quảng", "ngãi"): "quảng_ngãi",
    ("gia", "lai"): "gia_lai",
    ("đắk", "lắk"): "đắk_lắk",
    ("khánh", "hòa"): "khánh_hòa",
    ("lâm", "đồng"): "lâm_đồng",
    ("đồng", "nai"): "đồng_nai",
    ("đồng", "tháp"): "đồng_tháp",
    ("an", "giang"): "an_giang",
    ("vĩnh", "long"): "vĩnh_long",
    ("cà", "mau"): "cà_mau",
    ("bình", "thuận"): "bình_thuận",
    ("phú", "yên"): "phú_yên",
    ("kiên", "giang"): "phú_quốc",
    ("bà", "rịa"): "vũng_tàu",
}

_LOCATION_TRIGRAMS: dict[tuple[str, str, str], str] = {
    ("hồ", "chí", "minh"): "hồ_chí_minh",
    ("buôn", "ma", "thuột"): "buôn_ma_thuột",
    ("phan", "rang", "tháp"): "phan_rang",
    ("tp", "hồ", "chí"): "hồ_chí_minh",
    ("tp", "hà", "nội"): "hà_nội",
    ("tp", "hải", "phòng"): "hải_phòng",
    ("tp", "cần", "thơ"): "cần_thơ",
    ("bà", "rịa", "vũng"): "vũng_tàu",
}

LOCATION_KEYWORDS: dict[str, list[str]] = {
    "phú_quốc": ["phú quốc", "phu quoc", "kiên giang"],
    "đà_nẵng":  ["đà nẵng", "da nang"],
    "nha_trang": ["nha trang", "khánh hòa"],
    "hạ_long":  ["hạ long", "ha long", "quảng ninh"],
    "vũng_tàu": ["vũng tàu", "vung tau", "bà rịa"],
    "hội_an":   ["hội an", "hoi an", "quảng nam"],
    "sapa":     ["sa pa", "sapa", "lào cai"],
    "đà_lạt":   ["đà lạt", "da lat", "lâm đồng"],
    "hồ_chí_minh": ["hồ chí minh", "sài gòn", "hcm", "saigon", "thành phố hồ", "quận 1"],
    "hà_nội":   ["hà nội", "ha noi"],
    "cần_thơ":  ["cần thơ", "can_tho"],
    "huế":      ["huế", "thừa thiên huế"],
    "quy_nhơn": ["quy nhơn", "quynhon", "bình định"],
    "phan_thiết": ["phan thiết", "bình thuận", "phanthiet"],
    "mũi_né":   ["mũi né", "muine"],
    "ninh_bình": ["ninh bình", "tràng an", "tam cốc"],
    "quảng_bình": ["quảng bình", "đồng hới"],
    "phong_nha": ["phong nha", "kẻ bàng"],
    "hải_phòng": ["hải phòng", "haiphong", "cát bà"],
    "buôn_ma_thuột": ["buôn ma thuột", "đắk lắk", "đắc lắc"],
    "tây_ninh": ["tây ninh", "núi bà đen"],
    "tuyên_quang": ["tuyên quang", "tuyen quang"],
    "cao_bằng": ["cao bằng", "cao bang"],
    "lai_châu": ["lai châu", "lai chau"],
    "lào_cai": ["lào cai", "lao cai"],
    "thái_nguyên": ["thái nguyên", "thai nguyen"],
    "điện_biên": ["điện biên", "dien bien", "mường thanh", "muong thanh"],
    "lạng_sơn": ["lạng sơn", "lang son"],
    "sơn_la": ["sơn la", "son la", "mộc châu", "moc chau"],
    "phú_thọ": ["phú thọ", "phu tho"],
    "bắc_ninh": ["bắc ninh", "bac ninh"],
    "quảng_ninh": ["quảng ninh", "quang ninh", "hạ long", "ha long"],
    "hưng_yên": ["hưng yên", "hung yen"],
    "thanh_hóa": ["thanh hóa", "thanh hoa", "sầm sơn", "sam son"],
    "nghệ_an": ["nghệ an", "nghe an", "cửa lò", "cua lo"],
    "hà_tĩnh": ["hà tĩnh", "ha tinh"],
    "quảng_ngãi": ["quảng ngãi", "quang ngai", "lý sơn", "ly son"],
    "gia_lai": ["gia lai", "pleiku"],
    "đắk_lắk": ["đắk lắk", "dak lak", "buôn ma thuột", "bmt"],
    "khánh_hòa": ["khánh hòa", "khanh hoa", "nha trang", "cam ranh"],
    "lâm_đồng": ["lâm đồng", "lam dong", "đà lạt", "bảo lộc"],
    "đồng_nai": ["đồng nai", "dong nai", "biên hòa", "bien hoa"],
    "đồng_tháp": ["đồng tháp", "dong thap", "sa đéc", "sa dec"],
    "an_giang": ["an giang", "châu đốc", "chau doc", "núi cấm"],
    "vĩnh_long": ["vĩnh long", "vinh long"],
    "cà_mau": ["cà mau", "ca mau", "đất mũi", "dat mui"],
    "bình_thuận": ["bình thuận", "binh thuan", "phan thiết", "mũi né"],
    "phú_yên": ["phú yên", "phu yen", "tuy hòa", "tuy hoa"],
}

# KEYWORDS MÔ TẢ TÌM KIẾM
DESCRIPTOR_TOKENS: frozenset[str] = frozenset([
    "view", "nhìn", "hướng", "cảnh", "panorama",
    "view_núi", "view_đồi", "hướng_núi", "hướng_đồi",
    "núi", "đồi", "đồi_núi", "núi_rừng", "sườn_đồi", "thung_lũng", "cao_nguyên",
    "săn_mây", "mây", "rừng", "đồi_thông", "rừng_thông",
    "biển", "bãi_biển", "gần_biển", "bãi",
    "hồ_bơi", "hồ", "bể_bơi", "vô_cực",
    "resort", "villa", "bungalow", "homestay", "căn_hộ", "boutique", "luxury",
    "sang_trọng", "cao_cấp", "sang", "đẳng_cấp", "5_sao", "4_sao",
    "sạch_sẽ", "đẹp", "hiện_đại", "tiện_nghi", "thoải_mái",
    "spa", "gym", "nhà_hàng", "buffet", "bar", "massage", "sauna", "fitness",
    "ăn_sáng", "cà_phê", "đưa_đón", "sân_bay", "thuê_xe", "giặt_ủi",
    "giá_rẻ", "bình_dân", "tiết_kiệm", "giá_tốt", "hợp_lý",
    "wifi", "điều_hòa", "máy_lạnh", "tủ_lạnh", "tivi", "ban_công", "bồn_tắm",
    "gia_đình", "trẻ_em", "cặp_đôi", "lãng_mạn", "honey_moon", "yên_tĩnh",
    "trung_tâm", "phố_đi_bộ", "chợ_đêm", "gần_chợ",
    "checkin", "check_in", "sống_ảo", "view_đẹp", "chụp_hình",
    "co_working", "workation", "business", "công_tác", "phòng_họp", "meeting",
    "glamping", "camping", "farmstay", "eco", "sinh_thái", "xanh",
    "all_inclusive", "miễn_phí_hủy", "free_cancel", "combo",
    "gần_ga", "ga_tàu", "bến_xe", "metro", "tàu_điện",
    "kid_club", "sân_chơi_trẻ_em", "công_viên_nước",
    "pet_friendly", "thú_cưng", "pet", "đưa_thú_cưng",
    "bbq", "nướng", "bếp_nấu", "bếp_riêng", "ăn_tối",
    "ban_công_view_biển", "sunset", "bình_minh", "hoàng_hôn",
])

# TASK 5: Query Expansion
SYNONYM_MAP: dict[str, list[str]] = {
    "view": ["view", "nhìn", "hướng", "tầm_nhìn", "cảnh"],
    "view_núi": ["view_núi", "hướng_núi", "núi", "đồi", "thung_lũng", "cao_nguyên"],
    "núi": ["núi", "đồi", "đồi_núi", "núi_rừng", "sườn_đồi", "thung_lũng", "cao_nguyên"],
    "biển": ["biển", "bãi_biển", "gần_biển", "ven_biển", "bờ_biển"],
    "khách_sạn": ["khách_sạn", "hotel", "resort", "villa", "homestay"],
    "resort": ["resort", "khách_sạn", "villa", "khu_nghỉ_dưỡng"],
    "sang_trọng": ["sang_trọng", "cao_cấp", "luxury", "5_sao", "đẳng_cấp"],
    "sạch_sẽ": ["sạch_sẽ", "sạch", "gọn_gàng", "ngăn_nắp"],
    "giá_rẻ": ["giá_rẻ", "bình_dân", "tiết_kiệm", "hợp_lý", "rẻ"],
    "nhân_viên": ["nhân_viên", "staff", "phục_vụ", "lễ_tân"],
    "thân_thiện": ["thân_thiện", "nhiệt_tình", "chu_đáo", "tốt_bụng"],
    "hồ_bơi": ["hồ_bơi", "bể_bơi", "swimming_pool", "pool", "vô_cực"],
    "điều_hòa": ["điều_hòa", "máy_lạnh", "ac", "air_con"],
    "ăn_sáng": ["ăn_sáng", "buffet", "điểm_tâm"],
    "đưa_đón": ["đưa_đón", "đón_tiễn", "sân_bay", "xe_đón"],
    "gia_đình": ["gia_đình", "trẻ_em", "con_nhỏ", "baby", "kid"],
    "cặp_đôi": ["cặp_đôi", "lãng_mạn", "honey_moon", "tình_nhân"],
    "trung_tâm": ["trung_tâm", "gần_chợ", "phố_đi_bộ", "chợ_đêm"],
    "công_tác": ["công_tác", "business", "workation", "phòng_họp", "meeting"],
    "sống_ảo": ["sống_ảo", "checkin", "check_in", "chụp_hình", "vintage"],
    "sinh_thái": ["sinh_thái", "eco", "xanh", "rừng", "thiên_nhiên"],
    "pet_friendly": ["pet_friendly", "thú_cưng", "pet", "đưa_thú_cưng"],
    "gần_ga": ["gần_ga", "ga_tàu", "metro", "tàu_điện"],
    "bếp_nấu": ["bếp_nấu", "bếp_riêng", "kitchen", "bbq", "nướng"],
    "kid_club": ["kid_club", "sân_chơi_trẻ_em", "trẻ_em", "gia_đình"],
    "all_inclusive": ["all_inclusive", "combo", "bao_gồm", "trọn_gói"],
}

# Category hints used for query-time routing/filtering.
CATEGORY_TOKEN_HINTS: dict[str, set[str]] = {
    "mountain": {
        "núi", "đồi", "thung_lũng", "cao_nguyên", "săn_mây", "rừng", "view_núi", "hướng_núi"
    },
    "beach": {
        "biển", "bãi_biển", "ven_biển", "gần_biển", "đảo", "view_biển"
    },
    "family": {
        "gia_đình", "trẻ_em", "kid", "family", "phòng_gia_đình"
    },
    "budget": {
        "giá_rẻ", "bình_dân", "tiết_kiệm", "hợp_lý", "rẻ"
    },
    "luxury": {
        "sang_trọng", "cao_cấp", "luxury", "5_sao", "resort", "villa"
    },
    "center": {
        "trung_tâm", "phố_đi_bộ", "chợ_đêm", "gần_chợ"
    },
    "amenity_pool": {
        "hồ_bơi", "bể_bơi", "pool", "vô_cực"
    },
    "amenity_breakfast": {
        "ăn_sáng", "buffet", "điểm_tâm"
    },
    "airport": {
        "sân_bay", "airport", "đưa_đón", "xe_đón", "gần_ga", "ga_tàu", "metro"
    },
    "spa_gym": {
        "spa", "massage", "gym", "fitness", "xông_hơi", "jacuzzi"
    },
    "photo": {
        "sống_ảo", "checkin", "check_in", "chụp_hình", "vintage", "view_đẹp"
    },
    "quiet": {
        "yên_tĩnh", "nghỉ_dưỡng", "thư_giãn", "đọc_sách", "riêng_tư"
    },
    "pet": {
        "pet", "pet_friendly", "thú_cưng", "đưa_thú_cưng"
    },
    "kitchen": {
        "bếp_nấu", "bếp_riêng", "bbq", "nướng", "kitchen"
    },
    "business": {
        "công_tác", "business", "workation", "phòng_họp", "meeting", "co_working"
    },
}


def _token_forms(tok: str) -> set[str]:
    """Return normalized token forms for robust matching (e.g. view_núi -> view, núi)."""
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

    # 1. Normalize
    normalized = normalize_text(raw_query)

    # 2. Tokenize
    tokens = tokenize_vi(normalized)

    # 3. Lọc index_stopwords
    if stopwords_path and stopwords_path.exists():
        stopwords = load_stopwords(stopwords_path)
        tokens = remove_stopwords(tokens, stopwords)

    # 4. Remove QUERY stopwords (TASK 4)
    core = [tok for tok in tokens if tok.lower() not in QUERY_STOP_TOKENS]

    # 5. Detect Location (TASK 6)
    detected_loc = ""
    loc_remove: set[str] = set()
    
    # 5.1 Trigrams
    for i in range(len(core) - 2):
        tg = (core[i].lower(), core[i + 1].lower(), core[i + 2].lower())
        if tg in _LOCATION_TRIGRAMS:
            detected_loc = _LOCATION_TRIGRAMS[tg]
            loc_remove.update({core[i], core[i + 1], core[i + 2]})
            break

    # 5.2 Bigrams
    if not detected_loc:
        for i in range(len(core) - 1):
            bg = (core[i].lower(), core[i + 1].lower())
            if bg in _LOCATION_BIGRAMS:
                detected_loc = _LOCATION_BIGRAMS[bg]
                loc_remove.update({core[i], core[i + 1]})
                break
            
    # 5.3 Unigrams / Aliases
    if not detected_loc:
        for t in core:
            if t.lower() in _LOCATION_ALIASES:
                detected_loc = _LOCATION_ALIASES[t.lower()]
                loc_remove.add(t)
                break
                
    res.detected_location = detected_loc
    core = [t for t in core if t not in loc_remove]
    res.core_tokens = core

    # 6. Extract Descriptors
    res.descriptor_tokens = [
        t for t in core
        if any(form in DESCRIPTOR_TOKENS for form in _token_forms(t))
    ]

    # 7. Query Expansion (TASK 5)
    expanded: list[str] = []
    seen: set[str] = set()
    for t in core:
        syns = SYNONYM_MAP.get(t.lower(), [t])
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

def location_matched(detected_location: str, hotel_location: str) -> bool:
    if not detected_location or not hotel_location:
        return False
    kws = LOCATION_KEYWORDS.get(detected_location, [detected_location.replace("_", " ")])
    hloc_lower = str(hotel_location).lower()
    return any(kw.lower() in hloc_lower for kw in kws)
