"""
Debug Logger for Search Engine
Ghi log debug information ra file JSON để phân tích
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug_logs"


def save_debug_log(
    query: str,
    results: list[dict],
    query_understanding: dict,
    weights: dict,
    filename: str | None = None,
) -> Path:
    """Lưu debug information ra file JSON.
    
    Args:
        query: Query string người dùng nhập
        results: Danh sách kết quả từ search_hybrid
        query_understanding: Thông tin query understanding
        weights: Các trọng số đã sử dụng
        filename: Tên file tùy chọn, nếu không có sẽ tự tạo theo timestamp
        
    Returns:
        Path đến file đã lưu
    """
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{timestamp}.json"
    elif not filename.endswith(".json"):
        filename = f"{filename}.json"
        
    filepath = DEBUG_DIR / filename
    
    # Extract debug_info từ results
    debug_results = []
    for r in results:
        info = r.get("debug_info", {})
        debug_results.append({
            "rank": len(debug_results) + 1,
            "hotel_name": info.get("hotel_name", r.get("hotel_name", "")),
            "hybrid_score": r.get("hybrid_score", 0),
            "debug": info,
        })
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "weights": weights,
        "query_understanding": query_understanding,
        "total_results": len(debug_results),
        "results": debug_results,
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    return filepath


def get_latest_debug_log() -> dict | None:
    """Đọc debug log mới nhất.
    
    Returns:
        Dữ liệu từ file log mới nhất, hoặc None nếu không có file nào.
    """
    if not DEBUG_DIR.exists():
        return None
        
    files = list(DEBUG_DIR.glob("debug_*.json"))
    if not files:
        return None
        
    latest = max(files, key=lambda f: f.stat().st_mtime)
    
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_old_logs(keep: int = 5):
    """Xóa các log cũ, giữ lại `keep` file mới nhất."""
    if not DEBUG_DIR.exists():
        return
        
    files = sorted(DEBUG_DIR.glob("debug_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        f.unlink()