from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd

# -----------------------------
# Aliases (Vietnamese-friendly)
# -----------------------------

LEVEL_ALIASES = {
    "M1": [
        r"\bm1\b",
        r"mức\s*1",
        r"mức\s*độ\s*1",
        r"nhận\s*biết",
        r"\bnb\b",
        r"\bbiết\b",
    ],
    "M2": [
        r"\bm2\b",
        r"mức\s*2",
        r"mức\s*độ\s*2",
        r"thông\s*hiểu",
        r"\bth\b",
        r"\bhiểu\b",
    ],
    "M3": [
        r"\bm3\b",
        r"mức\s*3",
        r"mức\s*độ\s*3",
        r"vận\s*dụng",
        r"\bvd\b",
    ],
    "M3_HIGH": [r"vận\s*dụng\s*cao", r"\bvdc\b"],
}

TYPE_ALIASES = {
    "MCQ": [r"trắc\s*nghiệm", r"\btn\b", r"\bmcq\b", r"chọn\s*đáp\s*án", r"khoanh"],
    "TF": [r"đúng\s*/\s*sai", r"đúng\s*sai", r"\bđ/s\b", r"\btrue\s*false\b"],
    "MATCH": [r"nối", r"ghép", r"nối\s*cột", r"\bmatching\b"],
    "FILL": [r"điền\s*khuyết", r"điền\s*vào\s*chỗ\s*trống"],
    "SHORT": [r"trả\s*lời\s*ngắn", r"\bnêu\b", r"liệt\s*kê", r"\bviết\b"],
    "ESSAY": [r"tự\s*luận", r"\btl\b", r"giải\s*thích", r"trình\s*bày"],
    "PRACTICAL": [r"thực\s*hành", r"thao\s*tác", r"sản\s*phẩm", r"dự\s*án"],
}

# Column alias patterns for mapping
COL_ALIASES = {
    # core
    "content_scope": [r"mạch", r"chủ\s*đề", r"nội\s*dung", r"chương", r"\bbài\b", r"phần", r"tiết"],
    "yccd_ref": [r"mayccd", r"mã\s*yccd", r"\byccd\b", r"yêu\s*cầu\s*cần\s*đạt"],
    "level": [r"mức\s*độ", r"nhận\s*biết", r"thông\s*hiểu", r"vận\s*dụng", r"\bm1\b", r"\bm2\b", r"\bm3\b"],
    "item_type": [r"dạng", r"loại\s*câu", r"hình\s*thức", r"tn", r"tl", r"trắc\s*nghiệm", r"tự\s*luận"],
    "quantity": [r"số\s*c*]()
