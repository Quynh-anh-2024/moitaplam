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
    "quantity": [r"số\s*câu", r"số\s*lượng", r"\bsl\b", r"\bcâu\b"],
    "points_per_item": [r"điểm\s*/\s*câu", r"điểm/câu", r"điểm\s*mỗi\s*câu"],
    "row_points": [r"tổng\s*điểm", r"thành\s*tiền", r"điểm\s*dòng"],
    "constraints": [r"ghi\s*chú", r"yêu\s*cầu", r"lưu\s*ý", r"ràng\s*buộc"],
    # optional (nice to have)
    "topic": [r"chủ\s*điểm", r"chủ\s*đề", r"topic"],
    "lesson": [r"\bbài\b", r"tiết", r"lesson"],
    "knowledge_unit": [r"mạch\s*kiến\s*thức", r"mạch", r"kiến\s*thức"],
}

# -----------------------------
# Utilities
# -----------------------------


def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = _norm(x).replace(",", ".")
    # keep digits, dot, minus
    s = re.sub(r"[^0-9\.\-]", "", s)
    if not s:
        return None
    try:
        v = float(s)
        if v <= 0:
            return None
        return v
    except Exception:
        return None


def _score_col(colname: str, patterns: List[str]) -> int:
    c = _norm(colname)
    return sum(1 for p in patterns if re.search(p, c))


def _detect_level(text: Any) -> Optional[str]:
    t = _norm(text)
    if not t:
        return None
    for lvl, pats in LEVEL_ALIASES.items():
        for p in pats:
            if re.search(p, t):
                return "M3" if lvl == "M3_HIGH" else lvl
    return None


def _detect_type(text: Any) -> Optional[str]:
    t = _norm(text)
    if not t:
        return None
    for typ, pats in TYPE_ALIASES.items():
        for p in pats:
            if re.search(p, t):
                return typ
    return None


def _is_row_effectively_empty(row_dict: Dict[str, Any]) -> bool:
    # empty if all cells are empty strings or NaN
    for v in row_dict.values():
        if v is None:
            continue
        s = _norm(v)
        if s and s != "nan":
            return False
    return True


# -----------------------------
# Header detection & flattening
# -----------------------------


def _find_header_row(df: pd.DataFrame, max_rows: int = 7) -> int:
    """
    Heuristic: pick the row among first max_rows with most keyword hits.
    """
    keywords = [
        "môn",
        "lớp",
        "khối",
        "mức",
        "điểm",
        "câu",
        "nội dung",
        "yccd",
        "mayccd",
        "nhận biết",
        "thông hiểu",
        "vận dụng",
        "dạng",
        "trắc nghiệm",
        "tự luận",
    ]
    best_row, best_score = 0, -10
    lim = min(max_rows, len(df))
    for i in range(lim):
        row_text = " ".join(_norm(x) for x in df.iloc[i].tolist())
        score = sum(1 for k in keywords if k in row_text)

        # penalize rows that are mostly numeric
        numeric = 0
        for x in df.iloc[i].tolist():
            if re.fullmatch(r"[\d\.,]+", _norm(x) or ""):
                numeric += 1
        score -= numeric // 2

        if score > best_score:
            best_score, best_row = score, i
    return best_row


def _flatten_headers(df: pd.DataFrame, header_row: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Combine up to 2 header rows: header_row-1 and header_row
    to improve multi-level headers.
    """
    header0 = df.iloc[header_row].tolist()
    header1 = df.iloc[header_row - 1].tolist() if header_row > 0 else [""] * len(header0)

    cols: List[str] = []
    for a, b in zip(header1, header0):
        a, b = _norm(a), _norm(b)
        if a and b and a != b:
            cols.append(f"{a} {b}")
        elif b:
            cols.append(b)
        elif a:
            cols.append(a)
        else:
            cols.append("")

    for idx, c in enumerate(cols):
        if not c:
            cols[idx] = f"col_{idx+1}"

    data = df.iloc[header_row + 1 :].copy()
    data.columns = cols
    data.reset_index(drop=True, inplace=True)
    return data, {"header_row": header_row, "combined_headers": True, "columns": cols}


def _map_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for target, pats in COL_ALIASES.items():
        best, best_score = None, 0
        for c in columns:
            s = _score_col(c, pats)
            if s > best_score:
                best_score, best = s, c
        mapping[target] = best if best_score > 0 else None
    return mapping


# -----------------------------
# Global meta inference
# -----------------------------


def _extract_global_meta(meta_text: str) -> Dict[str, Optional[str]]:
    """
    Infer subject/grade/term/exam_type from meta text (paragraphs around tables / PDF text).
    """
    t = _norm(meta_text)

    # grade
    grade = None
    m = re.search(r"(lớp|khối)\s*(\d)", t)
    if m:
        grade = m.group(2)

    # term
    term = None
    if "hk1" in t or "học kì 1" in t or "học kỳ 1" in t:
        term = "HK1"
    elif "hk2" in t or "học kì 2" in t or "học kỳ 2" in t:
        term = "HK2"
    else:
        # sometimes "cả năm"
        if "cả năm" in t or "năm học" in t:
            term = "NAM"

    # exam_type
    exam_type = None
    if "giữa kì" in t or "giữa kỳ" in t or re.search(r"\bgk\b", t):
        exam_type = "GIUA_KY"
    elif "cuối kì" in t or "cuối kỳ" in t or re.search(r"\bck\b", t):
        exam_type = "CUOI_KY"
    elif "định kì" in t or "định kỳ" in t or "định kì" in t:
        exam_type = "DINH_KY"

    # subject heuristic
    subj = None
    candidates = [
        ("TOÁN", ["toán"]),
        ("TIẾNG VIỆT", ["tiếng việt", "tieng viet"]),
        ("KHOA HỌC", ["khoa học"]),
        ("LỊCH SỬ", ["lịch sử", "lich su"]),
        ("ĐỊA LÝ", ["địa lý", "dia ly"]),
        ("LỊCH SỬ VÀ ĐỊA LÝ", ["lịch sử và địa lý", "lsđl", "ls&đl", "lsdl"]),
        ("TIN HỌC", ["tin học", "cntt"]),
        ("CÔNG NGHỆ", ["công nghệ"]),
        ("ĐẠO ĐỨC", ["đạo đức"]),
        ("ÂM NHẠC", ["âm nhạc"]),
        ("MĨ THUẬT", ["mĩ thuật", "mỹ thuật"]),
    ]
    for name, keys in candidates:
        if any(k in t for k in keys):
            subj = name
            break

    return {
        "subject": subj,
        "grade": grade,
        "term": term,
        "exam_type": exam_type,
    }


# -----------------------------
# Main builder
# -----------------------------


def build_spec_from_tables(
    tables: List[pd.DataFrame],
    meta_text: str,
    origin_file: str = "",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convert extracted raw tables into a canonical Spec dataframe.

    Output remains compatible with current app pipeline:
    Required legacy columns:
      subject, grade, term_or_exam, content_scope, yccd_ref, level, item_type,
      quantity, points_per_item, row_points, constraints, source_table

    Additional helpful columns:
      spec_id, exam_type, term, origin_file, origin_table, origin_row,
      parse_confidence, inferred_flags, topic, lesson, knowledge_unit
    """
    report: Dict[str, Any] = {
        "tables_used": 0,
        "tables_skipped": 0,
        "table_reports": [],
        "global_inferred": _extract_global_meta(meta_text),
        "notes": [],
    }

    global_meta = report["global_inferred"]
    global_subject = global_meta.get("subject") or "UNKNOWN"
    global_grade = global_meta.get("grade") or "UNKNOWN"
    global_term = global_meta.get("term") or "UNKNOWN"
    global_exam_type = global_meta.get("exam_type") or "UNKNOWN"

    spec_rows: List[Dict[str, Any]] = []

    for ti, raw in enumerate(tables, start=1):
        # basic sanity
        if raw is None or raw.shape[0] < 2 or raw.shape[1] < 3:
            report["tables_skipped"] += 1
            continue

        # choose header & flatten
        header_row = _find_header_row(raw)
        df, hdr_report = _flatten_headers(raw, header_row)
        cols = list(df.columns)

        # map columns
        colmap = _map_columns(cols)

        # detect level columns (NB/TH/VD) in headers: split rows by these columns if present
        level_cols: Dict[str, List[str]] = {"M1": [], "M2": [], "M3": []}
        for c in cols:
            lvl = _detect_level(c)
            if lvl in ("M1", "M2", "M3"):
                level_cols[lvl].append(c)

        # record table report
        report["table_reports"].append(
            {
                "table_index": ti,
                "shape_raw": list(raw.shape),
                "shape_data": list(df.shape),
                "header": hdr_report,
                "colmap": colmap,
                "level_cols": level_cols,
            }
        )

        # If we cannot find any numeric-relevant columns, likely not a matrix
        if not any(colmap.get(k) for k in ["quantity", "points_per_item", "row_points"]) and not any(
            level_cols.values()
        ):
            report["tables_skipped"] += 1
            continue

        report["tables_used"] += 1

        # Iterate rows
        for ridx, row in df.iterrows():
            row_dict = row.to_dict()
            if _is_row_effectively_empty(row_dict):
                continue

            # base text fields
            content = row_dict.get(colmap.get("content_scope") or "", "") if colmap.get("content_scope") else ""
            yccd = row_dict.get(colmap.get("yccd_ref") or "", "") if colmap.get("yccd_ref") else ""
            constraints = row_dict.get(colmap.get("constraints") or "", "") if colmap.get("constraints") else ""

            topic = row_dict.get(colmap.get("topic") or "", "") if colmap.get("topic") else ""
            lesson = row_dict.get(colmap.get("lesson") or "", "") if colmap.get("lesson") else ""
            knowledge_unit = row_dict.get(colmap.get("knowledge_unit") or "", "") if colmap.get("knowledge_unit") else ""

            # row-based level/type
            level = _detect_level(row_dict.get(colmap["level"], "")) if colmap.get("level") else None
            item_type = _detect_type(row_dict.get(colmap["item_type"], "")) if colmap.get("item_type") else None

            # numeric fields
            qty = _parse_number(row_dict.get(colmap["quantity"], "")) if colmap.get("quantity") else None
            ppi = _parse_number(row_dict.get(colmap["points_per_item"], "")) if colmap.get("points_per_item") else None
            rp = _parse_number(row_dict.get(colmap["row_points"], "")) if colmap.get("row_points") else None

            # Confidence & inferred flags
            inferred_flags: List[str] = []
            confidence = 1.0

            def _flag(name: str, penalty: float = 0.1) -> None:
                nonlocal confidence
                inferred_flags.append(name)
                confidence = max(0.0, confidence - penalty)

            # If matrix uses NB/TH/VD columns (level_cols), split row into multiple spec rows
            if any(level_cols.values()):
                any_emitted = False
                for lvl, cands in level_cols.items():
                    if not cands:
                        continue
                    q_lvl = _parse_number(row_dict.get(cands[0], ""))
                    if q_lvl is None or q_lvl <= 0:
                        continue

                    any_emitted = True
                    it = item_type or "UNKNOWN"
                    if it == "UNKNOWN":
                        # do not guess too hard here; validator will handle
                        pass

                    row_points = (float(q_lvl) * float(ppi)) if (ppi is not None) else (rp if rp is not None else None)

                    # flags for missing
                    flags_local = list(inferred_flags)
                    conf_local = confidence

                    if not content and not yccd:
                        flags_local.append("INFER_CONTENT")
                        conf_local = max(0.0, conf_local - 0.2)
                    if item_type is None:
                        flags_local.append("INFER_TYPE")
                        conf_local = max(0.0, conf_local - 0.1)

                    spec_rows.append(
                        {
                            # legacy-compatible core
                            "subject": global_subject,
                            "grade": global_grade,
                            "term_or_exam": global_term if global_term != "UNKNOWN" else global_exam_type,
                            "content_scope": content,
                            "yccd_ref": yccd,
                            "level": lvl,
                            "item_type": it,
                            "quantity": int(q_lvl) if float(q_lvl).is_integer() else q_lvl,
                            "points_per_item": ppi,
                            "row_points": row_points,
                            "constraints": constraints,
                            "source_table": ti,
                            # canonical v1 extras
                            "exam_type": global_exam_type,
                            "term": global_term,
                            "spec_id": "",  # fill later
                            "origin_file": origin_file,
                            "origin_table": ti,
                            "origin_row": int(ridx) + 1,
                            "parse_confidence": round(conf_local, 4),
                            "inferred_flags": ";".join(flags_local),
                            "topic": topic,
                            "lesson": lesson,
                            "knowledge_unit": knowledge_unit,
                        }
                    )
                if any_emitted:
                    continue  # do not also add non-split row

            # No level split columns: add single spec row
            if level is None:
                _flag("INFER_LEVEL", 0.15)
                level_out = "UNKNOWN"
            else:
                level_out = level

            if item_type is None:
                _flag("INFER_TYPE", 0.10)
                type_out = "UNKNOWN"
            else:
                type_out = item_type

            if (not content) and (not yccd):
                _flag("INFER_CONTENT", 0.20)

            # If missing qty entirely, keep None; validator will default to 1
            qty_out: Any = int(qty) if (qty is not None and float(qty).is_integer()) else qty
            row_points = rp if rp is not None else (qty * ppi if (qty is not None and ppi is not None) else None)

            spec_rows.append(
                {
                    # legacy-compatible core
                    "subject": global_subject,
                    "grade": global_grade,
                    "term_or_exam": global_term if global_term != "UNKNOWN" else global_exam_type,
                    "content_scope": content,
                    "yccd_ref": yccd,
                    "level": level_out,
                    "item_type": type_out,
                    "quantity": qty_out,
                    "points_per_item": ppi,
                    "row_points": row_points,
                    "constraints": constraints,
                    "source_table": ti,
                    # canonical v1 extras
                    "exam_type": global_exam_type,
                    "term": global_term,
                    "spec_id": "",  # fill later
                    "origin_file": origin_file,
                    "origin_table": ti,
                    "origin_row": int(ridx) + 1,
                    "parse_confidence": round(confidence, 4),
                    "inferred_flags": ";".join(inferred_flags),
                    "topic": topic,
                    "lesson": lesson,
                    "knowledge_unit": knowledge_unit,
                }
            )

    if not spec_rows:
        report["notes"].append("Không trích xuất được dòng đặc tả hợp lệ; dùng fallback Spec tối thiểu.")
        spec_rows = [
            {
                "subject": global_subject,
                "grade": global_grade,
                "term_or_exam": global_term if global_term != "UNKNOWN" else global_exam_type,
                "content_scope": "Nội dung tổng hợp theo ma trận (fallback)",
                "yccd_ref": "",
                "level": "M1",
                "item_type": "MCQ",
                "quantity": 10,
                "points_per_item": 1.0,
                "row_points": 10.0,
                "constraints": "Fallback",
                "source_table": 1,
                "exam_type": global_exam_type,
                "term": global_term,
                "spec_id": "",
                "origin_file": origin_file,
                "origin_table": 1,
                "origin_row": 1,
                "parse_confidence": 0.5,
                "inferred_flags": "FALLBACK",
                "topic": "",
                "lesson": "",
                "knowledge_unit": "",
            }
        ]

    spec_df = pd.DataFrame(spec_rows)

    # Ensure expected columns exist (for downstream)
    must_cols = [
        "subject",
        "grade",
        "term_or_exam",
        "content_scope",
        "yccd_ref",
        "level",
        "item_type",
        "quantity",
        "points_per_item",
        "row_points",
        "constraints",
        "source_table",
        "spec_id",
        "exam_type",
        "term",
        "origin_file",
        "origin_table",
        "origin_row",
        "parse_confidence",
        "inferred_flags",
        "topic",
        "lesson",
        "knowledge_unit",
    ]
    for c in must_cols:
        if c not in spec_df.columns:
            spec_df[c] = ""

    # Clean string columns
    for c in ["content_scope", "yccd_ref", "constraints", "topic", "lesson", "knowledge_unit"]:
        spec_df[c] = spec_df[c].fillna("").astype(str)

    # Sort by table then row for stable IDs
    spec_df["origin_table"] = pd.to_numeric(spec_df["origin_table"], errors="coerce")
    spec_df["origin_row"] = pd.to_numeric(spec_df["origin_row"], errors="coerce")
    spec_df.sort_values(["origin_table", "origin_row"], inplace=True, na_position="last")
    spec_df.reset_index(drop=True, inplace=True)

    # ---- FIX: forward-fill (index-safe) within each extracted table ----
    # This avoids pandas "incompatible index" errors.
    spec_df["content_scope"] = (
        spec_df["content_scope"]
        .replace("", pd.NA)
        .groupby(spec_df["source_table"], dropna=False)
        .ffill()
        .fillna("")
    )
    spec_df["yccd_ref"] = (
        spec_df["yccd_ref"]
        .replace("", pd.NA)
        .groupby(spec_df["source_table"], dropna=False)
        .ffill()
        .fillna("")
    )
    # optional: forward-fill topic/lesson if they are used and often merged
    spec_df["topic"] = (
        spec_df["topic"]
        .replace("", pd.NA)
        .groupby(spec_df["source_table"], dropna=False)
        .ffill()
        .fillna("")
    )
    spec_df["lesson"] = (
        spec_df["lesson"]
        .replace("", pd.NA)
        .groupby(spec_df["source_table"], dropna=False)
        .ffill()
        .fillna("")
    )
    spec_df["knowledge_unit"] = (
        spec_df["knowledge_unit"]
        .replace("", pd.NA)
        .groupby(spec_df["source_table"], dropna=False)
        .ffill()
        .fillna("")
    )

    # Create spec_id sequential S01, S02...
    spec_df["spec_id"] = [f"S{str(i+1).zfill(2)}" for i in range(len(spec_df))]

    # Normalize enums
    spec_df["level"] = spec_df["level"].fillna("UNKNOWN").astype(str).str.upper()
    spec_df["item_type"] = spec_df["item_type"].fillna("UNKNOWN").astype(str).str.upper()
    spec_df["subject"] = spec_df["subject"].fillna("UNKNOWN").astype(str)
    spec_df["grade"] = spec_df["grade"].fillna("UNKNOWN").astype(str)
    spec_df["term_or_exam"] = spec_df["term_or_exam"].fillna("UNKNOWN").astype(str)
    spec_df["term"] = spec_df["term"].fillna("UNKNOWN").astype(str).str.upper()
    spec_df["exam_type"] = spec_df["exam_type"].fillna("UNKNOWN").astype(str).str.upper()

    return spec_df, report
