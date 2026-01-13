from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd

LEVEL_ALIASES = {
    "M1": [r"\bm1\b", r"mức\s*1", r"mức\s*độ\s*1", r"nhận\s*biết", r"\bnb\b", r"\bbiết\b"],
    "M2": [r"\bm2\b", r"mức\s*2", r"mức\s*độ\s*2", r"thông\s*hiểu", r"\bth\b", r"\bhiểu\b"],
    "M3": [r"\bm3\b", r"mức\s*3", r"mức\s*độ\s*3", r"vận\s*dụng", r"\bvd\b"],
    "M3_HIGH": [r"vận\s*dụng\s*cao", r"\bvdc\b"],
}

TYPE_ALIASES = {
    "MCQ": [r"trắc\s*nghiệm", r"\btn\b", r"mcq", r"chọn\s*đáp\s*án", r"khoanh"],
    "TF": [r"đúng\s*/\s*sai", r"đúng\s*sai", r"\bđ/s\b", r"true\s*false"],
    "MATCH": [r"nối", r"ghép", r"nối\s*cột", r"matching"],
    "FILL": [r"điền\s*khuyết", r"điền\s*vào\s*chỗ\s*trống"],
    "SHORT": [r"trả\s*lời\s*ngắn", r"nêu", r"liệt\s*kê", r"viết"],
    "ESSAY": [r"tự\s*luận", r"\btl\b", r"giải\s*thích", r"trình\s*bày"],
    "PRACTICAL": [r"thực\s*hành", r"thao\s*tác", r"sản\s*phẩm", r"dự\s*án"],
}

COL_ALIASES = {
    "content_scope": [r"mạch", r"chủ\s*đề", r"nội\s*dung", r"chương", r"bài", r"chủ\s*điểm", r"phần"],
    "yccd_ref": [r"mayccd", r"mã\s*yccd", r"yccd", r"yêu\s*cầu\s*cần\s*đạt"],
    "level": [r"mức\s*độ", r"nhận\s*biết", r"thông\s*hiểu", r"vận\s*dụng", r"m1", r"m2", r"m3", r"vd", r"nb", r"th"],
    "item_type": [r"dạng", r"loại\s*câu", r"hình\s*thức", r"tn", r"tl", r"trắc\s*nghiệm", r"tự\s*luận"],
    "quantity": [r"số\s*câu", r"số\s*lượng", r"\bsl\b", r"câu"],
    "points_per_item": [r"điểm\s*/\s*câu", r"điểm/câu", r"điểm"],
    "row_points": [r"tổng\s*điểm", r"thành\s*tiền", r"điểm\s*dòng"],
    "constraints": [r"ghi\s*chú", r"yêu\s*cầu", r"lưu\s*ý", r"ràng\s*buộc"],
}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _score_col(colname: str, patterns: List[str]) -> int:
    c = _norm(colname)
    return sum(1 for p in patterns if re.search(p, c))

def _find_header_row(df: pd.DataFrame, max_rows: int = 6) -> int:
    keywords = ["mức", "điểm", "câu", "nội dung", "yccd", "bài", "chủ đề", "nhận biết", "thông hiểu", "vận dụng"]
    best_row, best_score = 0, -1
    lim = min(max_rows, len(df))
    for i in range(lim):
        row = " ".join([_norm(str(x)) for x in df.iloc[i].tolist()])
        score = sum(1 for k in keywords if k in row)
        numeric = sum(1 for x in df.iloc[i].tolist() if re.fullmatch(r"[\d\.\,]+", _norm(str(x)) or ""))
        score -= numeric // 2
        if score > best_score:
            best_score, best_row = score, i
    return best_row

def _flatten_headers(df: pd.DataFrame, header_row: int) -> Tuple[pd.DataFrame, Dict]:
    header0 = df.iloc[header_row].tolist()
    header1 = df.iloc[header_row-1].tolist() if header_row > 0 else [""] * len(header0)
    cols = []
    for a, b in zip(header1, header0):
        a, b = _norm(str(a)), _norm(str(b))
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
    data = df.iloc[header_row+1:].copy()
    data.columns = cols
    data.reset_index(drop=True, inplace=True)
    return data, {"header_row": header_row, "combined_headers": True}

def _map_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    mapping = {}
    for target, pats in COL_ALIASES.items():
        best, best_score = None, 0
        for c in columns:
            s = _score_col(c, pats)
            if s > best_score:
                best_score, best = s, c
        mapping[target] = best if best_score > 0 else None
    return mapping

def _parse_number(x: str) -> Optional[float]:
    if x is None:
        return None
    s = _norm(str(x)).replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _detect_level(text: str) -> Optional[str]:
    t = _norm(text)
    for lvl, pats in LEVEL_ALIASES.items():
        for p in pats:
            if re.search(p, t):
                return "M3" if lvl == "M3_HIGH" else lvl
    return None

def _detect_type(text: str) -> Optional[str]:
    t = _norm(text)
    for typ, pats in TYPE_ALIASES.items():
        for p in pats:
            if re.search(p, t):
                return typ
    return None

def _extract_subject_grade(meta_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    t = _norm(meta_text)
    grade = None
    term = None
    subj = None

    m = re.search(r"lớp\s*(\d)", t)
    if m:
        grade = m.group(1)

    if "hk1" in t or "học kì 1" in t:
        term = "HK1"
    elif "hk2" in t or "học kì 2" in t:
        term = "HK2"
    elif "giữa kì" in t or "gk" in t:
        term = "GIUA_KY"
    elif "cuối kì" in t or "ck" in t:
        term = "CUOI_KY"

    candidates = ["toán", "tiếng việt", "khoa học", "lịch sử", "địa lý", "lsđl", "tin học", "công nghệ", "đạo đức", "âm nhạc", "mĩ thuật"]
    for c in candidates:
        if c in t:
            subj = c.upper()
            break
    return subj, grade, term

def build_spec_from_tables(tables: List[pd.DataFrame], meta_text: str) -> Tuple[pd.DataFrame, Dict]:
    report: Dict = {"tables_used": 0, "tables_skipped": 0, "table_reports": [], "global_inferred": {}, "notes": []}
    inferred_subject, inferred_grade, inferred_term = _extract_subject_grade(meta_text)
    report["global_inferred"] = {"subject": inferred_subject, "grade": inferred_grade, "term_or_exam": inferred_term}

    spec_rows = []
    for ti, raw in enumerate(tables, start=1):
        if raw.shape[0] < 2 or raw.shape[1] < 3:
            report["tables_skipped"] += 1
            continue

        header_row = _find_header_row(raw)
        df, hdr_report = _flatten_headers(raw, header_row)
        cols = list(df.columns)
        colmap = _map_columns(cols)

        report["table_reports"].append({"table_index": ti, "shape_raw": list(raw.shape), "shape_data": list(df.shape), "colmap": colmap, "header": hdr_report})

        if not any(colmap.get(k) for k in ["quantity", "points_per_item", "row_points"]):
            report["tables_skipped"] += 1
            continue

        report["tables_used"] += 1

        level_cols = {"M1": [], "M2": [], "M3": []}
        for c in cols:
            lvl = _detect_level(c)
            if lvl in level_cols:
                level_cols[lvl].append(c)

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            content = row_dict.get(colmap.get("content_scope") or "", "") if colmap.get("content_scope") else ""
            yccd = row_dict.get(colmap.get("yccd_ref") or "", "") if colmap.get("yccd_ref") else ""
            constraints = row_dict.get(colmap.get("constraints") or "", "") if colmap.get("constraints") else ""

            row_text = " ".join([_norm(str(v)) for v in row_dict.values()])
            if not (_norm(content) or _norm(yccd) or re.search(r"\d", row_text)):
                continue

            level = _detect_level(row_dict.get(colmap["level"], "")) if colmap.get("level") else None
            item_type = _detect_type(row_dict.get(colmap["item_type"], "")) if colmap.get("item_type") else None

            qty = _parse_number(row_dict.get(colmap["quantity"], "")) if colmap.get("quantity") else None
            ppi = _parse_number(row_dict.get(colmap["points_per_item"], "")) if colmap.get("points_per_item") else None
            rp = _parse_number(row_dict.get(colmap["row_points"], "")) if colmap.get("row_points") else None

            if any(level_cols.values()):
                for lvl, cands in level_cols.items():
                    if not cands:
                        continue
                    q = _parse_number(row_dict.get(cands[0], ""))
                    if q is None or q <= 0:
                        continue
                    spec_rows.append({
                        "subject": inferred_subject,
                        "grade": inferred_grade,
                        "term_or_exam": inferred_term,
                        "content_scope": content,
                        "yccd_ref": yccd,
                        "level": lvl,
                        "item_type": item_type,
                        "quantity": int(q) if float(q).is_integer() else q,
                        "points_per_item": ppi,
                        "row_points": (float(q) * ppi) if (ppi is not None) else rp,
                        "constraints": constraints,
                        "source_table": ti,
                    })
                continue

            spec_rows.append({
                "subject": inferred_subject,
                "grade": inferred_grade,
                "term_or_exam": inferred_term,
                "content_scope": content,
                "yccd_ref": yccd,
                "level": level,
                "item_type": item_type,
                "quantity": int(qty) if (qty is not None and float(qty).is_integer()) else qty,
                "points_per_item": ppi,
                "row_points": rp if rp is not None else (qty * ppi if (qty is not None and ppi is not None) else None),
                "constraints": constraints,
                "source_table": ti,
            })

    if not spec_rows:
        report["notes"].append("Không trích xuất được dòng đặc tả hợp lệ; dùng fallback Spec tối thiểu.")
        spec_rows = [{
            "subject": inferred_subject or "UNKNOWN",
            "grade": inferred_grade or "UNKNOWN",
            "term_or_exam": inferred_term or "UNKNOWN",
            "content_scope": "Nội dung tổng hợp theo ma trận (fallback)",
            "yccd_ref": "",
            "level": "M1",
            "item_type": "MCQ",
            "quantity": 10,
            "points_per_item": 1.0,
            "row_points": 10.0,
            "constraints": "Fallback",
            "source_table": None,
        }]

    spec_df = pd.DataFrame(spec_rows)

    # forward-fill within table to mitigate merged cells -> blanks
    spec_df["content_scope"] = spec_df["content_scope"].fillna("").astype(str)
    spec_df["yccd_ref"] = spec_df["yccd_ref"].fillna("").astype(str)
    spec_df["constraints"] = spec_df["constraints"].fillna("").astype(str)

    if "source_table" in spec_df.columns:
        spec_df.sort_values(["source_table"], inplace=True, na_position="last")
        spec_df["content_scope"] = spec_df.groupby("source_table")["content_scope"].apply(lambda s: s.replace("", pd.NA).ffill().fillna(""))
        spec_df["yccd_ref"] = spec_df.groupby("source_table")["yccd_ref"].apply(lambda s: s.replace("", pd.NA).ffill().fillna(""))
        spec_df.reset_index(drop=True, inplace=True)

    spec_df["level"] = spec_df["level"].fillna("UNKNOWN").astype(str)
    spec_df["item_type"] = spec_df["item_type"].fillna("UNKNOWN").astype(str)
    spec_df["subject"] = spec_df["subject"].fillna("UNKNOWN").astype(str)
    spec_df["grade"] = spec_df["grade"].fillna("UNKNOWN").astype(str)
    spec_df["term_or_exam"] = spec_df["term_or_exam"].fillna("UNKNOWN").astype(str)

    return spec_df, report
