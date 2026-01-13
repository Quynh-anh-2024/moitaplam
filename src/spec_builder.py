from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


# -----------------------------
# Aliases (TT27-friendly)
# -----------------------------

LEVEL_ALIASES: Dict[str, List[str]] = {
    "M1": [r"\bm1\b", r"mức\s*1", r"mức\s*độ\s*1", r"nhận\s*biết", r"\bnb\b", r"\bbiết\b"],
    "M2": [r"\bm2\b", r"mức\s*2", r"mức\s*độ\s*2", r"thông\s*hiểu", r"\bth\b", r"\bhiểu\b"],
    "M3": [r"\bm3\b", r"mức\s*3", r"mức\s*độ\s*3", r"vận\s*dụng", r"\bvd\b"],
    "M3_HIGH": [r"vận\s*dụng\s*cao", r"\bvdc\b"],
}

TYPE_ALIASES: Dict[str, List[str]] = {
    "MCQ": [r"trắc\s*nghiệm", r"nhiều\s*lựa\s*chọn", r"chọn\s*đáp\s*án", r"mcq", r"khoanh"],
    "TF": [r"đúng\s*-\s*sai", r"đúng\s*/\s*sai", r"đúng\s*sai", r"\bđ/s\b", r"true\s*false"],
    "MATCH": [r"nối\s*cột", r"\bnối\b", r"ghép", r"matching"],
    "FILL": [r"điền\s*khuyết", r"điền\s*chỗ\s*trống", r"điền\s*từ", r"fill"],
    "ESSAY": [r"tự\s*luận", r"\btl\b", r"tự luận", r"tự\s*viết", r"trình\s*bày", r"nêu"],
}

# columns (base, not count columns)
COL_ALIASES: Dict[str, List[str]] = {
    "tt": [r"^tt$", r"stt", r"số\s*tt", r"thứ\s*tự"],
    "topic": [r"chương\s*/\s*chủ\s*đề", r"chương", r"chủ\s*đề", r"mạch"],
    "content_unit": [r"nội\s*dung", r"đơn\s*vị\s*kiến\s*thức", r"bài"],
    "yccd_ref": [r"mayccd", r"mã\s*yccd", r"\byccd\b", r"yêu\s*cầu\s*cần\s*đạt"],
    "constraints": [r"ghi\s*chú", r"lưu\s*ý", r"ràng\s*buộc", r"yêu\s*cầu\s*khác"],
    # IMPORTANT: prioritize "điểm từng bài" over "số điểm cần đạt"
    "row_points": [r"điểm\s*từng\s*bài", r"điểm\s*từng", r"điểm\s*bài", r"tổng\s*điểm", r"điểm\s*dòng"],
    "total_qty": [r"tổng\s*số\s*câu", r"tổng\s*câu", r"số\s*câu/\s*ý", r"tổng\s*số\s*câu/\s*ý"],
}

DEFAULT_POINTS_PER_ITEM: Dict[str, float] = {
    "MCQ": 0.5,
    "TF": 0.5,
    "MATCH": 0.5,
    "FILL": 0.5,
    "ESSAY": 1.0,
}


# -----------------------------
# helpers
# -----------------------------

def _norm(s: Any) -> str:
    s = ("" if s is None else str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _score_col(colname: str, patterns: List[str]) -> int:
    c = _norm(colname)
    return sum(1 for p in patterns if re.search(p, c))

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

def _parse_number(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = _norm(x).replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in ("", "-", ".", "-."):
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

def _extract_subject_grade_term(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    t = _norm(text)

    grade = None
    m = re.search(r"lớp\s*[:\-]?\s*(\d+)", t)
    if m:
        grade = m.group(1)

    term = None
    if "hk1" in t or "học kì i" in t or "học kì 1" in t or "cuối học kì i" in t:
        term = "CUOI_HK1"
    elif "hk2" in t or "học kì ii" in t or "học kì 2" in t or "cuối học kì ii" in t:
        term = "CUOI_HK2"
    elif "giữa kì" in t or "giữa kỳ" in t or re.search(r"\bgk\b", t):
        term = "GIUA_KY"
    elif "cuối kì" in t or "cuối kỳ" in t or re.search(r"\bck\b", t):
        term = "CUOI_KY"

    subject = None
    if "ls&đl" in t or "ls&dl" in t or "lịch sử và địa lí" in t or "lịch sử và địa lý" in t:
        subject = "LS&ĐL"
    else:
        candidates = [
            ("toán", "TOAN"),
            ("tiếng việt", "TIENG_VIET"),
            ("khoa học", "KHOA_HOC"),
            ("tin học", "TIN_HOC"),
            ("công nghệ", "CONG_NGHE"),
            ("đạo đức", "DAO_DUC"),
            ("âm nhạc", "AM_NHAC"),
            ("mĩ thuật", "MI_THUAT"),
            ("lịch sử", "LICH_SU"),
            ("địa lí", "DIA_LI"),
        ]
        for k, v in candidates:
            if k in t:
                subject = v
                break

    return subject, grade, term

def _row_join(df: pd.DataFrame, start: int, end: int) -> str:
    end = min(end, len(df))
    chunks: List[str] = []
    for i in range(start, end):
        row = " ".join(_norm(x) for x in df.iloc[i].tolist() if _norm(x))
        if row:
            chunks.append(row)
    return " ".join(chunks)

def _detect_three_row_header(raw: pd.DataFrame, max_scan: int = 12) -> Optional[Tuple[int, int, int]]:
    base_keys = ["chương", "chủ đề", "nội dung", "đơn vị kiến thức", "yêu cầu cần đạt", "số tiết", "tỉ lệ", "điểm"]
    type_keys = ["trắc nghiệm", "nhiều lựa chọn", "đúng", "sai", "nối", "điền", "khuyết", "tự luận"]
    level_keys = ["biết", "nhận biết", "hiểu", "thông hiểu", "vd", "vận dụng"]

    lim = min(max_scan, len(raw) - 2)
    best = None
    best_score = -1
    for i in range(lim):
        t0 = _row_join(raw, i, i + 1)
        score0 = sum(1 for k in base_keys if k in t0)

        t1 = _row_join(raw, i + 1, i + 2)
        score1 = sum(1 for k in type_keys if k in t1)

        t2 = _row_join(raw, i + 2, i + 3)
        score2 = sum(1 for k in level_keys if k in t2)

        if score0 >= 3 and score1 >= 2 and score2 >= 2:
            total = score0 * 3 + score1 * 2 + score2 * 2
            if total > best_score:
                best_score = total
                best = (i, i + 1, i + 2)
    return best

def _build_df_three_row(raw: pd.DataFrame, base_row: int, type_row: int, level_row: int) -> Tuple[pd.DataFrame, Dict]:
    base = raw.iloc[base_row].tolist()
    typ = raw.iloc[type_row].tolist()
    lvl = raw.iloc[level_row].tolist()

    # forward-fill type labels, but reset when base row starts a new block (e.g., "Tự luận")
    typ_ff: List[str] = []
    last = ""
    for j, x in enumerate(typ):
        s = _norm(x)
        if s:
            last = s

        b = _norm(base[j])
        b_type = _detect_type(b)
        last_type = _detect_type(last) if last else None
        if b_type and last_type and (b_type != last_type) and (not s):
            last = b

        typ_ff.append(last)

    core_headers = {
        "tt", "stt", "chương/chủ đề", "chương", "chủ đề",
        "nội dung/đơn vị kiến thức", "nội dung", "yêu cầu cần đạt", "số tiết", "tỉ lệ",
        "số điểm cần đạt", "tổng số câu/ý", "điểm từng bài", "tổng số câu", "tổng điểm"
    }

    cols: List[str] = []
    for j, (b_raw, t_raw, l_raw) in enumerate(zip(base, typ_ff, lvl)):
        b0 = _norm(b_raw)
        t0 = _norm(t_raw)
        l0 = _norm(l_raw)

        if b0 in core_headers:
            name = b0
        else:
            if t0 and l0:
                name = f"{t0} {l0}"
            elif (not t0) and b0 and l0:
                name = f"{b0} {l0}"
            elif t0:
                name = t0
            elif b0:
                name = b0
            elif l0:
                name = l0
            else:
                name = f"col_{j}"
        cols.append(name)

    # de-duplicate columns (critical)
    seen: Dict[str, int] = {}
    cols2: List[str] = []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        cols2.append(c if seen[c] == 1 else f"{c}__{seen[c]}")

    df = raw.iloc[level_row + 1 :].copy()
    df.columns = cols2
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df, {"header_style": "three_row", "base_row": base_row, "type_row": type_row, "level_row": level_row}

def build_spec_from_tables(tables: List[pd.DataFrame], meta_text: str) -> Tuple[pd.DataFrame, Dict]:
    report: Dict[str, Any] = {
        "tables_used": 0,
        "tables_skipped": 0,
        "table_reports": [],
        "global_inferred": {},
        "notes": [],
    }

    inferred_subject, inferred_grade, inferred_term = _extract_subject_grade_term(meta_text)
    report["global_inferred"] = {"subject": inferred_subject, "grade": inferred_grade, "term_or_exam": inferred_term}

    spec_rows: List[Dict[str, Any]] = []

    for ti, raw in enumerate(tables, start=1):
        if raw is None or raw.shape[0] < 4 or raw.shape[1] < 6:
            report["tables_skipped"] += 1
            continue

        top_text = _row_join(raw, 0, min(12, len(raw)))
        sub_i, grade_i, term_i = _extract_subject_grade_term(meta_text + " " + top_text)
        subj = sub_i or inferred_subject or "UNKNOWN"
        grade = grade_i or inferred_grade or "UNKNOWN"
        term = term_i or inferred_term or "UNKNOWN"

        hdr3 = _detect_three_row_header(raw)
        if not hdr3:
            report["tables_skipped"] += 1
            report["table_reports"].append({"table": ti, "reason": "không nhận diện được header 3 tầng TT27"})
            continue

        df, hdr_report = _build_df_three_row(raw, *hdr3)
        cols = list(df.columns)
        colmap = _map_columns(cols)

        # identify count columns by parsing both type and level from header name
        count_cols: List[Tuple[str, str, str]] = []
        for c in cols:
            typ = _detect_type(c)
            lvl = _detect_level(c)
            if typ and lvl:
                count_cols.append((c, typ, lvl))

        if not count_cols:
            report["tables_skipped"] += 1
            report["table_reports"].append({"table": ti, "reason": "không có cột số câu theo dạng+mức", "colmap": colmap})
            continue

        report["tables_used"] += 1
        report["table_reports"].append({"table": ti, "header": hdr_report, "colmap": colmap, "count_cols": len(count_cols), "shape": list(df.shape)})

        topic_ff = ""
        summary_kw = ["tổng", "cộng", "điểm 1 câu", "mức độ đánh giá", "điểm từng phần", "số câu/ý"]

        for _, row in df.iterrows():
            row_dict = row.to_dict()

            # Hard filter: only keep real data rows using TT column
            tt_val = row_dict.get(colmap.get("tt") or "", "") if colmap.get("tt") else None
            tt_num = _parse_number(tt_val) if tt_val is not None else None

            content_unit_raw = str(row_dict.get(colmap.get("content_unit") or "", "")).strip() if colmap.get("content_unit") else ""
            if colmap.get("tt"):
                if tt_num is None or tt_num <= 0 or tt_num > 200:
                    if not re.match(r"^(bài|chủ đề)\b", _norm(content_unit_raw)):
                        continue
            else:
                if any(k in _norm(content_unit_raw) for k in summary_kw):
                    continue

            topic_val = row_dict.get(colmap.get("topic") or "", "") if colmap.get("topic") else ""
            if _norm(topic_val):
                topic_ff = str(topic_val).strip()

            content_unit = str(row_dict.get(colmap.get("content_unit") or "", "")).strip() if colmap.get("content_unit") else ""
            yccd = str(row_dict.get(colmap.get("yccd_ref") or "", "")).strip() if colmap.get("yccd_ref") else ""
            constraints = str(row_dict.get(colmap.get("constraints") or "", "")).strip() if colmap.get("constraints") else ""

            # target row points
            row_points_target = _parse_number(row_dict.get(colmap.get("row_points"), "")) if colmap.get("row_points") else None

            # collect quantities
            q_items: List[Tuple[str, str, float]] = []
            for c, typ, lvl in count_cols:
                q = _parse_number(row_dict.get(c, None))
                if q is None or q <= 0:
                    continue
                q_items.append((typ, lvl, q))

            if not q_items:
                continue

            content_scope_parts = []
            if topic_ff:
                content_scope_parts.append(topic_ff)
            if content_unit and content_unit not in content_scope_parts:
                content_scope_parts.append(content_unit)
            content_scope = " - ".join([p for p in content_scope_parts if p])

            # scale points_per_item so each lesson row matches "Điểm từng bài"
            default_sum = sum(DEFAULT_POINTS_PER_ITEM.get(t, 0.5) * q for t, _, q in q_items)
            scale = None
            if row_points_target is not None and default_sum > 0:
                scale = float(row_points_target) / float(default_sum)

            for typ, lvl, q in q_items:
                base_ppi = DEFAULT_POINTS_PER_ITEM.get(typ, 0.5)
                ppi = base_ppi
                if scale is not None:
                    ppi = base_ppi * scale
                    ppi = round(ppi * 4) / 4.0  # 0.25 step

                spec_rows.append({
                    "subject": subj,
                    "grade": grade,
                    "term_or_exam": term,
                    "content_scope": content_scope or "",
                    "yccd_ref": yccd or "",
                    "level": lvl,
                    "item_type": typ,
                    "quantity": int(q) if float(q).is_integer() else q,
                    "points_per_item": ppi,
                    "row_points": float(q) * float(ppi),
                    "constraints": constraints,
                    "source_table": ti,
                })

    spec_df = pd.DataFrame(spec_rows)
    if spec_df.empty:
        report["notes"].append("Không trích xuất được Spec.")
        return spec_df, report

    # forward fill within each source table (FIX groupby.apply bug)
    spec_df["content_scope"] = spec_df["content_scope"].fillna("").astype(str)
    spec_df["yccd_ref"] = spec_df["yccd_ref"].fillna("").astype(str)
    if "source_table" in spec_df.columns:
        spec_df.sort_values(["source_table"], inplace=True, na_position="last")
        spec_df["content_scope"] = spec_df.groupby("source_table")["content_scope"].transform(
            lambda s: s.replace("", pd.NA).ffill().fillna("")
        )
        spec_df["yccd_ref"] = spec_df.groupby("source_table")["yccd_ref"].transform(
            lambda s: s.replace("", pd.NA).ffill().fillna("")
        )
        spec_df.reset_index(drop=True, inplace=True)

    return spec_df, report
