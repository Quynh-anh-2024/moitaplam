from __future__ import annotations
from typing import Dict, Tuple
import math
import pandas as pd

DEFAULT_TYPE_POINTS = {
    "MCQ": 0.5,
    "TF": 0.25,
    "MATCH": 0.5,
    "FILL": 0.5,
    "SHORT": 1.0,
    "ESSAY": 1.0,
    "PRACTICAL": 1.0,
    "UNKNOWN": 0.5,
}

def _round_to_step(x: float, step: float) -> float:
    return round(round(x / step) * step, 10)

def validate_and_repair_spec(spec_df: pd.DataFrame, total_points_target: float = 10.0, point_step: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    df = spec_df.copy()
    report: Dict = {
        "errors_fixed": [],
        "warnings": [],
        "confidence": 1.0,
        "total_points_target": total_points_target,
        "point_step": point_step,
    }

    required = ["subject","grade","term_or_exam","content_scope","yccd_ref","level","item_type","quantity","points_per_item","row_points","constraints"]
    for c in required:
        if c not in df.columns:
            df[c] = "" if c not in ["quantity","points_per_item","row_points"] else None
            report["warnings"].append(f"Thiếu cột {c}; đã tạo cột rỗng.")

    def _to_int_or_none(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            if isinstance(v, str) and not v.strip():
                return None
            f = float(v)
            if f <= 0:
                return None
            return int(f) if float(f).is_integer() else f
        except Exception:
            return None

    def _to_float_or_none(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            if isinstance(v, str) and not v.strip():
                return None
            f = float(v)
            return f if f > 0 else None
        except Exception:
            return None

    df["quantity"] = df["quantity"].apply(_to_int_or_none)
    missing_q = df["quantity"].isna().sum()
    if missing_q:
        df.loc[df["quantity"].isna(), "quantity"] = 1
        report["errors_fixed"].append(f"Thiếu số câu ở {missing_q} dòng → mặc định 1.")

    df["points_per_item"] = df["points_per_item"].apply(_to_float_or_none)
    df["row_points"] = df["row_points"].apply(_to_float_or_none)

    for i in range(len(df)):
        q = df.at[i, "quantity"]
        ppi = df.at[i, "points_per_item"]
        rp = df.at[i, "row_points"]
        typ = str(df.at[i, "item_type"] or "UNKNOWN").upper()

        if ppi is None and rp is not None and q:
            ppi = rp / float(q)
            df.at[i, "points_per_item"] = ppi
            report["errors_fixed"].append(f"Dòng {i+1}: suy ra điểm/câu = tổng điểm / số câu.")
        if rp is None and ppi is not None and q:
            rp = float(q) * ppi
            df.at[i, "row_points"] = rp
            report["errors_fixed"].append(f"Dòng {i+1}: suy ra tổng điểm dòng = số câu × điểm/câu.")
        if ppi is None and rp is None:
            ppi = DEFAULT_TYPE_POINTS.get(typ, DEFAULT_TYPE_POINTS["UNKNOWN"])
            ppi = _round_to_step(ppi, point_step)
            df.at[i, "points_per_item"] = ppi
            df.at[i, "row_points"] = float(q) * ppi
            report["errors_fixed"].append(f"Dòng {i+1}: thiếu điểm → mặc định theo dạng câu ({typ}).")

    df["points_per_item"] = df["points_per_item"].apply(lambda x: _round_to_step(float(x), point_step))
    df["row_points"] = df.apply(lambda r: _round_to_step(float(r["quantity"]) * float(r["points_per_item"]), point_step), axis=1)

    total = float(df["row_points"].sum())
    diff = _round_to_step(total_points_target - total, point_step)

    unknown_level_ratio = (df["level"].astype(str).str.upper().eq("UNKNOWN").sum()) / max(1, len(df))
    unknown_type_ratio = (df["item_type"].astype(str).str.upper().eq("UNKNOWN").sum()) / max(1, len(df))
    report["confidence"] = max(0.0, 1.0 - 0.5 * unknown_level_ratio - 0.3 * unknown_type_ratio)

    if abs(diff) > 1e-9:
        last = len(df) - 1
        q_last = float(df.at[last, "quantity"])
        ppi_last = float(df.at[last, "points_per_item"])
        new_ppi = ppi_last + (diff / q_last)
        new_ppi = _round_to_step(new_ppi, point_step)

        if new_ppi > 0:
            df.at[last, "points_per_item"] = new_ppi
            df.at[last, "row_points"] = _round_to_step(q_last * new_ppi, point_step)
            report["errors_fixed"].append("Cân tổng điểm: điều chỉnh điểm/câu ở dòng cuối.")
        else:
            bal = {
                "subject": df.at[0, "subject"],
                "grade": df.at[0, "grade"],
                "term_or_exam": df.at[0, "term_or_exam"],
                "content_scope": "Câu cân điểm (tự động)",
                "yccd_ref": "",
                "level": "M1",
                "item_type": "MCQ",
                "quantity": 1,
                "points_per_item": abs(diff),
                "row_points": abs(diff),
                "constraints": "Auto-balance",
            }
            df = pd.concat([df, pd.DataFrame([bal])], ignore_index=True)
            report["errors_fixed"].append("Cân tổng điểm: thêm 1 câu cân điểm tự động.")

        df["points_per_item"] = df["points_per_item"].apply(lambda x: _round_to_step(float(x), point_step))
        df["row_points"] = df.apply(lambda r: _round_to_step(float(r["quantity"]) * float(r["points_per_item"]), point_step), axis=1)

    report["final_total_points"] = float(df["row_points"].sum())
    report["final_total_questions"] = int(df["quantity"].sum())

    if unknown_level_ratio > 0.3:
        report["warnings"].append("Nhiều dòng chưa nhận diện được mức độ (UNKNOWN). Đề vẫn sinh nhưng cần rà soát mức độ TT27.")
    if unknown_type_ratio > 0.3:
        report["warnings"].append("Nhiều dòng chưa nhận diện được dạng câu (UNKNOWN). Đề vẫn sinh nhưng cần rà soát hình thức câu hỏi.")
    if abs(report["final_total_points"] - total_points_target) > 1e-6:
        report["warnings"].append("Không cân được đúng tổng điểm mục tiêu theo bước điểm; kiểm tra lại Spec.")

    return df, report
