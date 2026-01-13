from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import pandas as pd

@dataclass
class QuestionItem:
    q_no: int
    spec_row_id: int
    level: str
    item_type: str
    points: float
    stem: str
    options: Optional[List[str]]
    answer: str
    explanation: str
    rubric: str

@dataclass
class Exam:
    subject: str
    grade: str
    term_or_exam: str
    questions: List[QuestionItem]

def _seed_from_spec(df: pd.DataFrame) -> int:
    s = "|".join(df["content_scope"].astype(str).head(30).tolist())
    return abs(hash(s)) % (2**32 - 1)

def _vn_level_phrase(level: str) -> str:
    level = (level or "UNKNOWN").upper()
    return {"M1": "Nhận biết", "M2": "Thông hiểu", "M3": "Vận dụng"}.get(level, "Chưa xác định")

def _make_mcq(rng: random.Random) -> (List[str], str):
    base = ["Phương án A", "Phương án B", "Phương án C", "Phương án D"]
    rng.shuffle(base)
    correct_idx = rng.randint(0, 3)
    options = [f"{chr(65+i)}. {base[i]}" for i in range(4)]
    answer = chr(65 + correct_idx)
    return options, answer

def _generate_one(content: str, subject: str, grade: str, level: str, item_type: str, points: float, rng: random.Random) -> Dict:
    content = (content or "").strip() or "nội dung trong ma trận"
    subject = subject or "Môn học"
    grade = grade or "?"
    level_phrase = _vn_level_phrase(level)
    item_type = (item_type or "UNKNOWN").upper()

    if item_type in ("MCQ", "TF", "MATCH", "FILL", "UNKNOWN"):
        stem = f"[{subject} Lớp {grade} – {level_phrase}] Câu hỏi trắc nghiệm về: {content}. Chọn đáp án đúng."
        options, ans = _make_mcq(rng)
        expl = "Giải thích: Câu hỏi được sinh theo dòng đặc tả. Nếu ma trận thiếu YCCĐ/bài cụ thể, cần rà soát bám sát SGK trước khi dùng chính thức."
        return {"stem": stem, "options": options, "answer": ans, "explanation": expl, "rubric": ""}
    else:
        stem = f"[{subject} Lớp {grade} – {level_phrase}] Trình bày ngắn gọn theo yêu cầu: {content}."
        rubric = f"Rubric ({points} điểm):\n- Đúng trọng tâm theo nội dung/bài học: {points*0.6:.2f} điểm\n- Trình bày rõ ràng, logic: {points*0.4:.2f} điểm"
        return {"stem": stem, "options": None, "answer": "Theo rubric", "explanation": "", "rubric": rubric}

def generate_exam_from_spec(spec_df: pd.DataFrame, allow_templates: bool = True) -> Exam:
    df = spec_df.copy().reset_index(drop=True)
    rng = random.Random(_seed_from_spec(df))

    subject = str(df.at[0, "subject"])
    grade = str(df.at[0, "grade"])
    term = str(df.at[0, "term_or_exam"])

    questions: List[QuestionItem] = []
    q_no = 1
    for ridx, row in df.iterrows():
        qty = int(row.get("quantity", 1) or 1)
        content = str(row.get("content_scope", "") or "")
        level = str(row.get("level", "UNKNOWN") or "UNKNOWN").upper()
        item_type = str(row.get("item_type", "UNKNOWN") or "UNKNOWN").upper()
        ppi = float(row.get("points_per_item", 0.5) or 0.5)

        for _ in range(qty):
            gen = _generate_one(content, subject, grade, level, item_type, ppi, rng) if allow_templates else {"stem": f"Câu hỏi: {content}", "options": None, "answer": "", "explanation": "", "rubric": ""}

            questions.append(QuestionItem(
                q_no=q_no,
                spec_row_id=ridx + 1,
                level=level,
                item_type=item_type,
                points=ppi,
                stem=gen["stem"],
                options=gen["options"],
                answer=gen["answer"],
                explanation=gen["explanation"],
                rubric=gen["rubric"],
            ))
            q_no += 1

    return Exam(subject=subject, grade=grade, term_or_exam=term, questions=questions)
