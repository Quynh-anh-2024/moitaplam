# Matrix Upload → Exam Generator (Guarantee Mode)

Ứng dụng Streamlit:
- Upload **ma trận**: Excel (.xlsx) / Word (.docx) / PDF (.pdf)
- Trích xuất bảng → chuẩn hóa thành **Spec**
- **Validator + Repair Rules**: cân tổng điểm, suy diễn khi thiếu dữ liệu
- Sinh **Đề + Đáp án/HDC + Spec + báo cáo đối soát** (Word)

## Chạy local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Community Cloud
- Push repo lên GitHub
- Streamlit Cloud → New app → chọn repo → `app.py`

## Ghi chú
- Excel cho độ chính xác cao nhất.
- PDF scan có thể trích bảng kém; nên xuất lại ma trận sang Excel/Word bảng.
- `src/generator.py` đang sinh câu theo **mẫu** để đảm bảo luôn tạo được đề.
  Khi bạn có **kho câu hỏi** (Firestore/Sheets), thay module này bằng lắp ghép từ kho.
