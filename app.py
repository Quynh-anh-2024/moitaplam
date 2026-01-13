import streamlit as st
import pandas as pd
from src.matrix_parser import parse_matrix_file
from src.spec_builder import build_spec_from_tables
from src.validators import validate_and_repair_spec
from src.generator import generate_exam_from_spec
from src.export_docx import export_exam_docx

st.set_page_config(page_title="Tool tạo đề từ Ma trận", layout="wide")

st.title("Tạo đề từ Ma trận (Upload Excel/Word/PDF)")
st.caption("Chế độ Guarantee Mode: hệ thống luôn tạo đề và xuất kèm báo cáo đối soát. Khuyến nghị: dùng ma trận Excel để đạt độ chính xác cao nhất.")

with st.sidebar:
    st.header("Thiết lập")
    total_points_target = st.number_input("Tổng điểm mục tiêu", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    default_point_step = st.selectbox("Bước điểm mặc định", [0.25, 0.5, 1.0], index=1)
    allow_templates = st.checkbox("Sinh câu theo mẫu (không cần kho câu hỏi)", value=True)
    st.divider()
    st.subheader("Thông tin đầu đề")
    school_name = st.text_input("Tên trường (tuỳ chọn)", value="")
    exam_title = st.text_input("Tiêu đề đề", value="ĐỀ KIỂM TRA ĐỊNH KỲ")
    time_limit = st.text_input("Thời gian làm bài", value="Thời gian: 40 phút")

uploaded = st.file_uploader("Upload ma trận (xlsx/docx/pdf)", type=["xlsx", "xls", "docx", "pdf"])

if uploaded is None:
    st.info("Hãy upload một file ma trận để bắt đầu.")
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name

with st.spinner("Đang trích xuất nội dung từ file..."):
    tables, meta_text = parse_matrix_file(filename, file_bytes)

st.subheader("1) Kết quả trích xuất thô")
st.write(f"- File: `{filename}`")
if meta_text.strip():
    with st.expander("Văn bản meta trích xuất (tham khảo)"):
        st.write(meta_text[:5000])

if not tables:
    st.error("Không trích xuất được bảng nào từ file. Hãy thử xuất ma trận sang Excel hoặc Word (bảng) rồi upload lại.")
    st.stop()

st.write(f"Phát hiện **{len(tables)}** bảng.")
with st.expander("Xem nhanh các bảng (tối đa 2 bảng đầu)"):
    for i, t in enumerate(tables[:2], start=1):
        st.markdown(f"**Bảng {i}**")
        st.dataframe(t, use_container_width=True)

with st.spinner("Đang chuẩn hóa ma trận thành đặc tả (Spec)..."):
    spec_df, mapping_report = build_spec_from_tables(tables, meta_text)

st.subheader("2) Đặc tả (Spec) sau chuẩn hóa")
st.dataframe(spec_df, use_container_width=True)

with st.expander("Báo cáo map cột / suy diễn"):
    st.json(mapping_report)

with st.spinner("Đang kiểm định và tự sửa (Repair Rules)..."):
    spec_df2, validation_report = validate_and_repair_spec(
        spec_df,
        total_points_target=float(total_points_target),
        point_step=float(default_point_step),
    )

st.subheader("3) Spec sau khi validator + repair")
st.dataframe(spec_df2, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Báo cáo kiểm định")
    st.json(validation_report)
with col2:
    st.markdown("### Thống kê nhanh")
    st.write({
        "Tổng số dòng Spec": int(spec_df2.shape[0]),
        "Tổng số câu": int(spec_df2["quantity"].fillna(0).sum()),
        "Tổng điểm": float(spec_df2["row_points"].fillna(0).sum()),
        "Confidence": validation_report.get("confidence", 0.0),
    })

st.subheader("4) Sinh đề")
if st.button("Sinh đề theo Spec", type="primary"):
    with st.spinner("Đang sinh đề (Guarantee Mode)..."):
        exam = generate_exam_from_spec(spec_df2, allow_templates=allow_templates)

    docx_bytes = export_exam_docx(
        exam=exam,
        spec_df=spec_df2,
        mapping_report=mapping_report,
        validation_report=validation_report,
        school_name=school_name.strip(),
        exam_title=exam_title.strip(),
        time_limit=time_limit.strip(),
    )

    st.success("Đã sinh đề.")
    st.download_button(
        label="Tải Word (Đề + Đáp án/HDC + Spec + Đối soát)",
        data=docx_bytes,
        file_name="de_tu_ma_tran.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    spec_csv = spec_df2.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Tải Spec (CSV)",
        data=spec_csv,
        file_name="spec_chuan_hoa.csv",
        mime="text/csv",
    )
