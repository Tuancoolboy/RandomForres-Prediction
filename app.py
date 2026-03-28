import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="RFM Random Forest Prediction",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("best_rf_ultimate.pkl")


def strategy_text(segment: str) -> str:
    strategies = {
        "Champions": "Duy trì nhóm VIP: ưu đãi độc quyền, chăm sóc cá nhân hóa, chương trình giới thiệu bạn bè.",
        "Loyal Customers": "Tăng gắn kết bằng loyalty points, combo ưu đãi và gợi ý sản phẩm phù hợp.",
        "New Customers": "Tập trung onboarding: ưu đãi cho đơn tiếp theo, chăm sóc sau mua trong 7-14 ngày.",
        "At Risk": "Kích hoạt chiến dịch giữ chân ngay: voucher có thời hạn, nhắc mua lại, liên hệ CSKH cá nhân hóa.",
        "Lost Customers": "Chạy chiến dịch tái kích hoạt chi phí thấp; nếu không phản hồi thì giảm ưu tiên ngân sách."
    }
    return strategies.get(segment, "Chưa có đề xuất cho nhóm này.")


st.title("Dự đoán phân khúc khách hàng với Random Forest")
st.markdown("Nhập 3 chỉ số **RFM** để dự đoán nhóm khách hàng và gợi ý chiến lược.")

model = load_model()

with st.form("rfm_form"):
    recency = st.number_input("Recency (số ngày từ lần mua gần nhất)", min_value=0, value=30, step=1)
    frequency = st.number_input("Frequency (số hóa đơn mua)", min_value=1, value=3, step=1)
    monetary = st.number_input("Monetary (tổng chi tiêu)", min_value=0.0, value=500.0, step=10.0)

    submitted = st.form_submit_button("Dự đoán")

if submitted:
    input_df = pd.DataFrame([
        {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
        }
    ])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.success(f"Nhóm khách hàng dự đoán: **{pred}**")
    st.info(strategy_text(pred))

    st.subheader("Xác suất theo từng nhóm")
    proba_df = pd.DataFrame({
        "Segment": model.classes_,
        "Probability": proba
    }).sort_values("Probability", ascending=False)

    st.dataframe(proba_df, use_container_width=True)

st.markdown("---")
st.caption("Model: best_rf_ultimate.pkl | Features: Recency, Frequency, Monetary")
