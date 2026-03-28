# RFM Customer Segmentation with Random Forest

Dự án này xây dựng một ứng dụng **Streamlit** để dự đoán phân khúc khách hàng dựa trên mô hình **Random Forest** và bộ chỉ số **RFM**:

- **Recency**: Số ngày từ lần mua gần nhất
- **Frequency**: Số lần/hóa đơn mua hàng
- **Monetary**: Tổng chi tiêu

Ứng dụng cho phép nhập 3 chỉ số RFM, sau đó:
1. Dự đoán nhóm khách hàng (segment)
2. Hiển thị xác suất theo từng nhóm
3. Gợi ý chiến lược chăm sóc/marketing tương ứng

## Công nghệ sử dụng

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## Cấu trúc chính

- `app.py`: Ứng dụng Streamlit để nhập dữ liệu và dự đoán
- `best_rf_ultimate.pkl`: Mô hình Random Forest đã huấn luyện
- `requirements.txt`: Danh sách thư viện cần cài
- `ecommerce.csv`, `online_retail_clean.csv`, `rfm_result.csv`, `rfm_labeled.csv`: Dữ liệu phục vụ xử lý/huấn luyện
- `Warmup_02_project.ipynb`: Notebook phân tích và thử nghiệm

## Cách chạy dự án

### 1) Cài thư viện

```bash
pip install -r requirements.txt
```

### 2) Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy, mở đường dẫn local do Streamlit cung cấp (thường là `http://localhost:8501`).

## Mô tả đầu ra

Khi bấm **Dự đoán**, ứng dụng sẽ hiển thị:

- Nhóm khách hàng dự đoán (ví dụ: Champions, Loyal Customers, New Customers, At Risk, Lost Customers)
- Gợi ý chiến lược hành động theo nhóm
- Bảng xác suất của tất cả nhóm để hỗ trợ ra quyết định

## Ghi chú

- Mô hình hiện sử dụng đúng 3 đặc trưng: `Recency`, `Frequency`, `Monetary`.
- Đảm bảo file `best_rf_ultimate.pkl` nằm cùng thư mục với `app.py` trước khi chạy.
# RandomForres-Prediction
