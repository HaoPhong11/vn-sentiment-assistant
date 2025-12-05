# 🇻🇳 Vietnamese Sentiment Analysis Assistant

## 📖 Giới thiệu (Introduction)
Đây là đồ án môn học **Seminar Chuyên đề**. Ứng dụng là một trợ lý ảo giúp phân loại cảm xúc của các câu văn bản tiếng Việt (Tích cực / Tiêu cực / Trung tính) sử dụng mô hình Deep Learning Transformer (PhoBERT).

**Sinh viên thực hiện:** Nguyễn Hào Phong 

## 🚀 Tính năng chính (Features)
- **Phân loại cảm xúc:** Sử dụng model `wonrax/phobert-base-vietnamese-sentiment` đạt độ chính xác cao.
- **Xử lý ngôn ngữ tự nhiên:** Tự động chuẩn hóa tiếng Việt không dấu, viết tắt (VD: "rat vui" -> "rất vui").
- **Lưu trữ lịch sử:** Tự động lưu các câu đã phân tích vào cơ sở dữ liệu SQLite.
- **Giao diện trực quan:** Xây dựng bằng Streamlit, thân thiện với người dùng.

## 🛠 Cài đặt (Installation)

### Yêu cầu hệ thống
- Python 3.8 trở lên.

### Các bước cài đặt
1. Clone dự án về máy:
   ```bash
   git clone https://github.com/HaoPhong11/vn-sentiment-assistant.git
   cd vn-sentiment-assistant
2. Cài đặt thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
## 💻 Hướng dẫn sử dụng (Usage)
1. Chạy ứng dụng:
   ```bash
   streamlit run app.py
2. Truy cập:
   Mở trình duyệt tại địa chỉ: http://localhost:8501
3. Thao tác:
   Nhập câu tiếng Việt vào ô trống.
   Nhấn nút Phân tích ngay để xem kết quả.
## 📂 Cấu trúc thư mục
📦 vn-sentiment-assistant
 ┣ 📜 app.py                # Mã nguồn chính
 ┣ 📜 sentiment_history.db  # Database (Tự tạo khi chạy)
 ┣ 📜 requirements.txt      # Danh sách thư viện
 ┗ 📜 README.md             # Tài liệu hướng dẫn này

