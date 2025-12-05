# 🇻🇳 Vietnamese Sentiment Analysis Assistant

## 📖 Giới thiệu (Introduction)
Đây là đồ án môn học **Seminar Chuyên đề**. Ứng dụng là một trợ lý ảo giúp phân loại cảm xúc của các câu văn bản tiếng Việt (Tích cực / Tiêu cực / Trung tính) sử dụng mô hình Deep Learning Transformer (PhoBERT).

**Sinh viên thực hiện:** [Tên Của Bạn]
**GVHD:** [Tên Giảng Viên]

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
   git clone [Dán link GitHub của bạn vào đây]
   cd vn-sentiment-assistant
