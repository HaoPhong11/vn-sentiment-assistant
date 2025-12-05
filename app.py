import streamlit as st
import sqlite3
import pandas as pd
import re
from datetime import datetime

from networkx.algorithms.distance_measures import center
from transformers import pipeline

# --- CẤU HÌNH TRANG (Page Config phải ở dòng đầu tiên) ---
st.set_page_config(
    page_title="VinaSentiment AI",
    page_icon="🧠",
    layout="wide",  # Sử dụng chế độ màn hình rộng
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ĐỂ LÀM ĐẸP THÊM (Optional) ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #4F46E5;}
    .sub-header {font-size: 1.2rem; font-weight: 500; color: #64748B;}
    .result-card {padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; color: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);}
    .pos-card {background: linear-gradient(135deg, #10B981, #059669);} /* Xanh lá */
    .neg-card {background: linear-gradient(135deg, #EF4444, #DC2626);} /* Đỏ */
    .neu-card {background: linear-gradient(135deg, #6B7280, #4B5563);} /* Xám */
    /* Tùy chỉnh bảng lịch sử */
    [data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden; border: 1px solid #E2E8F0;}
</style>
""", unsafe_allow_html=True)


# --- 1. PHẦN DATABASE & NLP (GIỮ NGUYÊN LOGIC CŨ) ---
# (Mình rút gọn phần này để tập trung vào giao diện, logic y hệt code trước)
def init_db():
    conn = sqlite3.connect('sentiment_history.db');
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS sentiments (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, sentiment TEXT NOT NULL, score REAL, timestamp TEXT NOT NULL)''')
    conn.commit();
    conn.close()


def save_to_db(text, sentiment, score):
    conn = sqlite3.connect('sentiment_history.db');
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO sentiments (text, sentiment, score, timestamp) VALUES (?, ?, ?, ?)',
              (text, sentiment, score, timestamp))
    conn.commit();
    conn.close()


def load_history():
    conn = sqlite3.connect('sentiment_history.db')
    df = pd.read_sql_query(
        "SELECT text as 'Câu nhập', sentiment as 'Cảm xúc', timestamp as 'Thời gian' FROM sentiments ORDER BY id DESC LIMIT 50",
        conn)
    conn.close();
    return df


init_db()


def normalize_text(text):
    text = text.lower().strip()
    replace_dict = {"rat": "rất", "tot": "tốt", "tuyet": "tuyệt", "thich": "thích", "yeu": "yêu", "dep": "đẹp",
                    "ok": "ổn", "ngon": "ngon", "vui": "vui", "buon": "buồn", "chan": "chán", "te": "tệ", "do": "dở",
                    "xau": "xấu", "met": "mệt", "ghet": "ghét", "buc": "bực", "khong": "không", "ko": "không",
                    "k": "không", "qua": "quá", "lam": "lắm", "hom": "hôm", "nay": "nay", "bt": "bình thường","bth": "bình thường"}
    words = text.split()
    new_words = [replace_dict.get(word, word) for word in words]
    return " ".join(new_words)


def validate_input(text):
    """
    Kiểm tra xem đầu vào có hợp lệ không.
    Trả về: (Bool, String) -> (Hợp lệ hay không, Thông báo lỗi)
    """
    # 1. Kiểm tra rỗng
    if not text or not text.strip():
        return False, "Vui lòng nhập nội dung!"

    # 2. Kiểm tra độ dài tối thiểu (sau khi strip)
    if len(text.strip()) < 5:
        return False, "Câu quá ngắn, vui lòng nhập đầy đủ hơn (VD: 'Hôm nay trời đẹp')."

    # 3. Kiểm tra xem có phải toàn là số không (VD: 123456)
    if text.strip().isdigit():
        return False, "Vui lòng nhập văn bản, hệ thống không phân tích dãy số."

    # 4. Kiểm tra xem có chứa chữ cái không (Chặn trường hợp: "!!!???", "@#$%", "...")
    # Regex này tìm xem có ít nhất 1 ký tự chữ cái (a-z hoặc tiếng Việt) hay không
    contains_letters = re.search(r'[a-zA-ZđĐáàảãạăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý]', text)
    if not contains_letters:
        return False, "Câu nhập vào vô nghĩa hoặc toàn ký tự đặc biệt."

    return True, ""

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment")


try:
    with st.spinner("Đang khởi động AI Engine..."):
        classifier = load_model()
    model_ready = True
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    model_ready = False

# --- 2. GIAO DIỆN NGƯỜI DÙNG MỚI ---

# Header Section
st.markdown('<div class="main-header">🧠 VinaSentiment AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống phân loại cảm xúc tiếng Việt sử dụng mô hình Transformer (PhoBERT)</div>',
            unsafe_allow_html=True)
st.divider()

# Main Content - Sử dụng 2 cột
col1, col2 = st.columns([3, 2], gap="large")  # Cột trái rộng hơn cột phải một chút

with col1:
    st.subheader("📝 Nhập liệu")
    st.caption("Hỗ trợ tiếng Việt có dấu, không dấu và viết tắt thông dụng.")
    user_input = st.text_area("Nội dung văn bản:", height=180,
                              placeholder="Ví dụ: Hôm nay tôi rất vui, đồ ăn ngon tuyệt! \nHoặc: rat buon vi mon nay do qua...")

    analyze_button = st.button("🚀 Phân tích ngay", type="primary", use_container_width=True, disabled=not model_ready)

with col2:
    st.subheader("🎯 Kết quả phân tích")
    result_placeholder = st.empty()  # Tạo một chỗ trống để điền kết quả sau

    if analyze_button:
        # --- LOGIC MỚI: GỌI HÀM KIỂM TRA ĐẦU VÀO ---
        is_valid, error_message = validate_input(user_input)

        if not is_valid:
            # Nếu không hợp lệ -> Hiện lỗi ngay và dừng lại
            result_placeholder.warning(f"⚠️ {error_message}")
        else:
            # Nếu hợp lệ -> Mới bắt đầu chạy AI
            with result_placeholder.container():
                with st.spinner("AI đang đọc và suy nghĩ..."):
                    # Xử lý chuẩn hóa
                    cleaned_text = normalize_text(user_input)

                    # Gọi Model AI
                    result = classifier(cleaned_text)[0]
                    label_raw = result['label']
                    score = result['score']

                    # Mapping & Styling (Giữ nguyên code cũ phần này)
                    if label_raw == "POS":
                        final_label = "TÍCH CỰC (POSITIVE) 😄"
                        card_style = "pos-card"
                    elif label_raw == "NEG":
                        final_label = "TIÊU CỰC (NEGATIVE) 😔"
                        card_style = "neg-card"
                    else:
                        final_label = "TRUNG TÍNH (NEUTRAL) 😐"
                        card_style = "neu-card"

                    # Hiển thị kết quả (Giữ nguyên code cũ)
                    st.markdown(f"""
                            <div class="result-card {card_style}">
                                <h2 style="margin:0;">{final_label}</h2>
                            </div>
                        """, unsafe_allow_html=True)

                    st.metric(label="Độ tin cậy của mô hình", value=f"{score * 100:.1f}%", delta=None)

                    if cleaned_text != user_input.lower().strip():
                        with st.expander("ℹ️ Chi tiết xử lý ngôn ngữ"):
                            st.write("Hệ thống đã tự động chuẩn hóa đầu vào:")
                            st.code(cleaned_text, language="text")

                    save_to_db(user_input, final_label, score)
                    st.toast("Đã lưu kết quả vào lịch sử!", icon="✅")

    elif not model_ready:
        result_placeholder.info("Đang tải mô hình, vui lòng đợi giây lát...")
    else:
        # Trạng thái chờ ban đầu
        result_placeholder.info("👈 Nhập văn bản bên trái và nhấn nút để xem kết quả tại đây.")

st.divider()

# History Section
with st.container():
    st.subheader("📜 Lịch sử phân loại gần đây")
    history_df = load_history()

    if not history_df.empty:
        # === DÒNG QUAN TRỌNG CẦN THÊM ===
        # Chuyển đổi cột 'Thời gian' từ dạng Text sang dạng Datetime để Streamlit hiểu
        history_df['Thời gian'] = pd.to_datetime(history_df['Thời gian'])
        # ================================

        # Sử dụng data_editor để hiển thị bảng đẹp hơn dataframe thường
        st.data_editor(
            history_df,
            column_config={
                "Câu nhập": st.column_config.TextColumn(width="medium"),
                "Cảm xúc": st.column_config.TextColumn(width="small"),
                # Giờ đây Streamlit đã hiểu đây là datetime object
                "Thời gian": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
            },
            hide_index=True,
            use_container_width=True,
            disabled=True  # Chỉ đọc
        )
    else:
        st.text("Chưa có dữ liệu lịch sử.")

# --- SIDEBAR ---
with st.sidebar:
    # --- 1. Căn giữa hình ảnh ---
    # Tạo 3 cột: Cột giữa rộng hơn một chút để chứa ảnh
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:  # Đặt ảnh vào cột giữa
        st.image("https://cdn-icons-png.flaticon.com/512/2620/2620951.png", width=80)

    st.title("Thông tin đồ án")

    # --- 2. Thông tin xuống dòng ---
    # Lưu ý: Cuối mỗi dòng mình đã thêm 2 dấu cách (space) để Markdown hiểu là xuống dòng
    st.info(
        """
        **Môn học:** Seminar Chuyên đề  
        **Đề tài:** Trợ lý phân loại cảm xúc tiếng Việt  
        **GVHD:** Nguyễn Tuấn Đăng 
        **Sinh viên:** Nguyễn Hào Phong
        **MSSV:** 3121560070
        """
    )

    st.markdown("---")
    st.write("🛠 **Tech Stack:**")
    st.write("- Python & Streamlit")
    st.write("- Hugging Face (PhoBERT)")
    st.write("- SQLite")