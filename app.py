# app.py

import streamlit as st
import os
import sys
import base64 # Thư viện cần thiết để hiển thị PDF

# --- Thêm thư mục gốc vào sys.path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import từ kiến trúc mới ---
from src.components.data_loader import DataLoader
from src.components.chunker import Chunker
from src.components.embedder import Embedder
from src.components.vector_store import VectorStore
from src.pipelines.llm_models import GroqLLM
from src.pipelines.rag_pipeline import RAGPipeline

# --- Cấu hình trang ---
st.set_page_config(page_title="📚 AI Chatbot Pro", layout="wide", initial_sidebar_state="expanded")

# --- HÀM TIỆN ÍCH ĐỂ HIỂN THỊ PDF ---
def display_pdf(file):
    """
    Hiển thị một file PDF trong Streamlit bằng cách nhúng nó vào thẻ <embed>.
    Phương pháp này tương thích tốt hơn với chính sách bảo mật của trình duyệt.
    """
    # Đọc file
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    
    # Tạo HTML để nhúng PDF bằng thẻ <embed>
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf">'
    
    # Hiển thị HTML
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- KHỞI TẠO CÁC ĐỐI TƯỢNG (DÙNG CACHE) ---
@st.cache_resource
def initialize_models():
    try:
        embedder = Embedder()
        llm = GroqLLM()
        return embedder, llm
    except ValueError as e:
        st.error(f"Lỗi khởi tạo: {e}. Vui lòng kiểm tra API key trong Streamlit Secrets.")
        return None, None

def initialize_rag_pipeline(embedder, llm):
    return RAGPipeline(Chunker(), embedder, VectorStore(), llm)

# Tải model và kiểm tra lỗi
embedder_model, llm_model = initialize_models()
if not (embedder_model and llm_model):
    st.stop()

# Khởi tạo pipeline và các biến session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# --- GIAO DIỆN SIDEBAR ĐỂ UPLOAD ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    uploaded_file = st.file_uploader("Tải lên tài liệu PDF của bạn", type=["pdf"])

    if uploaded_file:
        # Xử lý chỉ khi có file mới được tải lên
        if st.session_state.uploaded_file is None or st.session_state.uploaded_file.name != uploaded_file.name:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.document_processed = False
            st.session_state.messages = [] # Reset chat khi có file mới

            with st.status("⚙️ Đang xử lý tài liệu...", expanded=True) as status:
                status.write("Đang đọc file...")
                loader = DataLoader()
                # Chỉ dùng luồng xử lý PDF nhanh cho giao diện tương tác này
                content = loader.load_from_upload(uploaded_file)
                
                if content:
                    status.write("Đang phân tích và ghi nhớ nội dung...")
                    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
                    st.session_state.rag_pipeline.setup_with_text(content)
                    st.session_state.document_processed = True
                    status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
                else:
                    status.update(label="Lỗi đọc file", state="error")
    
    st.markdown("---")
    st.markdown(
        "**Lưu ý:** Chức năng xem trước PDF và hỏi đáp nhanh hoạt động tốt nhất với các file PDF có text rõ ràng."
    )

# --- BỐ CỤC GIAO DIỆN CHÍNH (2 CỘT) ---
# Tách biệt hai cột ngay từ đầu
col1, col2 = st.columns([5, 4]) # Tăng không gian cho PDF một chút

# Cột 1: Hiển thị PDF
with col1:
    st.subheader("📄 Nội dung tài liệu")
    if st.session_state.uploaded_file is not None:
        display_pdf(st.session_state.uploaded_file)
    else:
        st.info("Nội dung tài liệu sẽ được hiển thị ở đây sau khi bạn tải lên.")

# Cột 2: Giao diện Chat
with col2:
    st.subheader("🤖 Chat với AI")
    
    if st.session_state.uploaded_file is None:
        st.info("Vui lòng tải lên một tài liệu để bắt đầu trò chuyện.")
    else:
        if not st.session_state.document_processed:
            st.warning("Đang chờ xử lý tài liệu...")
        else:
            st.success(f"Đã sẵn sàng! Hỏi đáp về: **{st.session_state.uploaded_file.name}**")

    # Vùng chứa tin nhắn
    chat_container = st.container(height=600) # Giới hạn chiều cao để cuộn
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Ô nhập liệu chat
    if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
        if st.session_state.document_processed:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("AI đang suy nghĩ..."):
                        response = st.session_state.rag_pipeline.query(prompt)
                        st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Vui lòng tải lên và chờ xử lý tài liệu trước.")
