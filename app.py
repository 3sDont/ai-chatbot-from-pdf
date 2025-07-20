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
    Hiển thị một file PDF trong Streamlit bằng cách nhúng nó vào iframe.
    """
    # Đọc file
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    # Nhúng vào iframe
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    # Hiển thị
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
if st.session_state.uploaded_file is None:
    st.info("Chào mừng bạn! Vui lòng tải lên một tài liệu PDF ở thanh bên để bắt đầu.")
else:
    # Tạo hai cột: một cho hiển thị PDF, một cho chatbot
    col1, col2 = st.columns([1, 1]) # Tỉ lệ 1:1

    # Cột 1: Hiển thị PDF
    with col1:
        st.subheader(f"📄 Nội dung tài liệu: {st.session_state.uploaded_file.name}")
        display_pdf(st.session_state.uploaded_file)

    # Cột 2: Giao diện Chat
    with col2:
        st.subheader("🤖 Chat với AI")
        
        # Vùng chứa tin nhắn
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Ô nhập liệu chat
        if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            if not st.session_state.document_processed:
                st.warning("Tài liệu chưa được xử lý xong. Vui lòng chờ.")
            else:
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("AI đang suy nghĩ..."):
                            response = st.session_state.rag_pipeline.query(prompt)
                            st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
