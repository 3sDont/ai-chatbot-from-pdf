# app.py

import streamlit as st
import os
import sys

# Thêm thư mục gốc vào sys.path để import từ src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_loader import DataLoader
from src.components.chunker import Chunker
from src.components.embedder import Embedder
from src.components.vector_store import VectorStore
from src.pipelines.llm_models import GroqLLM
from src.pipelines.rag_pipeline import RAGPipeline

# --- Cấu hình trang ---
st.set_page_config(page_title="📚 AI Chatbot Pro", layout="wide", initial_sidebar_state="expanded")
st.title("📚 Hỏi đi BaDong trả lời cho")

# --- Khởi tạo các đối tượng (dùng cache để tối ưu) ---
@st.cache_resource
def initialize_models():
    """Tải model embedding và khởi tạo kết nối LLM."""
    try:
        embedder = Embedder()
        llm = GroqLLM()
        return embedder, llm
    except ValueError as e:
        st.error(f"Lỗi khởi tạo: {e}. Vui lòng kiểm tra API key trong Streamlit Secrets.")
        return None, None

def initialize_rag_pipeline(embedder, llm):
    """Khởi tạo pipeline RAG cho mỗi session."""
    return RAGPipeline(Chunker(), embedder, VectorStore(), llm)

# Tải model và kiểm tra lỗi
embedder_model, llm_model = initialize_models()
if not (embedder_model and llm_model):
    st.stop()

# Khởi tạo pipeline trong session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# --- Giao diện Sidebar để Upload và Xử lý ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    st.info(
        "**Bước 1:** Tải lên file PDF hoặc Markdown.\n\n"
        "**Bước 2:** Chờ xử lý xong và bắt đầu hỏi đáp!"
    )
    
    uploaded_file = st.file_uploader("Tải lên tài liệu", type=["pdf", "md"])

    if uploaded_file:
        # Xử lý chỉ khi có file mới được tải lên
        if st.session_state.get("last_file_name") != uploaded_file.name:
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.document_processed = False
            st.session_state.messages = [] # Reset chat khi có file mới

            with st.status("⚙️ Đang xử lý tài liệu...", expanded=True) as status:
                st.write("Đang đọc file...")
                loader = DataLoader()
                content = loader.load_from_upload(uploaded_file)
                
                if content:
                    st.write("Đang phân tích và ghi nhớ nội dung...")
                    # Reset pipeline để nạp dữ liệu mới
                    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
                    st.session_state.rag_pipeline.setup_with_text(content)
                    st.session_state.document_processed = True
                    status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
                else:
                    status.update(label="Lỗi đọc file", state="error")
    

# --- Giao diện Chat Chính ---
if not st.session_state.document_processed:
    st.info("Chào mừng bạn! Vui lòng tải lên một tài liệu ở thanh bên để bắt đầu.")
else:
    st.success(f"Đã sẵn sàng! Hỏi đáp về tài liệu: **{st.session_state.last_file_name}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Câu hỏi của bạn về tài liệu..."):
    if not st.session_state.document_processed:
        st.warning("Vui lòng tải lên và chờ xử lý tài liệu trước.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 AI đang suy nghĩ..."):
                response = st.session_state.rag_pipeline.query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
