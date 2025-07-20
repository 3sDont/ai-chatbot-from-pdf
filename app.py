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
st.set_page_config(page_title="📚 AI Chatbot", layout="centered", initial_sidebar_state="auto")
st.title("📚 AI Chatbot")
st.markdown("Trợ lý AI giúp bạn hỏi đáp và khai thác thông tin từ tài liệu của mình.")

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

# Khởi tạo pipeline và các biến session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# --- Giao diện Sidebar để Upload và Xử lý ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    
    uploaded_file = st.file_uploader("Tải lên tài liệu (PDF/MD)", type=["pdf", "md"])

    if uploaded_file:
        if st.session_state.get("last_file_name") != uploaded_file.name:
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.document_processed = False
            st.session_state.messages = [{"role": "assistant", "content": f"Chào bạn, tôi đã sẵn sàng để trả lời các câu hỏi về tài liệu '{uploaded_file.name}'."}]

            with st.status("⚙️ Đang xử lý tài liệu...", expanded=True) as status:
                status.write("Đang đọc file...")
                loader = DataLoader()
                content = loader.load_from_upload(uploaded_file)
                
                if content:
                    status.write("Đang phân tích và ghi nhớ nội dung...")
                    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
                    st.session_state.rag_pipeline.setup_with_text(content)
                    st.session_state.document_processed = True
                    status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
                else:
                    status.update(label="Lỗi đọc file", state="error")

# --- Giao diện Chat Chính (Một cột) ---
if not st.session_state.messages:
    st.info("Chào mừng bạn! Vui lòng tải lên một tài liệu ở thanh bên để bắt đầu.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
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
