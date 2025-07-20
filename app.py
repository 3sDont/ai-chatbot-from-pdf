# app.py

import streamlit as st
import os
import sys

# Thêm thư mục gốc vào sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import từ kiến trúc mới
from src.components.data_loader import DataLoader
from src.components.chunker import Chunker
from src.components.embedder import Embedder
from src.components.vector_store import VectorStore
from src.pipelines.llm_models import FlanT5
from src.pipelines.rag_pipeline import RAGPipeline

st.set_page_config(page_title="📚 AI Chatbot Siêu Cấp", layout="wide")
st.title("📚 AI Chatbot Siêu Cấp")

# --- KHỞI TẠO CÁC ĐỐI TƯỢNG (DÙNG CACHE) ---

@st.cache_resource
def initialize_models():
    """Tải các model AI nặng."""
    embedder = Embedder()
    llm = FlanT5()
    return embedder, llm

# Khởi tạo pipeline RAG một lần cho mỗi session
# Dùng session_state thay vì cache_resource để có thể reset pipeline khi đổi file
def initialize_rag_pipeline(embedder, llm):
    chunker = Chunker()
    vector_store = VectorStore()
    return RAGPipeline(chunker, embedder, vector_store, llm)

embedder_model, llm_model = initialize_models()
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)

# --- GIAO DIỆN UPLOAD VÀ XỬ LÝ ---
with st.sidebar:
    st.header("Tài liệu của bạn")
    st.info(
        "**Lựa chọn 1 (Nhanh):** Tải trực tiếp file PDF để có câu trả lời ngay.\n\n"
        "**Lựa chọn 2 (Chất lượng cao):** Sử dụng script `preprocess_pdf.py` trên máy của bạn để chuyển PDF thành Markdown, sau đó tải file Markdown lên."
    )
    
    uploaded_file = st.file_uploader("Tải lên file PDF hoặc Markdown", type=["pdf", "md"])

    if uploaded_file is not None:
        # Sử dụng session_state để lưu tên file và tránh xử lý lại không cần thiết
        if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.document_processed = False

        if not st.session_state.get("document_processed", False):
            with st.spinner("Đang nạp và xử lý tài liệu..."):
                loader = DataLoader()
                content = ""
                file_type = uploaded_file.type
                
                if "pdf" in file_type:
                    st.write("Đang xử lý file PDF (chế độ nhanh)...")
                    content = loader.load_from_upload(uploaded_file)
                elif "markdown" in file_type or "text" in file_type:
                    st.write("Đang xử lý file Markdown (chế độ chất lượng cao)...")
                    # Streamlit đọc file text/md dưới dạng string
                    content = uploaded_file.getvalue().decode("utf-8")
                
                if content:
                    # Khởi tạo lại pipeline để xóa dữ liệu cũ
                    st.session_state.rag_pipeline = initialize_rag_pipeline(embedder_model, llm_model)
                    st.session_state.rag_pipeline.setup_with_text(content)
                    st.session_state.document_processed = True
                    st.session_state.messages = [] # Xóa lịch sử chat cũ
                    st.success("Tài liệu đã được nạp và sẵn sàng để hỏi đáp.")
                else:
                    st.error("Không thể đọc nội dung từ file được tải lên.")

# --- GIAO DIỆN CHAT ---
st.header("💬 Trò chuyện với tài liệu")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.get("document_processed", False):
        st.warning("Vui lòng tải lên và chờ xử lý tài liệu trước khi đặt câu hỏi.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang suy nghĩ..."):
                response = st.session_state.rag_pipeline.query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
