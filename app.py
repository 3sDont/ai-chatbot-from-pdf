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
    st.info("Đang tải các mô hình AI...")
    embedder = Embedder()
    llm = FlanT5()
    st.success("Mô hình AI đã sẵn sàng.")
    return embedder, llm

@st.cache_resource
def initialize_pipeline(_embedder, _llm):
    """Khởi tạo pipeline RAG."""
    # Các thành phần không cần model AI có thể khởi tạo ở đây
    chunker = Chunker()
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(chunker, _embedder, vector_store, _llm)
    return rag_pipeline

# Tải model và khởi tạo pipeline
embedder_model, llm_model = initialize_models()
rag_pipeline = initialize_pipeline(embedder_model, llm_model)

# --- XỬ LÝ TÀI LIỆU ---

# Sử dụng session_state để đánh dấu tài liệu đã được xử lý hay chưa
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

with st.sidebar:
    st.header("Tài liệu")
    # Thay vì upload, chúng ta chọn file đã xử lý
    # Trong thực tế, bạn có thể tạo một dropdown để chọn từ các file trong `documents/markdowns`
    processed_doc_path = "documents/markdowns/your_document.md" # <-- THAY TÊN FILE

    if st.button("Nạp và Xử lý Tài liệu"):
        if os.path.exists(processed_doc_path):
            with st.spinner("Đang nạp và xử lý tài liệu..."):
                loader = DataLoader()
                content = loader.load(processed_doc_path)
                rag_pipeline.setup_with_text(content)
                st.session_state.document_processed = True
            st.success("Tài liệu đã được nạp và sẵn sàng để hỏi đáp.")
        else:
            st.error(f"File không tồn tại: {processed_doc_path}. Vui lòng chạy script `preprocess_pdf.py` trước.")

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

    if not st.session_state.document_processed:
        st.warning("Vui lòng nhấn nút 'Nạp và Xử lý Tài liệu' ở thanh bên trước.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang suy nghĩ..."):
                response = rag_pipeline.query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
