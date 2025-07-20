# app.py

import streamlit as st
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pdf_reader import PDFReader
from src.text_splitter import TextSplitter
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_model import LLMModel
from src.rag_chain import RAGChain

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="📘 AI Chatbot từ PDF", layout="wide")
st.title("📘 AI Chatbot Hỗ Trợ Học Tập từ PDF")
st.markdown("Trợ lý ảo có khả năng đọc file PDF và trả lời câu hỏi dựa trên nội dung tài liệu bạn cung cấp.")

# --- CACHING CÁC MODEL TỐN KÉM TÀI NGUYÊN ---
@st.cache_resource
def load_llm_model():
    """Tải và cache mô hình LLM. Chỉ chạy một lần duy nhất."""
    st.write("⏳ Đang tải mô hình ngôn ngữ (LLM)... Lần đầu có thể mất vài phút.")
    # Sử dụng model vinai/PhoGPT-4B-Chat vì nó tốt cho tiếng Việt
    # trust_remote_code=True là cần thiết cho một số model
    model = LLMModel(model_name="vinai/PhoGPT-4B-Chat")
    st.write("✅ Đã tải xong mô hình LLM.")
    return model

@st.cache_resource
def load_embedding_model():
    """Tải và cache mô hình Embedding. Chỉ chạy một lần duy nhất."""
    st.write("⏳ Đang tải mô hình Embedding...")
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.write("✅ Đã tải xong mô hình Embedding.")
    return embedder

# Tải các model
llm = load_llm_model()
embedder = load_embedding_model()

# --- QUẢN LÝ TRẠNG THÁI SESSION ---
# Dùng st.session_state để lưu trữ dữ liệu giữa các lần chạy lại script
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- GIAO DIỆN UPLOAD VÀ XỬ LÝ ---
with st.sidebar:
    st.header("Tài liệu của bạn")
    uploaded_file = st.file_uploader("📎 Tải lên tài liệu PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("Xử lý tài liệu"):
            with st.spinner("📖 Đang đọc và xử lý tài liệu... Việc này có thể mất một lúc."):
                try:
                    # 1. Đọc PDF
                    pdf_reader = PDFReader()
                    text_content = pdf_reader.read(uploaded_file)

                    # 2. Chia nhỏ văn bản
                    # (Lưu ý: Bạn đang dùng class TextSplitter từ code của bạn, không phải langchain)
                    text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split(text_content)

                    if not chunks or not any(chunk.strip() for chunk in chunks):
                         st.error("Không thể trích xuất nội dung văn bản từ file PDF. Vui lòng thử file khác.")
                    else:
                        # 3. Tạo embedding và lưu vào Vector Store
                        vector_store = VectorStore()
                        st.write("Tạo embeddings cho các đoạn văn bản...")
                        embeddings = embedder.embed_documents(chunks)
                        vector_store.add_embeddings(embeddings, chunks)

                        # 4. Tạo RAG chain và lưu vào session state để dùng lại
                        st.session_state.rag_chain = RAGChain(embedder, vector_store, llm)
                        
                        # Xóa lịch sử chat cũ khi có tài liệu mới
                        st.session_state.messages = []
                        st.success("✅ Tài liệu đã được xử lý xong! Bạn có thể bắt đầu trò chuyện.")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

# --- GIAO DIỆN CHAT ---
st.header("💬 Bắt đầu trò chuyện")

# Hiển thị các tin nhắn đã có trong lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input mới từ người dùng
if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    # Thêm tin nhắn của người dùng vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kiểm tra xem tài liệu đã được xử lý chưa
    if st.session_state.rag_chain is None:
        with st.chat_message("assistant"):
            st.warning("Vui lòng tải lên và nhấn nút 'Xử lý tài liệu' trước khi đặt câu hỏi.")
    else:
        # Tạo và hiển thị câu trả lời của bot
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang suy nghĩ..."):
                response = st.session_state.rag_chain.query(prompt)
                st.markdown(response)
        
        # Thêm câu trả lời của bot vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": response})
