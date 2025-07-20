# app.py

import streamlit as st
import os
import sys

# ====================================================================
# GIẢI QUYẾT VẤN ĐỀ IMPORT KHI DEPLOY
# Thêm thư mục gốc của dự án vào sys.path để Python tìm thấy 'src'
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ====================================================================

from src.pdf_reader import PDFReader
from src.text_splitter import TextSplitter
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_model import LLMModel
from src.rag_chain import RAGChain

st.set_page_config(page_title="📘 AI Chatbot từ PDF", layout="wide")
st.title("📘 AI Chatbot Hỗ Trợ Học Tập từ PDF")
st.markdown("Trợ lý ảo có khả năng đọc và hiểu nội dung từ tài liệu PDF bạn cung cấp.")

@st.cache_resource
def load_models():
    """Tải và cache tất cả các model nặng một lần duy nhất."""
    st.info("⏳ Đang tải các mô hình AI... Lần đầu có thể mất vài phút.")
    embedder = Embedder()
    # Dùng model mặc định trong LLMModel là google/flan-t5-base
    llm = LLMModel() 
    st.success("✅ Các mô hình đã sẵn sàng!")
    return embedder, llm

embedder, llm = load_models()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Tài liệu của bạn")
    uploaded_file = st.file_uploader("📎 Tải lên tài liệu PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("Xử lý tài liệu"):
            with st.spinner("📖 Đang đọc, phân tích và ghi nhớ tài liệu..."):
                try:
                    pdf_reader = PDFReader()
                    text_content = pdf_reader.read(uploaded_file)

                    if not text_content or not text_content.strip():
                        st.error("Không thể trích xuất nội dung từ file PDF. Vui lòng thử file khác.")
                    else:
                        text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = text_splitter.split(text_content)
                        
                        vector_store = VectorStore()
                        embeddings = embedder.embed_documents(chunks)
                        vector_store.add_documents(chunks, embeddings)

                        st.session_state.rag_chain = RAGChain(llm, vector_store, embedder)
                        st.session_state.messages = []
                        st.success("✅ Đã xử lý xong! Bạn có thể bắt đầu trò chuyện.")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")

st.header("💬 Bắt đầu trò chuyện")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain is None:
        st.warning("Vui lòng tải lên và xử lý một file PDF trước khi đặt câu hỏi.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang suy nghĩ..."):
                response = st.session_state.rag_chain.query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
