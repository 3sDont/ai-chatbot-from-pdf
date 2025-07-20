# app.py

import streamlit as st
import os

# --- CÁC LỆNH IMPORT ĐÃ ĐƯỢC SỬA ---
# Không còn 'src.' ở đầu nữa vì các file đã ngang hàng
from pdf_reader import PDFReader
from text_splitter import TextSplitter
from embedder import Embedder
from vector_store import VectorStore
from llm_model import LLMModel
from rag_chain import RAGChain

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="📘 AI Chatbot từ PDF", layout="wide")
st.title("📘 AI Chatbot Hỗ Trợ Học Tập từ PDF")
st.markdown("Trợ lý ảo có khả năng đọc file PDF và trả lời câu hỏi dựa trên nội dung tài liệu bạn cung cấp.")

# --- CACHING CÁC MODEL TỐN KÉM TÀI NGUYÊN ---
@st.cache_resource
def load_llm_model():
    st.info("⏳ Đang tải mô hình ngôn ngữ (LLM)... Lần đầu có thể mất vài phút.")
    model = LLMModel(model_name="vinai/PhoGPT-4B-Chat")
    return model

@st.cache_resource
def load_embedding_model():
    st.info("⏳ Đang tải mô hình Embedding...")
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedder

llm = load_llm_model()
embedder = load_embedding_model()

# --- QUẢN LÝ TRẠNG THÁI SESSION ---
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
            with st.spinner("📖 Đang đọc và xử lý tài liệu..."):
                try:
                    # Chuyển đổi file upload thành đối tượng Class
                    pdf_reader_instance = PDFReader()
                    text_content = pdf_reader_instance.read(uploaded_file)
                    
                    if not text_content or not text_content.strip():
                        st.error("Không thể trích xuất nội dung từ file PDF. Vui lòng thử file khác.")
                    else:
                        text_splitter_instance = TextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = text_splitter_instance.split(text_content)
                        
                        vector_store = VectorStore()
                        embeddings = embedder.embed_documents(chunks)
                        vector_store.add_embeddings(embeddings, chunks)

                        st.session_state.rag_chain = RAGChain(embedder, vector_store, llm)
                        st.session_state.messages = []
                        st.success("✅ Tài liệu đã được xử lý xong! Bạn có thể bắt đầu trò chuyện.")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")

# --- GIAO DIỆN CHAT ---
st.header("💬 Bắt đầu trò chuyện")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain is None:
        with st.chat_message("assistant"):
            st.warning("Vui lòng tải lên và nhấn nút 'Xử lý tài liệu' trước khi đặt câu hỏi.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang suy nghĩ..."):
                response = st.session_state.rag_chain.query(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
