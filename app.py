# app.py

import streamlit as st
from src.pdf_reader import PDFReader
from src.text_splitter import TextSplitter
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_model import LLMModel
from src.rag_chain import RAGChain
import os

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="📘 AI Chatbot từ PDF", layout="wide")
st.title("📘 AI Chatbot hỗ trợ học tập")
st.markdown("Trợ lý ảo có khả năng đọc file PDF và trả lời câu hỏi dựa trên nội dung tài liệu bạn cung cấp.")

# Upload file
uploaded_file = st.file_uploader("📎 Tải lên tài liệu PDF của bạn", type="pdf")

# Tạo các đối tượng pipeline
pdf_reader = PDFReader()
text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
embedder = Embedder()
vector_store = VectorStore()
llm = LLMModel()
rag_chain = RAGChain(embedder, vector_store, llm)

# Xử lý khi có file upload
if uploaded_file is not None:
    with st.spinner("📖 Đang đọc và xử lý tài liệu..."):
        text = pdf_reader.read(uploaded_file)
        chunks = text_splitter.split(text)
        embeddings = embedder.embed_documents(chunks)
        vector_store.add_embeddings(embeddings, chunks)
        vector_store.save()  # lưu vào embedding_store.pkl

    st.success("✅ Tài liệu đã được xử lý xong! Bạn có thể bắt đầu đặt câu hỏi.")

    # Khung hỏi đáp
    query = st.text_input("💬 Nhập câu hỏi của bạn về tài liệu:")

    if query:
        with st.spinner("🤖 Đang tạo câu trả lời..."):
            answer = rag_chain.query(query)
        st.markdown(f"**📌 Trả lời:** {answer}")
