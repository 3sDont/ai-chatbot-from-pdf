# app.py

import streamlit as st
from src.pdf_loader import PDFLoader
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_model import LLMModel
from src.rag_chain import RAGPipeline
import tempfile

st.set_page_config(page_title="AI Chatbot học từ PDF", layout="centered")
st.title("📚 AI Chatbot hỗ trợ học tập từ giáo trình PDF")

# --- Tải file PDF từ người dùng ---
uploaded_file = st.file_uploader("📄 Tải lên file giáo trình (.pdf)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Đang xử lý file PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # 1. Tách văn bản
        loader = PDFLoader(pdf_path)
        chunks = loader.split_pdf()

        # 2. Mã hóa & lưu vector
        embedder = Embedder()
        vectors = embedder.encode(chunks)
        vector_store = VectorStore()
        vector_store.add_documents(chunks, vectors)

        # 3. Load LLM
        llm = LLMModel()
        rag = RAGPipeline(embedder, vector_store, llm)

        st.success("✅ File đã xử lý xong! Bạn có thể đặt câu hỏi.")

        # --- Chat interface ---
        question = st.text_input("💬 Câu hỏi của bạn:")
        if question:
            with st.spinner("🤖 Đang suy nghĩ..."):
                answer = rag.ask(question)
                st.markdown("### ✅ Trả lời:")
                st.markdown(answer)
