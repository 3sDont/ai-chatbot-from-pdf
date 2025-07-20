# src/pipelines/rag_pipeline.py

from typing import List
from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.vector_store import VectorStore
from .llm_models import GroqLLM

class RAGPipeline:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore, llm: GroqLLM):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def setup_with_text(self, text_content: str):
        chunks = self.chunker.split(text_content)
        embeddings = self.embedder.embed_documents(chunks)
        self.vector_store.add_documents(chunks, embeddings)
        print(f"✅ Đã xây dựng xong Vector Store từ {len(chunks)} chunks.")

    def query(self, question: str) -> str:
        if self.vector_store.embeddings.size == 0:
            return "Lỗi: Chưa có tài liệu nào được xử lý."

        query_embedding = self.embedder.embed_query(question)
        relevant_chunks = self.vector_store.search(query_embedding, top_k=4)
        
        if not relevant_chunks:
            return "Tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn trong tài liệu."
        
        context = "\n\n---\n\n".join(relevant_chunks)
        
        return self.llm.generate(context, question)
