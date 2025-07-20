# src/pipelines/rag_pipeline.py

from typing import List
from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.vector_store import VectorStore
from .llm_models import GroqLLM 

class RAGPipeline:
    def __init__(self, chunker, embedder, vector_store, llm: GroqLLM):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def setup_with_text(self, text_content: str):
        """Xử lý một văn bản và xây dựng vector store."""
        chunks = self.chunker.split(text_content)
        embeddings = self.embedder.embed_documents(chunks)
        self.vector_store.add_documents(chunks, embeddings)
        print("✅ Đã xây dựng xong Vector Store từ văn bản.")

    def query(self, question: str) -> str:
        """Trả lời câu hỏi dựa trên RAG."""
        if self.vector_store.embeddings.size == 0:
            return "Lỗi: Vector Store chưa được xây dựng. Vui lòng chạy setup_with_text trước."

        query_embedding = self.embedder.embed_query(question)
        relevant_chunks = self.vector_store.search(query_embedding, top_k=3)
        
        if not relevant_chunks:
            return "Tôi không tìm thấy thông tin liên quan trong tài liệu."
        
        context = "\n\n---\n\n".join(relevant_chunks)
        
        return self.llm.generate(context, question)
