# src/rag_chain.py
from typing import List

# Sử dụng import tương đối (dấu chấm ở đầu)
from .llm_model import LLMModel
from .vector_store import VectorStore
from .embedder import Embedder

class RAGChain:
    def __init__(self, llm: LLMModel, vector_store: VectorStore, embedder: Embedder):
        self.llm = llm
        self.vector_store = vector_store
        self.embedder = embedder

    def query(self, question: str, top_k: int = 3) -> str:
        """Thực hiện chuỗi RAG: embed, search, generate."""
        # 1. Embed câu hỏi
        query_embedding = self.embedder.embed_query(question)
        
        # 2. Tìm kiếm trong vector store
        relevant_chunks = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not relevant_chunks:
            return "Rất tiếc, tôi không tìm thấy bất kỳ thông tin nào liên quan trong tài liệu."
        
        # 3. Tạo context
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # 4. Tạo câu trả lời
        return self.llm.generate_answer(context, question)
