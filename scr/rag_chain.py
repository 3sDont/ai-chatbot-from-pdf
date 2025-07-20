# src/rag_chain.py
from .embedder import Embedder
from .vector_store import VectorStore
from .llm_model import LLMModel

class RAGChain:
    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def query(self, question, top_k=3):
        # 1. Tạo embedding cho câu hỏi
        q_vector = self.embedder.embed_query(question)

        # 2. Tìm kiếm các chunks liên quan trong vector store
        contexts = self.vector_store.search(q_vector, top_k=top_k)
        
        # 3. Kết hợp các contexts lại
        combined_context = "\n\n---\n\n".join(contexts)
        
        # 4. Đưa context và câu hỏi cho LLM để tạo câu trả lời
        return self.llm.generate_answer(combined_context, question)
