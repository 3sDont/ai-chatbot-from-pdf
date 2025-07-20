# src/rag_chain.py

from .embedder import Embedder
from .vector_store import VectorStore
from .llm_model import LLMModel

class RAGChain:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, llm: LLMModel):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def query(self, question: str, top_k: int = 3) -> str:
        # 1. Tạo embedding cho câu hỏi
        q_vector = self.embedder.embed_query(question)

        # 2. Tìm kiếm các chunks liên quan
        contexts = self.vector_store.search(q_vector, top_k=top_k)

        # 3. Kết hợp context
        combined_context = "\n\n---\n\n".join(contexts)

        # 4. Tạo câu trả lời
        return self.llm.generate_answer(combined_context, question)
