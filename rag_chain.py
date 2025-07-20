# src/rag_chain.py

from embedder import Embedder
from vector_store import VectorStore
from llm_model import LLMModel

class RAGPipeline:
    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def ask(self, question, top_k=3):
        q_vector = self.embedder.encode([question])[0]
        contexts = self.vector_store.search(q_vector, top_k=top_k)
        combined_context = "\n\n".join(contexts)
        return self.llm.generate_answer(combined_context, question)
