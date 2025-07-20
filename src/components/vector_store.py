# src/components/vector_store.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class VectorStore:
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def add_documents(self, texts: List[str], embeddings: np.ndarray):
        if len(texts) != len(embeddings):
            raise ValueError("Số lượng văn bản và embeddings phải bằng nhau.")
        self.documents.extend(texts)
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        if self.embeddings.size == 0:
            return []
        
        query_vec = query_embedding.reshape(1, -1)
        sim_scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
