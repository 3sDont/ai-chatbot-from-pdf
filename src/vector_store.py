# src/vector_store.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def add_documents(self, texts: List[str], embeddings: np.ndarray):
        """Lưu trữ các đoạn văn và embeddings tương ứng."""
        if len(texts) != len(embeddings):
            raise ValueError("Số lượng văn bản và embeddings phải bằng nhau.")
        self.documents = texts
        self.embeddings = np.array(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """Tìm kiếm các đoạn văn bản liên quan nhất."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Reshape query_embedding để thực hiện tính toán ma trận
        query_vec = query_embedding.reshape(1, -1)
        
        # Tính cosine similarity
        sim_scores = cosine_similarity(query_vec, self.embeddings)[0]
        
        # Lấy top_k chỉ số có điểm số cao nhất
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
