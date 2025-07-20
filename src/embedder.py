# src/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embeddings cho một danh sách các đoạn văn bản (documents/chunks).
        Tên phương thức này nhất quán với cách gọi từ app.py.
        """
        print(f"Embedding {len(texts)} a document.")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho một câu hỏi duy nhất (query).
        """
        print(f"Embedding a query.")
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
