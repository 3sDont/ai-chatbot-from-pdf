# src/components/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device='cpu') # Chạy trên CPU cho tương thích

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
