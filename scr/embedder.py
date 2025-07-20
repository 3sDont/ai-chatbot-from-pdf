# src/embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Nhận danh sách các đoạn văn, trả về danh sách vector embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)
