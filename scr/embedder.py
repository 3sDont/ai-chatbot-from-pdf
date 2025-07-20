# src/embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Model này sẽ được cache bởi Streamlit trong app.py
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Nhận danh sách các đoạn văn, trả về danh sách vector embeddings.
        """
        print(f"Đang tạo embedding cho {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("✅ Tạo embedding thành công.")
        return embeddings

    def embed_query(self, text):
        """
        Tạo embedding cho một câu query duy nhất.
        """
        return self.model.encode(text)
