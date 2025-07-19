# src/vector_store.py

import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):
        """
        embeddings: List[np.array], dạng (n_chunks x dim)
        chunks: List[str] đoạn văn gốc
        """
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(chunks)

    def search(self, query_vector, top_k=3):
        """
        Trả về top_k đoạn văn giống nhất với vector truy vấn
        """
        D, I = self.index.search(np.array([query_vector]).astype("float32"), top_k)
        return [self.text_chunks[i] for i in I[0]]

    def save(self, path="faiss_index.bin", meta_path="text_chunks.pkl"):
        faiss.write_index(self.index, path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, path="faiss_index.bin", meta_path="text_chunks.pkl"):
        self.index = faiss.read_index(path)
        with open(meta_path, "rb") as f:
            self.text_chunks = pickle.load(f)
