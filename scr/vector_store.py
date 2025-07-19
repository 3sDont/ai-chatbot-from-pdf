# src/vector_store.py

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):
        self.embeddings.extend(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_vector, top_k=3):
        if not self.embeddings:
            return []
        embedding_matrix = np.array(self.embeddings)
        query_vector = np.array(query_vector).reshape(1, -1)
        similarities = cosine_similarity(query_vector, embedding_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.text_chunks[i] for i in top_indices]

    def save(self, path="embedding_store.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "text_chunks": self.text_chunks
            }, f)

    def load(self, path="embedding_store.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.text_chunks = data["text_chunks"]
