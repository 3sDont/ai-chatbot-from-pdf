# src/vector_store.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):
        self.embeddings = list(embeddings)
        self.text_chunks = list(chunks)

    def search(self, query_vector, top_k=3):
        if not self.embeddings:
            return []
        embedding_matrix = np.array(self.embeddings)
        query_vector_reshaped = np.array(query_vector).reshape(1, -1)
        similarities = cosine_similarity(query_vector_reshaped, embedding_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.text_chunks[i] for i in top_indices]
