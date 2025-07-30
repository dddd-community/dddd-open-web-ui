import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeFaiss:
    def __init__(self, documents: list[str], embedding_model: str = 'all-MiniLM-L6-v2'):
        self.documents = documents
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.embeddings = None
        self.build_index()

    def build_index(self):
        print("[KnowledgeFaiss] build...")
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True).astype('float32')
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print(f"[KnowledgeFaiss] build success,documents: {len(self.documents)}")

    def query(self, question: str, top_k: int = 3) -> list[str]:
        query_vec = self.model.encode([question], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        results = [self.documents[i] for i in indices[0]]
        return results
