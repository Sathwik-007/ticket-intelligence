import faiss
import numpy as np

class TicketIndexer:
    def __init__(self, vector_dim: int = 384, use_hnsw: bool = True):
        """ Initialises the class whether to use hnsw """
        self.dim = vector_dim
        if use_hnsw:
            # Approximate Nearest Neighbor (HNSW Graph)
            # 32 represents the number of bi-directional links per node
            self.index = faiss.IndexHNSWFlat(vector_dim, 32)
        else:
            # Brute force search
            self.index = faiss.IndexFlatL2(vector_dim)

    def build_index(self, embeddings: np.ndarray):
        """Loads the dense vectors and builds the index."""
        embeddings_flattened = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_flattened)

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Search the query against the built index"""
        query_flattened = np.ascontiguousarray(query_vector, dtype=np.float32)
        distances, indices = self.index.search(query_flattened, k)
        return distances[0], indices[0]
        