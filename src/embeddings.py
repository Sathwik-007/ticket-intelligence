from sentence_transformers import SentenceTransformer
import numpy as np


sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

def embed_tickets(tickets: list[str]) -> np.ndarray:
    embeddings = sentence_transformer.encode(
        tickets,
        batch_size=14,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings