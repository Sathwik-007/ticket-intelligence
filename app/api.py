from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss
import os

class TicketRequest(BaseModel):
    text: str

class SimilarTicket(BaseModel):
    text: str
    distance: float

class TicketResponse(BaseModel):
    cluster_id: int
    confidence: float
    similar_tickets: list[SimilarTicket]

app = FastAPI(title="Ticket Routing API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = None
kmeans_model = None
corpus = None
faiss_index = None

@app.on_event("startup")
async def load_artifacts():
    global embedder, kmeans_model, corpus, faiss_index
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Dynamically build the absolute path to ticket-intelligence/data/processed
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    with open(os.path.join(data_dir, "corpus.pkl"), "rb") as f:
        corpus = pickle.load(f)
        
    with open(os.path.join(data_dir, "kmeans_model.pkl"), "rb") as f:
        kmeans_model = pickle.load(f)
        
    faiss_index = faiss.read_index(os.path.join(data_dir, "faiss_full.index"))
    print(f"Backend artifacts loaded successfully from {data_dir}")

@app.post("/route-ticket", response_model=TicketResponse)
async def route_ticket(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty.")
        
    query_vector = embedder.encode([request.text], normalize_embeddings=True)
    
    sq_dists = np.sum((kmeans_model.cluster_centers_ - query_vector) ** 2, axis=1)
    assigned_cluster = int(np.argmin(sq_dists))
    
    sorted_dists = np.sort(sq_dists)
    confidence = float(1.0 - (sorted_dists[0] / sorted_dists[1])) if sorted_dists[1] > 0 else 1.0
    
    q_f32 = np.ascontiguousarray(query_vector, dtype=np.float32)
    distances, indices = faiss_index.search(q_f32, 5)
    
    similar_tickets = [
        SimilarTicket(text=corpus[idx], distance=float(dist)) 
        for dist, idx in zip(distances[0], indices[0])
    ]
    
    return TicketResponse(
        cluster_id=assigned_cluster,
        confidence=confidence,
        similar_tickets=similar_tickets
    )