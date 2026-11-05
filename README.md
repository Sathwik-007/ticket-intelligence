## 📌 Overview

This project is a production-grade machine learning microservice designed to automate customer support ticket routing. Built with a "first-principles" engineering philosophy, the system was initially developed from scratch using pure NumPy and linear algebra to unpack the black box of NLP clustering. It was then scaled into a high-performance, decoupled architecture utilizing dense semantic embeddings, approximate nearest neighbor (ANN) graph search, and density-based noise filtering.

The system analyzes incoming customer queries, assigns them to highly specific semantic clusters (e.g., "Refunds", "Delayed Shipping"), calculates a mathematical confidence score, and retrieves the top 5 most similar historical cases in sub-millisecond time.

## 🚀 Business Impact & Engineering Highlights

* **Sub-Millisecond Retrieval:** Migrated from exact linear scan ($O(N)$) to a FAISS HNSW (Hierarchical Navigable Small World) index, enabling sub-millisecond semantic search across 27,000+ support utterances.
* **Automated Triage Boundary:** Implemented HDBSCAN density-based clustering to automatically isolate ambiguous, multi-intent, or gibberish tickets as "noise," preventing algorithmic misrouting and establishing a mathematically sound human-triage queue.
* **Semantic Coherence:** Benchmarked sparse lexical matching (TF-IDF) against dense contextual embeddings (`all-MiniLM-L6-v2`), proving a 41% improvement in cluster coherence (Silhouette Score/Adjusted Rand Index) by eliminating high-dimensional vocabulary overlap.
* **Decoupled Architecture:** Segregated heavy matrix operations and FAISS graph traversals into a standalone FastAPI backend, ensuring the client UI thread is never blocked during inference.

## 🧠 The Evolution Pipeline: Scratch to Production

To deeply understand the optimization surfaces and vector mathematics, this system was built in two distinct phases:

### Phase 1: The Scratch Implementation (First Principles)

* **Vectorization:** Implemented TF-IDF from scratch to understand sparse, lexical vector spaces.
* **Clustering:** Authored a custom K-Means clustering algorithm using pure Python and NumPy, featuring probabilistic K-Means++ centroid initialization, vectorized L2 distance computation, and inertia convergence tracking.
* **Limitations Unlocked:** Mathematically demonstrated the "curse of dimensionality" and memory bottleneck of broadcasting a `(27000, 20000)` float64 tensor in memory without batching.

### Phase 2: The Production Architecture (Current State)

* **Dense Embeddings:** Transitioned to Sentence Transformers to project tickets into a 384-dimensional continuous semantic space, capturing intent rather than exact word matches.
* **Dimensionality Reduction:** Replaced linear PCA with UMAP to preserve non-linear local neighborhood structures for accurate 2D topological mapping.
* **Serving:** Wrapped the optimized Scikit-Learn K-Means++ model and FAISS index into a RESTful API.

## 🏗️ Tech Stack

* **Machine Learning & NLP:** `sentence-transformers`, `scikit-learn`, `hdbscan`, `umap-learn`, `numpy`
* **Vector Search:** `faiss-cpu` (HNSW Graph Indexing)
* **Backend API:** `FastAPI`, `Uvicorn`, `Pydantic` (Data Validation)
* **Frontend Interface:** `Streamlit`, `Requests`, Custom HTML/CSS injection

## ⚙️ System Architecture

1. **Client (Streamlit):** Captures raw text input and sends a JSON payload via POST request.
2. **Inference Engine (FastAPI):**
* Encodes text on-the-fly using cached `all-MiniLM-L6-v2` weights.
* Computes squared Euclidean distance against trained K-Means centroids for routing assignment.
* Calculates a confidence ratio using the distance delta between the primary and secondary centroids.


3. **Retrieval Engine (FAISS):** Traverses the HNSW graph using the query vector to return the 5 nearest historical ticket matches in `< 1ms`.
4. **Response:** Returns JSON payload to the UI for rendering via custom CSS cards.

## 💻 Local Setup & Installation

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/Sathwik-007/ticket-intelligence.git
cd ticket-intelligence
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

**2. Generate Artifacts (Embeddings, Model, FAISS Index)**
*This will download the 27,000-row Bitext dataset and compile the FAISS `.index` file.*

```bash
python scripts/embeddings_and_index.py

```

**3. Launch the Backend Microservice (FastAPI)**

```bash
uvicorn app.api:app --reload --port 8000

```

*The Swagger documentation will be available at `http://localhost:8000/docs*`

**4. Launch the UI (Streamlit)**
*Open a new terminal window:*

```bash
streamlit run app/streamlit_app.py

```