# Vector Database

An end-to-end semantic search project built on the AG News dataset. Documents are cleaned, converted into 384-dimensional SentenceTransformer embeddings, indexed with a handwritten HNSW graph, and exposed through a FastAPI backend with a React + Tailwind frontend.

## What It Does

The application lets a user enter a natural-language query and retrieve the most similar AG News documents. Every query runs through:

1. React frontend
2. FastAPI API
3. SentenceTransformer query embedding
4. Existing handwritten HNSW index
5. Document ID and text lookup
6. Ranked results shown in the browser

The frontend also runs exact brute-force search for the same query. This makes the exact result set the ground truth and allows live comparison of latency, speedup, and Recall@10.

## Project Structure

```text
vector_db/
├── backend/
│   ├── main.py                  # FastAPI application and API routes
│   ├── hnsw_runtime.py          # Loads the handwritten HNSW implementation
│   ├── requirements.txt         # Backend dependencies
│   └── data/                    # Runtime HNSW cache, generated locally
├── data/
│   ├── raw/                     # Original AG News parquet data
│   ├── processed/               # Cleaned JSONL documents
│   └── embeddings/              # Existing .npy embeddings and IDs
├── frontend/
│   ├── src/App.jsx              # React search interface
│   ├── src/api/search.js        # API client
│   └── .env.example             # Frontend API URL configuration
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_collab.ipynb
│   ├── 03_exact_search.ipynb
│   ├── 04_fix_hnsw.ipynb
│   └── 05_benchmark.ipynb
└── README.md
```

## Dataset

- Dataset: AG News
- Cleaned documents: 119,921
- Embedding dimension: 384
- Embeddings: `data/embeddings/embeddings.npy`
- IDs: `data/embeddings/ids.npy`
- Documents: `data/processed/documents.jsonl`
- Embedding model: `all-MiniLM-L6-v2`

The original data and generated embeddings are read from disk. The API does not regenerate embeddings during normal startup.

## HNSW Benchmark

The strongest recorded benchmark configuration was:

| Setting               |                   Value |
| --------------------- | ----------------------: |
| Vectors               |                 119,921 |
| Dimensions            |                     384 |
| M                     |                      12 |
| ef_construction       |                     100 |
| ef_search             |                     200 |
| Recall@10             |                  84.40% |
| HNSW average latency  |  approximately 3.535 ms |
| Exact average latency | approximately 11.546 ms |
| Speedup               |     approximately 3.27x |

These values are recorded benchmark results, not values returned as fake search responses by the API.

**Recall@10 is not accuracy.** For each query, HNSW's top ten IDs are compared with the exact brute-force top ten IDs. A Recall@10 of 84.4% means that HNSW retrieved about 8.44 of the ten exact nearest neighbors on average for that benchmark.

## Requirements

- Python 3.10+
- Node.js 18+
- npm
- The existing `.venv` environment or a new Python virtual environment

## Backend Setup

From the project root:

```powershell
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

Alternatively, from inside `backend/`:

```powershell
..\.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

The backend runs at `http://127.0.0.1:8000`.

### First Startup

If `backend/data/hnsw_index.pkl` does not exist, the backend builds the HNSW graph once from the 119,921 existing vectors. This can take several minutes. The terminal prints progress every 5,000 vectors, including percentage complete, elapsed time, and estimated remaining time.

After the first successful build, the graph is saved to `backend/data/hnsw_index.pkl`. The cache is ignored by Git and loaded on later restarts. Do not delete it unless you intentionally want to rebuild the graph.

### API Documentation

FastAPI provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

#### Health

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "vectors": 119921,
  "dimension": 384,
  "index": "HNSW"
}
```

#### HNSW Search

```http
POST /api/search
Content-Type: application/json
```

Request:

```json
{
  "query": "wall street bears",
  "top_k": 10,
  "ef_search": 200
}
```

Response fields include the query, ranked results, document IDs, similarity scores, document text, result count, search type, and measured latency.

#### Exact Search

```http
POST /api/search/exact
Content-Type: application/json
```

This performs brute-force search over the full embedding matrix and is used as the comparison ground truth.

## Frontend Setup

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs at the URL printed by Vite, normally `http://localhost:5173`.

The frontend API URL can be configured in `frontend/.env`:

```text
VITE_API_URL=http://127.0.0.1:8000
```

The React interface provides:

- Natural-language semantic search
- Loading and retry states
- Empty-query validation
- Real HNSW results
- Exact-search comparison
- Live HNSW and exact latency
- Live speedup
- Live Recall@10
- Labels for HNSW results missing from the exact top ten
- Responsive layout

## Validation

Backend syntax check:

```powershell
.venv\Scripts\python.exe -m py_compile backend\main.py backend\hnsw_runtime.py
```

Frontend production build:

```powershell
cd frontend
npm run build
```

Manual checks:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/health
```

Then open the frontend and search for:

```text
wall street bears
```

The first request should return real document text from `documents.jsonl`, with IDs and scores generated by the backend.

## Notebooks

The notebooks document the project stages:

- `01_data_exploration.ipynb`: inspect and clean the AG News source data.
- `02_embedding_collab.ipynb`: generate and save normalized document embeddings.
- `03_exact_search.ipynb`: establish brute-force exact search as the ground truth.
- `04_fix_hnsw.ipynb`: develop and inspect the handwritten HNSW implementation.
- `05_benchmark.ipynb`: benchmark HNSW against exact search and generate project reports.

The API reuses the existing HNSW implementation from `04_fix_hnsw.ipynb` rather than maintaining a second algorithm implementation.

## Design Notes and Limitations

- HNSW is handwritten and uses NumPy plus Python data structures; no FAISS, Pinecone, Chroma, sklearn neighbor index, hnswlib, Annoy, or external vector database is used.
- The graph cache is a local pickle created by the backend. It is not portable across incompatible changes to the notebook-defined class.
- The API uses lazy loading at startup and keeps the index in memory for repeated searches.
- Exact search is available for comparison but is slower as the dataset grows.
- Benchmark statistics describe the recorded benchmark and are not guarantees for every machine or query.
- The source data, embeddings, IDs, and benchmark outputs should be treated as read-only project artifacts.

