<div align="center">

# 🧠 Vector DB from Scratch
### A hand-built HNSW vector database — no FAISS, no Pinecone, no shortcuts

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![No External Vector DB](https://img.shields.io/badge/Vector%20Index-Handwritten-critical?style=flat-square)](#)
[![Dataset](https://img.shields.io/badge/Dataset-AG%20News-blueviolet?style=flat-square)](#)

**119,921 documents · 384-dim embeddings · 3.27x faster than brute force · 84.4% Recall@10**

</div>

---

## 🎯 The Challenge

> *"Everybody imports a vector database and almost nobody can tell you what happens inside one."*

The brief: **don't import one — build one.** No FAISS. No Pinecone. No Chroma. No `sklearn.neighbors`. Just NumPy, a real text corpus, and a from-scratch approximate nearest-neighbor index that has to earn its speedup against brute-force ground truth.

This project answers that brief end-to-end: a handwritten **HNSW graph index**, benchmarked honestly against exact search, wrapped in a real API and a real UI.

---

## ✨ What It Does

Type a search query → get the most semantically similar AG News articles back, ranked, in milliseconds — while the app quietly runs the same query through brute-force search in parallel so you can *see* the accuracy/speed trade-off live, not just take it on faith.

```
 User Query
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   React UI   │────▶│   FastAPI    │────▶│ SentenceTransf.  │
│ (search box) │     │  /api/search │     │ query embedding  │
└─────────────┘     └──────┬───────┘     └────────┬─────────┘
                            │                       │
                            ▼                       ▼
                   ┌─────────────────┐   ┌────────────────────┐
                   │ Handwritten HNSW │   │  Exact Brute-Force  │
                   │      Index       │   │   (ground truth)    │
                   └────────┬────────┘   └──────────┬──────────┘
                            └──────────┬─────────────┘
                                       ▼
                        Ranked results + latency + Recall@10
```

---

## 📊 Benchmark — The Number That Matters

| Metric | Value |
|---|---:|
| Vectors indexed | **119,921** |
| Dimensions | 384 |
| HNSW params | `M=12`, `ef_construction=100`, `ef_search=200` |
| **Recall@10** | **84.40%** |
| HNSW latency (avg) | ~3.5 ms |
| Exact latency (avg) | ~11.5 ms |
| **Speedup** | **~3.27×** |

> Recall@10 ≠ accuracy. It means HNSW returned ~8.44 of the true top-10 nearest neighbors per query, on average — the honest cost of approximating.

---

## 🏗️ How It's Built

| Stage | Notebook | What happens |
|---|---|---|
| 1 | `01_data_exploration.ipynb` | Clean & inspect AG News |
| 2 | `02_embedding_collab.ipynb` | Generate 384-d `all-MiniLM-L6-v2` embeddings |
| 3 | `03_exact_search.ipynb` | Brute-force cosine search → ground truth |
| 4 | `04_fix_hnsw.ipynb` | Handwritten HNSW graph (NumPy only) |
| 5 | `05_benchmark.ipynb` | HNSW vs. exact — latency, speedup, Recall@10 |

The FastAPI backend imports the *same* HNSW class from step 4 — one implementation, no duplication.

<details>
<summary><b>⚠️ Design note: deletion</b></summary>

<br>

Deleting a node from an HNSW graph mid-flight is genuinely awkward — removing a well-connected node can fragment neighbor paths across every layer above it. This project treats the index as **append-only / rebuild-on-change** rather than shipping a half-correct delete that silently degrades recall.
</details>

---

## 🚀 Quick Start

**Backend**
```powershell
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```
→ `http://127.0.0.1:8000` · Docs at `/docs`

> First run builds the HNSW graph from 119,921 vectors (~a few minutes, progress printed every 5,000 vectors). Cached to `backend/data/hnsw_index.pkl` after that.

**Frontend**
```powershell
cd frontend
npm install
npm run dev
```
→ `http://localhost:5173`

**Try it**
```
Search: "wall street bears"
```
Watch HNSW and exact search race — latency, speedup, and Recall@10 update live.

---

## 🔌 API

<table>
<tr><td width="120"><code>GET</code></td><td><code>/api/health</code></td><td>Index status, vector count, dimension</td></tr>
<tr><td><code>POST</code></td><td><code>/api/search</code></td><td>HNSW approximate top-k search</td></tr>
<tr><td><code>POST</code></td><td><code>/api/search/exact</code></td><td>Brute-force ground-truth search</td></tr>
</table>

```json
// POST /api/search
{ "query": "wall street bears", "top_k": 10, "ef_search": 200 }
```

---

## 📁 Project Structure

```
vector_db/
├── backend/            → FastAPI + HNSW runtime
├── data/
│   ├── raw/            → AG News source
│   ├── processed/      → Cleaned documents (119,921)
│   └── embeddings/     → 384-d vectors + IDs
├── frontend/           → React + Tailwind search UI
└── notebooks/          → Build stages 01 → 05
```

---

## 🧩 What's Handwritten vs. What's Not

| Handwritten (the point of this project) | Off-the-shelf (not the point) |
|---|---|
| HNSW graph construction & search | `all-MiniLM-L6-v2` embedding model |
| Exact brute-force cosine search | FastAPI / React / Vite |
| Recall@10 + latency benchmarking | NumPy for array math |

**Explicitly not used:** FAISS, Pinecone, Chroma, `sklearn.neighbors`, `hnswlib`, Annoy.

---

<div align="center">

*Built to answer one question: what actually happens inside a vector database?*

</div>