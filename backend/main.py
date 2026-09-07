from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

try:
    from backend.hnsw_runtime import (
        EF_SEARCH,
        EMBEDDINGS_PATH,
        HNSWIndex,
        IDS_PATH,
        load_or_build_index,
    )
except ModuleNotFoundError:
    from hnsw_runtime import (
        EF_SEARCH,
        EMBEDDINGS_PATH,
        HNSWIndex,
        IDS_PATH,
        load_or_build_index,
    )

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = PROJECT_DIR / "data" / "processed" / "documents.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOP_K = 50
logger = logging.getLogger("vector_db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(10, ge=1, le=MAX_TOP_K)
    ef_search: int = Field(EF_SEARCH, ge=1, le=1000)


class AppState:
    index: Any = None
    embeddings = None
    ids = None
    documents: dict[int, dict] = {}
    model: SentenceTransformer | None = None


state = AppState()


def load_documents():
    documents = {}
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                document = json.loads(line)
                documents[int(document["id"])] = document
    return documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("========== Vector Database startup ==========")
    state.index, state.embeddings, state.ids, built = load_or_build_index()
    logger.info("[4/4] Loading cleaned documents: %s", DOCUMENTS_PATH)
    state.documents = load_documents()
    logger.info("[4/4] Loaded %d document records", len(state.documents))
    logger.info("[4/4] Loading embedding model: %s", MODEL_NAME)
    state.model = SentenceTransformer(MODEL_NAME)
    logger.info("========== HNSW ready: vectors=%d, built_now=%s ==========" , len(state.embeddings), built)
    yield
    state.index = None
    state.model = None


app = FastAPI(title="Vector Database API", description="Semantic search over AG News embeddings", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled API error on %s", request.url.path)
    return Response(
        content='{"error":true,"message":"An unexpected server error occurred."}',
        status_code=500,
        media_type="application/json",
    )


@app.get("/api/health", tags=["system"])
async def health():
    if state.index is None or state.embeddings is None:
        raise HTTPException(status_code=503, detail="Vector index is not ready.")
    return {"status": "ok", "vectors": len(state.embeddings), "dimension": int(state.embeddings.shape[1]), "index": "HNSW"}


def embed_query(query: str):
    try:
        return np.asarray(state.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0], dtype=np.float32)
    except Exception as exc:
        logger.exception("Query embedding failed")
        raise HTTPException(status_code=503, detail="Unable to generate a query embedding.") from exc


def search_index(payload: SearchRequest, exact: bool = False):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if state.index is None or state.model is None:
        raise HTTPException(status_code=503, detail="Search service is not ready.")

    query_vector = embed_query(payload.query.strip())
    start = time.perf_counter()
    if exact:
        scores = state.embeddings @ query_vector
        candidate_indices = np.argsort(scores)[::-1][: payload.top_k]
        raw_results = [{"id": int(state.ids[i]), "score": float(scores[i])} for i in candidate_indices]
    else:
        state.index.ef_search = payload.ef_search
        try:
            raw_results = state.index.search(query_vector, k=payload.top_k)
        except TypeError:
            raw_results = state.index.search(query_vector, top_k=payload.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    results = []
    for rank, result in enumerate(raw_results, 1):
        doc_id = int(result["id"])
        document = state.documents.get(doc_id, {})
        results.append({
            "rank": rank,
            "id": doc_id,
            "score": float(result["score"]),
            "text": document.get("text", "Document text unavailable."),
        })
    return {"query": payload.query.strip(), "results": results, "count": len(results), "search_type": "exact" if exact else "hnsw", "latency_ms": round(latency_ms, 3)}


@app.post("/api/search", tags=["search"])
async def search(payload: SearchRequest):
    return search_index(payload)


@app.post("/api/search/exact", tags=["search"])
async def exact_search(payload: SearchRequest):
    return search_index(payload, exact=True)
