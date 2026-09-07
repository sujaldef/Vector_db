from __future__ import annotations

import ast
import heapq
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "04_fix_hnsw.ipynb"
EMBEDDINGS_PATH = PROJECT_DIR / "data" / "embeddings" / "embeddings.npy"
IDS_PATH = PROJECT_DIR / "data" / "embeddings" / "ids.npy"
INDEX_PATH = Path(__file__).resolve().parent / "data" / "hnsw_index.pkl"
M = 12
EF_CONSTRUCTION = 100
EF_SEARCH = 200
logger = logging.getLogger("vector_db")


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def _load_notebook_cells(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def load_hnsw_class():
    namespace = {
        "__name__": __name__,
        "np": np,
        "heapq": heapq,
        "time": time,
        "RANDOM_SEED": 42,
        "cosine_similarity": cosine_similarity,
    }
    from dataclasses import dataclass, field

    namespace.update({"dataclass": dataclass, "field": field})
    cells = _load_notebook_cells(NOTEBOOK_PATH)
    markers = (
        "class HNSWNode:",
        "class HNSWIndex:",
        "HNSWIndex.search =",
        "def validate_graph(self):",
        "HNSWIndex.validate_graph =",
    )
    for source in cells:
        if any(marker in source for marker in markers):
            tree = ast.parse(source)
            exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    if "HNSWIndex" not in namespace:
        raise RuntimeError("HNSWIndex was not found in the reference notebook")
    if "HNSWNode" in namespace:
        globals()["HNSWNode"] = namespace["HNSWNode"]
    return namespace["HNSWIndex"]


HNSWIndex = load_hnsw_class()


def load_or_build_index():
    logger.info("[1/4] Loading embeddings: %s", EMBEDDINGS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    ids = np.load(IDS_PATH, mmap_mode="r")
    logger.info("[1/4] Embeddings loaded: %d vectors x %d dimensions", embeddings.shape[0], embeddings.shape[1])
    if INDEX_PATH.exists():
        logger.info("[2/4] Loading persisted HNSW cache: %s", INDEX_PATH)
        with INDEX_PATH.open("rb") as file:
            index = pickle.load(file)
        if len(index.vectors) == len(embeddings):
            logger.info("[2/4] Cached HNSW loaded: %d vectors", len(index.vectors))
            return index, embeddings, ids, False
        logger.warning("[2/4] Cache size does not match embeddings; rebuilding index")

    logger.info("[2/4] No usable cache found; building HNSW with M=%d, ef_construction=%d", M, EF_CONSTRUCTION)
    index = HNSWIndex(M=M, ef_construction=EF_CONSTRUCTION, ef_search=EF_SEARCH, seed=42)
    start = time.perf_counter()
    total = len(embeddings)
    for vector, doc_id in zip(embeddings, ids):
        index.insert(vector, int(doc_id))
        inserted = len(index.vectors)
        if inserted == 1 or inserted % 5_000 == 0 or inserted == total:
            elapsed = time.perf_counter() - start
            rate = inserted / elapsed if elapsed else 0.0
            remaining = (total - inserted) / rate if rate else 0.0
            logger.info(
                "[2/4] HNSW build progress: %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs",
                inserted,
                total,
                inserted / total * 100,
                elapsed,
                remaining,
            )
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[3/4] Saving HNSW cache: %s", INDEX_PATH)
    with INDEX_PATH.open("wb") as file:
        pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("[3/4] HNSW cache saved")
    return index, embeddings, ids, True
