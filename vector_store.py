"""
Vector store using sentence-transformers + FAISS for semantic retrieval
over the SHL assessment catalog.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILE = Path("catalog.faiss")
META_FILE = Path("catalog_meta.pkl")

_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBED_MODEL_NAME)


def _build_document(item: dict) -> str:
    """Convert catalog item to searchable text."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        f"Test type: {item.get('test_type', '')}",
        f"Job levels: {', '.join(item.get('job_levels', []))}",
        f"Languages: {', '.join(item.get('languages', []))}",
        f"Remote testing: {'Yes' if item.get('remote_testing') else 'No'}",
        f"Adaptive: {'Yes' if item.get('adaptive_irt') else 'No'}",
    ]
    if item.get("duration_minutes"):
        parts.append(f"Duration: {item['duration_minutes']} minutes")
    return " | ".join(filter(None, parts))


class CatalogVectorStore:
    def __init__(self, catalog: list[dict]):
        self.catalog = catalog
        self.embedder = None
        self.index: Optional[faiss.Index] = None
        self.doc_texts: list[str] = []

    def build(self) -> None:
        """Build FAISS index from catalog."""
        logger.info(f"Building vector index for {len(self.catalog)} assessments...")
        self.embedder = _load_embedder()
        self.doc_texts = [_build_document(item) for item in self.catalog]
        embeddings = self.embedder.encode(self.doc_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
        self.index.add(embeddings)

        # Persist
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump((self.doc_texts, self.catalog), f)
        logger.info("Vector index built and saved.")

    def load(self) -> bool:
    """Try to load persisted index. Return True if successful."""
    if INDEX_FILE.exists() and META_FILE.exists():
        try:
            self.index = faiss.read_index(str(INDEX_FILE))

            with open(META_FILE, "rb") as f:
                self.doc_texts, self.catalog = pickle.load(f)

            logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors).")
            return True

        except Exception as e:
            logger.warning(f"Failed to load index: {e}")

    return False

    def search(self, query: str, k: int = 10, filters: Optional[dict] = None) -> list[dict]:
        """
        Semantic search over catalog.
        filters: dict with optional keys:
          - test_type_codes: list[str]  e.g. ["A", "P"]
          - remote_only: bool
          - adaptive_only: bool
          - job_level: str
          - language: str
          - max_duration: int  (minutes)
        Returns up to k matching assessments, sorted by relevance.
        """
        if self.index is None or self.embedder is None:
            raise RuntimeError("Index not built/loaded.")

        query_vec = self.embedder.encode([query], show_progress_bar=False)
        query_vec = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        # Retrieve more candidates then filter
        retrieve_k = min(len(self.catalog), k * 5)
        scores, indices = self.index.search(query_vec, retrieve_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.catalog[idx]
            if not _passes_filters(item, filters or {}):
                continue
            results.append({**item, "_score": float(score)})
            if len(results) >= k:
                break

        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact or fuzzy name lookup."""
        name_lower = name.lower()
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    def get_all(self) -> list[dict]:
        return self.catalog


def _passes_filters(item: dict, filters: dict) -> bool:
    if filters.get("remote_only") and not item.get("remote_testing"):
        return False
    if filters.get("adaptive_only") and not item.get("adaptive_irt"):
        return False
    if filters.get("test_type_codes"):
        codes = filters["test_type_codes"]
        if not any(c in item.get("test_type_codes", []) for c in codes):
            return False
    if filters.get("job_level"):
        jl = filters["job_level"].lower()
        item_levels = [lvl.lower() for lvl in item.get("job_levels", [])]
        if item_levels and not any(jl in lvl or lvl in jl for lvl in item_levels):
            return False
    if filters.get("language"):
        lang = filters["language"].lower()
        item_langs = [l.lower() for l in item.get("languages", [])]
        if item_langs and not any(lang in l for l in item_langs):
            return False
    if filters.get("max_duration") and item.get("duration_minutes"):
        if item["duration_minutes"] > filters["max_duration"]:
            return False
    return True
