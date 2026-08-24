"""
Load embedding models + FAISS indexes once; reuse for every compare-chat query.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from utils.session_log import get_logger
from utils.torch_win import bootstrap_torch

log = get_logger(__name__)

_embedding_models: dict[str, object] = {}
_store: Optional["LibraryIndexStore"] = None


def get_embedding_model(model_name: str):
    if model_name not in _embedding_models:
        bootstrap_torch()
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model (once): %s", model_name)
        _embedding_models[model_name] = SentenceTransformer(model_name)
    return _embedding_models[model_name]


def get_llm_client():
    from services.llm_client import get_llm_client as _get_llm_client

    return _get_llm_client()


def get_groq_client():
    """Backward-compatible alias — returns the Ollama client."""
    return get_llm_client()


@dataclass
class LoadedIndex:
    stem: str
    index: faiss.Index
    fine_chunks: list
    group_embeddings: np.ndarray
    all_chunk_groups: list


def _encode_query(model, model_name: str, query: str) -> np.ndarray:
    if "bge" in model_name.lower():
        q = f"Represent this sentence for searching relevant passages: {query}"
        return model.encode([q], normalize_embeddings=True).astype("float32")
    return model.encode([query], normalize_embeddings=True).astype("float32")


def _flat_search(loaded: LoadedIndex, model, model_name: str, query: str, top_k: int):
    if loaded.index is None:
        return []
    query_vec = _encode_query(model, model_name, query)
    _k = min(top_k, len(loaded.fine_chunks))
    if _k <= 0:
        return []
    distances, indices = loaded.index.search(query_vec, _k)
    return [
        (loaded.fine_chunks[i], float(distances[0][j]))
        for j, i in enumerate(indices[0])
        if i >= 0
    ]


class LibraryIndexStore:
    """In-memory FAISS indexes for all library documents × embedding models."""

    def __init__(self):
        self.ready = False
        self.summary: dict[str, dict[str, LoadedIndex]] = {}
        self.extracted: dict[str, dict[str, LoadedIndex]] = {}
        self.models: dict[str, object] = {}

    def _scan_pairs(self, folder: str, summary: bool = False) -> list[tuple[str, str, str]]:
        if not os.path.isdir(folder):
            return []
        suffix = "_summary_faiss.index" if summary else "_faiss.index"
        meta_suffix = "_summary_meta.npz" if summary else "_meta.npz"
        pairs = []
        for name in os.listdir(folder):
            if not name.endswith(suffix):
                continue
            stem = name[: -len(suffix)]
            meta_path = os.path.join(folder, f"{stem}{meta_suffix}")
            if os.path.exists(meta_path):
                pairs.append((stem, os.path.join(folder, name), meta_path))
        return pairs

    def _load_one(self, stem: str, index_path: str, meta_path: str) -> LoadedIndex:
        index = faiss.read_index(index_path)
        data = np.load(meta_path, allow_pickle=True)
        return LoadedIndex(
            stem=stem,
            index=index,
            fine_chunks=list(data["fine_chunks"]),
            group_embeddings=data["group_embeddings"],
            all_chunk_groups=list(data["all_chunk_groups"]),
        )

    def warmup(self, force: bool = False) -> dict:
        if self.ready and not force:
            return self.stats()

        log.info("Warming retrieval cache (embedding models + FAISS indexes)...")
        self.summary.clear()
        self.extracted.clear()
        self.models.clear()

        counts = {"minilm": {"summary": 0, "extracted": 0}, "bge": {"summary": 0, "extracted": 0}}

        for model_key, model_name in Config.EMBEDDING_MODELS.items():
            self.models[model_key] = get_embedding_model(model_name)
            self.summary[model_key] = {}
            self.extracted[model_key] = {}

            for stem, index_path, meta_path in self._scan_pairs(
                Config.faiss_root(model_key, "summary"), summary=True
            ):
                self.summary[model_key][stem] = self._load_one(stem, index_path, meta_path)
                counts[model_key]["summary"] += 1

            for stem, index_path, meta_path in self._scan_pairs(
                Config.faiss_root(model_key, "extracted"), summary=False
            ):
                self.extracted[model_key][stem] = self._load_one(stem, index_path, meta_path)
                counts[model_key]["extracted"] += 1

        self.ready = True
        log.info(
            "Retrieval cache ready — "
            "MiniLM: %s summary / %s extracted, "
            "BGE: %s summary / %s extracted.",
            counts['minilm']['summary'],
            counts['minilm']['extracted'],
            counts['bge']['summary'],
            counts['bge']['extracted'],
        )
        return self.stats()

    def stats(self) -> dict:
        return {
            "ready": self.ready,
            "embedding_models": list(_embedding_models.keys()),
            "minilm_summary": len(self.summary.get("minilm", {})),
            "minilm_extracted": len(self.extracted.get("minilm", {})),
            "bge_summary": len(self.summary.get("bge", {})),
            "bge_extracted": len(self.extracted.get("bge", {})),
        }

    def search(
        self,
        model_key: str,
        query: str,
        top_summary_docs: int = 2,
        top_k_per_doc: int = 6,
        top_k_summary_hits: int = 3,
    ) -> list[dict]:
        if not self.ready:
            self.warmup()

        model_name = Config.EMBEDDING_MODELS[model_key]
        model = self.models[model_key]
        summary_indexes = self.summary.get(model_key, {})
        extracted_indexes = self.extracted.get(model_key, {})

        doc_scores = []
        for stem, loaded in summary_indexes.items():
            hits = _flat_search(loaded, model, model_name, query, top_k_summary_hits)
            if not hits:
                continue
            best_chunk, best_score = hits[0]
            doc_scores.append({"stem": stem, "document": stem, "score": best_score})

        doc_scores.sort(key=lambda x: x["score"], reverse=True)
        selected = doc_scores[:top_summary_docs]

        all_chunks = []
        for doc in selected:
            stem = doc["stem"]
            loaded = extracted_indexes.get(stem)
            if loaded is None:
                continue
            for chunk, score in _flat_search(loaded, model, model_name, query, top_k_per_doc):
                all_chunks.append(
                    {"document": stem, "chunk": chunk, "score": float(score)}
                )

        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        return all_chunks


def get_index_store() -> LibraryIndexStore:
    global _store
    if _store is None:
        _store = LibraryIndexStore()
    return _store


def warmup_retrieval(force: bool = False) -> dict:
    return get_index_store().warmup(force=force)


def invalidate_retrieval_cache() -> None:
    global _store
    if _store is not None:
        _store.ready = False
        _store.summary.clear()
        _store.extracted.clear()
