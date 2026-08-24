"""
Two-stage retrieval: bi-encoder (BGE) → cross-encoder rerank.

Used by rate_retrieval.py to compare base BGE, fine-tuned BGE, and base+rerank.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.finetune_store import load_corpus
from utils.torch_win import bootstrap_torch
from utils.session_log import get_logger

log = get_logger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_BASE = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_FINETUNED = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Documents", "Models", "bge-rag-finetuned-40-1ep")
)

Method = Literal["base_bge", "finetuned_bge", "base_bge_rerank"]


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    bi_score: float | None = None
    rerank_score: float | None = None
    heading: str = ""
    document: str = ""
    preview: str = ""


class RetrievalEngine:
    """Encode corpus once; run base, fine-tuned, or reranked search per query."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE,
        finetuned_path: str = DEFAULT_FINETUNED,
        reranker_model: str = DEFAULT_RERANKER,
        retrieve_n: int = 30,
        load_reranker: bool = True,
    ):
        bootstrap_torch()
        from sentence_transformers import CrossEncoder, SentenceTransformer

        corpus = load_corpus()
        self.chunk_ids = [c["id"] for c in corpus]
        self.chunk_texts = [c["text"] for c in corpus]
        self.id_to_meta = {c["id"]: c for c in corpus}
        self.retrieve_n = retrieve_n

        log.info("Loading base bi-encoder: %s", base_model)
        self.base = SentenceTransformer(base_model)
        log.info("Encoding corpus with base BGE…")
        self.base_embs = self._encode_corpus(self.base, self.chunk_texts)

        self.finetuned = None
        self.tuned_embs = None
        if finetuned_path and os.path.isdir(finetuned_path):
            log.info("Loading fine-tuned bi-encoder: %s", finetuned_path)
            self.finetuned = SentenceTransformer(finetuned_path)
            log.info("Encoding corpus with fine-tuned BGE…")
            self.tuned_embs = self._encode_corpus(self.finetuned, self.chunk_texts)
        else:
            log.warning("Fine-tuned model not found (skip): %s", finetuned_path)

        self.reranker = None
        if load_reranker:
            log.info("Loading cross-encoder reranker: %s", reranker_model)
            self.reranker = CrossEncoder(reranker_model)

    @staticmethod
    def _encode_corpus(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

    def _encode_query(self, model, question: str) -> np.ndarray:
        text = BGE_QUERY_PREFIX + question
        return model.encode([text], normalize_embeddings=True, show_progress_bar=False).astype("float32")[0]

    def _bi_search(
        self,
        query_vec: np.ndarray,
        chunk_embs: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        scores = (chunk_embs @ query_vec.T).flatten()
        k = min(max(top_k, self.retrieve_n), len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]

    def _rerank(self, question: str, candidates: list[tuple[str, float]], top_k: int) -> list[RetrievedChunk]:
        if self.reranker is None:
            raise RuntimeError("Reranker not loaded")
        if not candidates:
            return []
        pairs = [(question, self.id_to_meta[cid]["text"]) for cid, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]
        out: list[RetrievedChunk] = []
        for (cid, bi_score), rr_score in ranked:
            meta = self.id_to_meta.get(cid, {})
            text = meta.get("text", "")
            preview = text[:220].replace("\n", " ") + ("…" if len(text) > 220 else "")
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(rr_score),
                    bi_score=bi_score,
                    rerank_score=float(rr_score),
                    heading=meta.get("heading", ""),
                    document=meta.get("document", meta.get("stem", "")),
                    preview=preview,
                )
            )
        return out

    def _to_chunks(self, ranked: list[tuple[str, float]]) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for cid, score in ranked:
            meta = self.id_to_meta.get(cid, {})
            text = meta.get("text", "")
            preview = text[:220].replace("\n", " ") + ("…" if len(text) > 220 else "")
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=score,
                    bi_score=score,
                    heading=meta.get("heading", ""),
                    document=meta.get("document", meta.get("stem", "")),
                    preview=preview,
                )
            )
        return out

    def retrieve(self, question: str, method: Method, top_k: int = 5) -> list[RetrievedChunk]:
        if method == "base_bge":
            q = self._encode_query(self.base, question)
            hits = self._bi_search(q, self.base_embs, top_k)
            return self._to_chunks(hits[:top_k])

        if method == "finetuned_bge":
            if self.finetuned is None or self.tuned_embs is None:
                raise RuntimeError("Fine-tuned model not loaded")
            q = self._encode_query(self.finetuned, question)
            hits = self._bi_search(q, self.tuned_embs, top_k)
            return self._to_chunks(hits[:top_k])

        if method == "base_bge_rerank":
            q = self._encode_query(self.base, question)
            pool = self._bi_search(q, self.base_embs, self.retrieve_n)
            return self._rerank(question, pool, top_k)

        raise ValueError(f"Unknown method: {method}")
