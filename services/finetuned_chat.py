"""
Demo chat: base BGE vs fine-tuned BGE retrieval + LLM answers.

Uses the finetune corpus (all library chunks) with cached embeddings for both
models — same fair comparison as the evaluation scripts.

Usage (via API):
  POST /chat/finetuned
  GET  /chat/finetuned/status
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.compare_chat import (
    MAX_HISTORY_TURNS,
    _llm_answer,
    _retrieval_query,
)
from services.finetune_store import load_corpus
from services.library_pipeline import load_library_state
from utils.torch_win import bootstrap_torch

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

_store: Optional["FinetunedCorpusStore"] = None
_warmup_lock = threading.Lock()
_warming = False


class FinetunedCorpusStore:
    """Corpus-wide search with base and fine-tuned BGE embeddings."""

    def __init__(self):
        self.ready = False
        self.error: str | None = None
        self.base_model_name = Config.EMBEDDING_MODELS["bge"]
        self.finetuned_path = Config.FINETUNED_BGE_PATH
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.id_to_doc: dict[str, str] = {}
        self.base_embs: np.ndarray | None = None
        self.tuned_embs: np.ndarray | None = None
        self.base: object | None = None
        self.finetuned: object | None = None

    def warmup(self, force: bool = False) -> dict:
        if self.ready and not force:
            return self.stats()

        if not os.path.isdir(self.finetuned_path):
            self.error = f"Fine-tuned model not found: {self.finetuned_path}"
            raise FileNotFoundError(self.error)

        bootstrap_torch()
        from sentence_transformers import SentenceTransformer

        corpus = load_corpus()
        if not corpus:
            self.error = "Finetune corpus is empty. Run library_pipeline first."
            raise RuntimeError(self.error)

        self.chunk_ids = [c["id"] for c in corpus]
        self.chunk_texts = [c["text"] for c in corpus]
        self.id_to_doc = {
            c["id"]: c.get("document") or c.get("stem", "")
            for c in corpus
        }

        print(f"[finetuned-chat] Loading base BGE: {self.base_model_name}")
        self.base = SentenceTransformer(self.base_model_name)
        print(f"[finetuned-chat] Encoding {len(self.chunk_texts)} chunks (base)…")
        self.base_embs = self._encode(self.base, self.chunk_texts)

        print(f"[finetuned-chat] Loading fine-tuned BGE: {self.finetuned_path}")
        self.finetuned = SentenceTransformer(self.finetuned_path)
        print(f"[finetuned-chat] Encoding {len(self.chunk_texts)} chunks (fine-tuned)…")
        self.tuned_embs = self._encode(self.finetuned, self.chunk_texts)

        self.ready = True
        self.error = None
        print("[finetuned-chat] Ready — base vs fine-tuned corpus search enabled.")
        return self.stats()

    @staticmethod
    def _encode(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

    def _encode_query(self, model, query: str) -> np.ndarray:
        text = BGE_PREFIX + query
        return model.encode([text], normalize_embeddings=True, show_progress_bar=False).astype("float32")[0]

    def _search(self, query_vec: np.ndarray, embs: np.ndarray, top_k: int) -> list[dict]:
        scores = (embs @ query_vec.T).flatten()
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out = []
        for i in top_idx:
            cid = self.chunk_ids[i]
            out.append({
                "chunk_id": cid,
                "document": self.id_to_doc.get(cid, cid.split("::")[0]),
                "chunk": self.chunk_texts[i],
                "score": float(scores[i]),
            })
        return out

    def search_base(self, query: str, top_k: int = 8) -> list[dict]:
        q = self._encode_query(self.base, query)
        return self._search(q, self.base_embs, top_k)

    def search_finetuned(self, query: str, top_k: int = 8) -> list[dict]:
        q = self._encode_query(self.finetuned, query)
        return self._search(q, self.tuned_embs, top_k)

    def stats(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "chunks": len(self.chunk_ids),
            "base_model": self.base_model_name,
            "finetuned_model": self.finetuned_path,
            "finetuned_exists": os.path.isdir(self.finetuned_path),
        }


def get_finetuned_store() -> FinetunedCorpusStore:
    global _store
    if _store is None:
        _store = FinetunedCorpusStore()
    return _store


def warmup_finetuned_chat(force: bool = False) -> dict:
    return get_finetuned_store().warmup(force=force)


def warmup_finetuned_chat_async(force: bool = False) -> None:
    global _warming

    def _run():
        global _warming
        try:
            warmup_finetuned_chat(force=force)
        except Exception as e:
            store = get_finetuned_store()
            store.error = str(e)
            print(f"[finetuned-chat] Warmup failed: {e}")
        finally:
            _warming = False

    with _warmup_lock:
        store = get_finetuned_store()
        if store.ready and not force:
            return
        if _warming:
            return
        _warming = True
        threading.Thread(target=_run, daemon=True).start()


def _normalize_finetuned_history(history: Optional[list]) -> list[dict]:
    if not history:
        return []
    turns = []
    for item in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        if not q:
            continue
        turns.append({
            "question": q,
            "base_answer": (item.get("base_answer") or item.get("bge_answer") or "").strip(),
            "finetuned_answer": (item.get("finetuned_answer") or "").strip(),
            # aliases for shared helpers
            "minilm_answer": (item.get("base_answer") or "").strip(),
            "bge_answer": (item.get("finetuned_answer") or "").strip(),
        })
    return turns


def _ensure_store_ready(timeout: float = 180) -> FinetunedCorpusStore:
    store = get_finetuned_store()
    if store.ready:
        return store

    warmup_finetuned_chat_async()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.ready:
            return store
        if store.error:
            raise RuntimeError(store.error)
        time.sleep(0.5)

    raise RuntimeError(
        "Fine-tuned model is still loading (~1–2 min on first start). Please wait and retry."
    )


def finetuned_retrieve(
    question: str,
    top_k: int = 8,
    history: Optional[list] = None,
) -> dict:
    """Fast retrieval-only compare (no LLM). Returns in ~1–2 seconds."""
    state = load_library_state()
    if not state or state.get("status") != "ready":
        raise RuntimeError("Library not ready. Run library_pipeline or POST /process first.")

    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    store = _ensure_store_ready()
    history = _normalize_finetuned_history(history)
    search_query = _retrieval_query(question, history)

    base_label = f"Base BGE ({store.base_model_name})"
    tuned_label = f"Fine-tuned BGE (1 epoch, 40-Q)"

    base_chunks = store.search_base(search_query, top_k=top_k)
    tuned_chunks = store.search_finetuned(search_query, top_k=top_k)

    return {
        "question": question,
        "base_bge": {
            "model": base_label,
            "answer": "",
            "sources": base_chunks[:5],
        },
        "finetuned_bge": {
            "model": tuned_label,
            "answer": "",
            "sources": tuned_chunks[:5],
        },
        "documents_in_library": state.get("documents", []),
        "history_turns_used": len(history),
        "finetuned_model_path": store.finetuned_path,
    }


def finetuned_compare_answer(
    question: str,
    top_k: int = 8,
    history: Optional[list] = None,
    with_answers: bool = True,
    llm_top_k: int = 3,
) -> dict:
    state = load_library_state()
    if not state or state.get("status") != "ready":
        raise RuntimeError("Library not ready. Run library_pipeline or POST /process first.")

    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    store = _ensure_store_ready()
    history = _normalize_finetuned_history(history)
    search_query = _retrieval_query(question, history)

    base_label = f"Base BGE ({store.base_model_name})"
    tuned_label = f"Fine-tuned BGE (1 epoch, 40-Q)"

    base_chunks = store.search_base(search_query, top_k=top_k)
    tuned_chunks = store.search_finetuned(search_query, top_k=top_k)

    base_answer = ""
    tuned_answer = ""
    if with_answers:
        llm_base = base_chunks[:llm_top_k]
        llm_tuned = tuned_chunks[:llm_top_k]
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_base = pool.submit(
                _llm_answer, question, llm_base, base_label, history, "base_answer"
            )
            f_tuned = pool.submit(
                _llm_answer, question, llm_tuned, tuned_label, history, "finetuned_answer"
            )
            base_answer = f_base.result()
            tuned_answer = f_tuned.result()

    return {
        "question": question,
        "base_bge": {
            "model": base_label,
            "answer": base_answer,
            "sources": base_chunks[:5],
        },
        "finetuned_bge": {
            "model": tuned_label,
            "answer": tuned_answer,
            "sources": tuned_chunks[:5],
        },
        "documents_in_library": state.get("documents", []),
        "history_turns_used": len(history),
        "finetuned_model_path": store.finetuned_path,
    }


def finetuned_single_answer(
    question: str,
    top_k: int = 8,
    history: Optional[list] = None,
    llm_top_k: int = 3,
) -> dict:
    """Single fine-tuned retriever + LLM answer (no base BGE compare)."""
    state = load_library_state()
    if not state or state.get("status") != "ready":
        raise RuntimeError("Library not ready. Run library_pipeline or POST /process first.")

    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    store = _ensure_store_ready()
    history = _normalize_finetuned_history(history)
    search_query = _retrieval_query(question, history)

    tuned_label = "Fine-tuned BGE"
    tuned_chunks = store.search_finetuned(search_query, top_k=top_k)
    answer = _llm_answer(
        question,
        tuned_chunks[:llm_top_k],
        tuned_label,
        history,
        "finetuned_answer",
    )

    return {
        "question": question,
        "model": tuned_label,
        "answer": answer,
        "sources": tuned_chunks[:5],
        "documents_in_library": state.get("documents", []),
        "finetuned_model_path": store.finetuned_path,
    }
