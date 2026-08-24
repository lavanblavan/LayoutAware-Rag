import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.session_log import configure_logging, get_logger
from Retrieve_Documents.Retriever import Retriever

log = get_logger(__name__)
from Retrieve_Documents.Reranker import ReRanker
from settings.Settings import Config as settings_module


class SelectDocuments:
    """
    Performs:
    - Retrieval from ALL summary FAISS indexes.
    - Reranking using CrossEncoder.
    - Selecting top matching documents.
    """

    def __init__(self):
        self.summary_faiss_folder = settings_module.Summarry_Faiss

        if not os.path.exists(self.summary_faiss_folder):
            raise FileNotFoundError(f"Summary FAISS folder not found: {self.summary_faiss_folder}")

        self.retriever = Retriever()
        self.reranker = ReRanker()

    def load_all_indexes(self):
        log.info("Scanning for FAISS indexes...")

        files = os.listdir(self.summary_faiss_folder)

        index_files = [f for f in files if f.endswith(".index")]
        meta_files  = [f for f in files if f.endswith(".npz")]

        index_pairs = []

        for idx_file in index_files:
            base = idx_file.replace(".index", "")

            # SMART MATCH: find any NPZ containing base prefix (e.g., without _faiss)
            possible_prefix = base.replace("_faiss", "")

            candidates = [
                f for f in meta_files
                if possible_prefix in f
            ]

            if not candidates:
                log.warning("No meta found for index: %s", idx_file)
                continue

            index_path = os.path.join(self.summary_faiss_folder, idx_file)
            meta_path = os.path.join(self.summary_faiss_folder, candidates[0])

            log.info("  Matched: %s  ↔  %s", idx_file, candidates[0])

            index_pairs.append((possible_prefix, index_path, meta_path))

        log.info("Found %s valid index/meta pairs.", len(index_pairs))
        return index_pairs


    def retrieve_for_file(self, query, index_path, meta_path, top_k=5):
        """
        Perform flat FAISS search on single document index.
        Returns a list of (chunk, score)
        """
        try:
            self.retriever.load_index(index_path, meta_path)
            results = self.retriever.flat_search(query, top_k)
            return results
        except Exception as e:
            log.warning("Error retrieving from %s: %s", index_path, e)
            return []

    def select_documents(self, query, top_k_docs=3, top_k_chunks=5):
        """
        Runs retrieval on each file → rerank → pick top documents.

        Returns:
            List of dictionaries:
            {
                "document": "filename",
                "best_chunk": "...",
                "score": float
            }
        """

        all_indexes = self.load_all_indexes()
        document_scores = []

        log.info("Running retrieval across %s documents...", len(all_indexes))

        for base, index_path, meta_path in all_indexes:
            log.info("Searching in ➜ %s", base)

            retrieved_chunks = self.retrieve_for_file(
                query, index_path, meta_path, top_k=top_k_chunks
            )

            if not retrieved_chunks:
                continue

            # Rerank results
            reranked = self.reranker.rerank(query, retrieved_chunks, top_k=1)

            if not reranked:
                continue

            best_chunk, score = reranked[0]

            document_scores.append({
                "document": base,
                "best_chunk": best_chunk,
                "score": score
            })

        # Sort documents by rerank score
        document_scores.sort(key=lambda x: x["score"], reverse=True)

        return document_scores[:top_k_docs]
if __name__ == "__main__":
    configure_logging()
    selector = SelectDocuments()
    query = "What is the police arrest procedure?"
    top_docs = selector.select_documents(query, top_k_docs=2, top_k_chunks=5)

    log.info("=== Top Retrieved Documents ===")
    for doc in top_docs:
        log.info("Document: %s, Score: %.4f", doc["document"], doc["score"])
        log.info("Best Chunk: %s", doc["best_chunk"])