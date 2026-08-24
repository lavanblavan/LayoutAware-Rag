import sys,os
from sentence_transformers import CrossEncoder
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.session_log import get_logger

log = get_logger(__name__)
class ReRanker:
    """
    Reranks candidate chunks using a cross-encoder model for fine-grained semantic relevance.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        log.info("Loading ReRanker model: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates, top_k: int = 5):
        """
        Re-rank a list of retrieved text chunks based on query relevance.

        Args:
            query (str): The input search query.
            candidates (list): List of (chunk_text, score) tuples or just chunk_texts.
            top_k (int): Number of top reranked results to return.

        Returns:
            list: [(chunk_text, rerank_score), ...] sorted by score descending.
        """
        # Handle input type flexibility
        if not candidates:
            log.warning("No candidates provided for reranking.")
            return []

        # Extract text if provided as (text, score) tuples
        if isinstance(candidates[0], tuple):
            candidate_texts = [c[0] for c in candidates]
        else:
            candidate_texts = candidates

        # Build query-passage pairs
        pairs = [(query, text) for text in candidate_texts]

        # Predict similarity scores
        scores = self.model.predict(pairs)

        # Sort and pick top_k
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]

        reranked = [(candidate_texts[i], float(scores[i])) for i in top_indices]

        return reranked


