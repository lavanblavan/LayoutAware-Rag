import os,sys
import numpy as np
import faiss
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Retriever:
    """
    Loads FAISS index and metadata to perform retrieval (flat or hierarchical).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", shared_model=None):
        self.model_name = model_name
        if shared_model is not None:
            self.model = shared_model
        else:
            from services.retrieval_cache import get_embedding_model
            self.model = get_embedding_model(model_name)
        self.index = None
        self.fine_chunks = None
        self.all_chunk_groups = None
        self.group_embeddings = None

    def load_index(self, index_path, meta_path):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(index_path)
        data = np.load(meta_path, allow_pickle=True)
        self.fine_chunks = data['fine_chunks']
        self.group_embeddings = data['group_embeddings']
        self.all_chunk_groups = data['all_chunk_groups']
        print(f"Loaded {len(self.fine_chunks)} fine chunks and {len(self.all_chunk_groups)} groups.")

    def flat_search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("⚠️ Index not loaded. Use load_index() first.")

        if hasattr(self, "embedder") and self.embedder is not None:
            query_vec = self.embedder.embed_query(query)
        elif "bge" in getattr(self, "model_name", "").lower():
            q = f"Represent this sentence for searching relevant passages: {query}"
            query_vec = self.model.encode([q], normalize_embeddings=True).astype('float32')
        else:
            query_vec = self.model.encode([query], normalize_embeddings=True).astype('float32')
        D, I = self.index.search(query_vec, top_k)
        results = [(self.fine_chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results

    def hierarchical_search(self, query, top_k_groups=5, top_k_fine=5):
        if self.group_embeddings is None or len(self.group_embeddings) == 0:
            raise ValueError("⚠️ No group embeddings found.")

        query_vec = self.model.encode([query], normalize_embeddings=True).astype('float32')

        # 1️⃣ Group search
        sim_scores = np.dot(self.group_embeddings, query_vec.T).squeeze()
        top_group_indices = sim_scores.argsort()[::-1][:top_k_groups]

        # 2️⃣ Collect candidate fine chunks
        candidate_chunks = []
        for idx in top_group_indices:
            candidate_chunks.extend(self.all_chunk_groups[idx])

        # 3️⃣ Fine-grained reranking within candidate chunks
        candidate_embeddings = self.model.encode(candidate_chunks, normalize_embeddings=True)
        fine_scores = np.dot(candidate_embeddings, query_vec.T).squeeze()
        top_indices = fine_scores.argsort()[::-1][:top_k_fine]

        results = [(candidate_chunks[i], float(fine_scores[i])) for i in top_indices]
        return results


