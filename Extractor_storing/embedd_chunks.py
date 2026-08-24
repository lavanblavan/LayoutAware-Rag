import os,sys
import numpy as np
import faiss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.torch_win import bootstrap_torch
bootstrap_torch()
from sentence_transformers import SentenceTransformer
from Extractor_storing.Tokenization import SentenceTokenizer
from Extractor_storing.create_chunks import SemanticChunker  # assumes you have this file separately
from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

class EmbedChunks:
    """
    Handles:
    - Chunking the document (via SemanticChunker)
    - Embedding chunks
    - Building and saving FAISS index
    """
    BGE_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.tokenizer = SentenceTokenizer(model_name)
        self.chunker = SemanticChunker(model_name)
        self.index = None
        self.fine_chunks = []
        self.all_chunk_groups = []
        self.group_embeddings = None

    def _is_bge(self):
        return "bge" in self.model_name.lower()

    def embed_query(self, query: str):
        if self._is_bge():
            query = f"Represent this sentence for searching relevant passages: {query}"
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        return vec

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")

    def compute_group_embeddings(self, chunk_groups):
        group_embeddings = []
        for group in chunk_groups:
            if not group:
                continue
            group_vecs = self.embed_texts(group)
            group_mean = np.mean(group_vecs, axis=0)
            group_embeddings.append(group_mean)
        if not group_embeddings:
            return np.zeros((0, self.dimension), dtype="float32")
        return np.array(group_embeddings, dtype='float32')

    def build_index_from_chunks(self, fine_chunks, all_chunk_groups=None):
        """Embed already-created chunks and build a FAISS index."""
        if all_chunk_groups is None:
            all_chunk_groups = [[chunk] for chunk in fine_chunks]

        log.info("%s fine-grained chunks created.", len(fine_chunks))
        if fine_chunks:
            fine_embeddings = self.embed_texts(fine_chunks)
        else:
            fine_embeddings = np.zeros((0, self.dimension), dtype="float32")

        index = faiss.IndexFlatIP(self.dimension)
        if fine_embeddings.shape[0] > 0:
            index.add(fine_embeddings)

        self.index = index
        self.fine_chunks = fine_chunks
        self.all_chunk_groups = all_chunk_groups
        self.group_embeddings = self.compute_group_embeddings(all_chunk_groups)

        log.info("FAISS index built with %s vectors.", index.ntotal)
        return index, fine_chunks, all_chunk_groups

    def build_index(self, text):
        """
        Full pipeline: chunk → embed → build FAISS index.
        """
        log.info("Chunking document...")
        sentences, coarse_chunks, fine_chunks, all_chunk_groups = self.chunker.run(text, strict_layout=True)
        return self.build_index_from_chunks(fine_chunks, all_chunk_groups)

    def save_index(self, index_path="faiss_index.index", meta_path="meta.npz"):
        """
        Save FAISS index and metadata (chunks, groups, group embeddings).
        """
        if self.index is None:
            raise ValueError("⚠️ No index found. Build it first using build_index().")

        faiss.write_index(self.index, index_path)
        np.savez_compressed(
            meta_path,
            fine_chunks=self.fine_chunks,
            group_embeddings=self.group_embeddings,
            all_chunk_groups=np.array(self.all_chunk_groups, dtype=object)
        )
        log.info("Saved index to %s and metadata to %s", index_path, meta_path)

if __name__ == "__main__":
    configure_logging()
    file_path = r"C:\Users\Lavan\Desktop\Chatbot\Document_Summarizer\Document_Summarizer\Documents\Police\LK_Police_Ordinance_summary.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    embedder = EmbedChunks()
    index, fine_chunks, groups = embedder.build_index(content)

    embedder.save_index(
        index_path="C:\\Users\\Lavan\\Desktop\\Chatbot\\Document_Summarizer\\faiss_index.index",
        meta_path="C:\\Users\\Lavan\\Desktop\\Chatbot\\Document_Summarizer\\meta.npz"
    )
    log.info("Index and metadata saved.")