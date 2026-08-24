# import os
# import sys
# from pathlib import Path
# import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from Retrieve_Documents.Select_documents import SelectDocuments
# from Retrieve_Documents.Retriever import Retriever
# from Retrieve_Documents.Reranker import ReRanker
# from settings.Settings import Config as settings_module


# class MainRetriever:
#     """
#     Two-stage retrieval pipeline:
#     1) Select top summary documents for the query.
#     2) Retrieve detailed chunks from extracted FAISS for selected documents.
#     3) Rerank all retrieved chunks together.
#     4) Return top 5 chunks overall with document names and scores.
#     """

#     def __init__(self):
#         self.summary_selector = SelectDocuments()
#         self.extract_faiss_folder = settings_module.Extracted_Faiss

#         if not os.path.exists(self.extract_faiss_folder):
#             raise FileNotFoundError(f"Extracted FAISS folder missing: {self.extract_faiss_folder}")

#         self.retriever = Retriever()
#         self.reranker = ReRanker()

#     def find_extracted_files(self, summary_base_name):
#         """
#         Map a summary document (e.g., "01-1990_E_summary") to its extracted FAISS files:
#         "01-1990_E_faiss.index" + "01-1990_E_meta.npz"
#         """
#         base_name = summary_base_name.replace("_summary", "")
#         index_file = f"{base_name}_faiss.index"
#         meta_file  = f"{base_name}_meta.npz"

#         index_path = os.path.join(self.extract_faiss_folder, index_file)
#         meta_path  = os.path.join(self.extract_faiss_folder, meta_file)

#         if not os.path.exists(index_path) or not os.path.exists(meta_path):
#             print(f"⚠️ Extracted FAISS missing for {summary_base_name}")
#             return None, None

#         return index_path, meta_path

#     def retrieve_detailed_chunks(self, query, selected_summary_docs, top_k_per_doc=10):
#         """
#         Retrieve detailed chunks from extracted FAISS for selected summary documents.
#         Returns a list of (document_name, chunk_text, score) tuples.
#         """
#         all_chunks = []

#         for doc in selected_summary_docs:
#             doc_name = doc["document"]
#             index_path, meta_path = self.find_extracted_files(doc_name)
#             if not index_path or not meta_path:
#                 continue

#             # Load extracted FAISS
#             try:
#                 self.retriever.load_index(index_path, meta_path)
#             except Exception as e:
#                 print(f"⚠️ Could not load extracted index for {doc_name}: {e}")
#                 continue

#             # Retrieve top_k_per_doc chunks
#             retrieved_chunks = self.retriever.flat_search(query, top_k=top_k_per_doc)
#             for chunk, score in retrieved_chunks:
#                 all_chunks.append({
#                     "document": doc_name,
#                     "chunk": chunk,
#                     "score": score
#                 })

#         return all_chunks

#     def search(self, query, top_summary_docs=3, top_k_per_doc=10, final_top_k=5):
#         """
#         Full two-stage pipeline:
#         1) Summary retrieval → select documents
#         2) Extracted retrieval → collect all chunks from selected documents
#         3) Rerank all chunks together
#         4) Return top final_top_k chunks overall
#         """
#         print(f"\n🔎 Running search for query: {query}")

#         # Stage 1: select top summary documents
#         selected_docs = self.summary_selector.select_documents(query, top_k_docs=top_summary_docs, top_k_chunks=5)

#         print("\n=== 📝 Selected Summary Documents ===")
#         for doc in selected_docs:
#             print(f"{doc['document']} → Score: {doc['score']:.4f}")

#         # Stage 2: retrieve detailed chunks from extracted FAISS
#         detailed_chunks = self.retrieve_detailed_chunks(query, selected_docs, top_k_per_doc=top_k_per_doc)
#         if not detailed_chunks:
#             print("⚠️ No detailed chunks retrieved.")
#             return []

#         # Stage 3: rerank all retrieved chunks together
#         chunk_texts = [c["chunk"] for c in detailed_chunks]
#         reranked = self.reranker.rerank(query, chunk_texts, top_k=min(final_top_k, len(chunk_texts)))

#         # Combine reranked chunks with document names
#         top_chunks = []
#         for i, (chunk_text, score) in enumerate(reranked):
#             # Find the original document
#             doc_name = next((c["document"] for c in detailed_chunks if c["chunk"] == chunk_text), "Unknown")
#             top_chunks.append({
#                 "document": doc_name,
#                 "chunk": chunk_text,
#                 "score": score
#             })

#         return top_chunks


# if __name__ == "__main__":
#     retriever = MainRetriever()
#     query = "What is the police arrest procedure?"

#     results = retriever.search(query, top_summary_docs=2, top_k_per_doc=10, final_top_k=5)

#     print("\n\n=== FINAL TOP 5 DETAILED CHUNKS ===")
#     for res in results:
#         print(f"Document: {res['document']}, Score: {res['score']:.4f}")
#         print(f"Chunk: {res['chunk']}")
#         print("-" * 50)
import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Retrieve_Documents.Select_documents import SelectDocuments
from Retrieve_Documents.Retriever import Retriever
from settings.Settings import Config as settings_module


class MainRetriever:
    """
    Two-stage retrieval pipeline:
    1) Select top summary documents for the query.
    2) Retrieve detailed chunks from extracted FAISS for selected documents.
    3) Return top chunks from all retrieved detailed chunks without reranking.
    """

    def __init__(self):
        self.summary_selector = SelectDocuments()
        self.extract_faiss_folder = settings_module.Extracted_Faiss

        if not os.path.exists(self.extract_faiss_folder):
            raise FileNotFoundError(f"Extracted FAISS folder missing: {self.extract_faiss_folder}")

        self.retriever = Retriever()

    def find_extracted_files(self, summary_base_name):
        """
        Map a summary document (e.g., "01-1990_E_summary") to its extracted FAISS files:
        "01-1990_E_faiss.index" + "01-1990_E_meta.npz"
        """
        base_name = summary_base_name.replace("_summary", "")
        index_file = f"{base_name}_faiss.index"
        meta_file  = f"{base_name}_meta.npz"

        index_path = os.path.join(self.extract_faiss_folder, index_file)
        meta_path  = os.path.join(self.extract_faiss_folder, meta_file)

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            print(f"⚠️ Extracted FAISS missing for {summary_base_name}")
            return None, None

        return index_path, meta_path

    def retrieve_detailed_chunks(self, query, selected_summary_docs, top_k_per_doc=10):
        """
        Retrieve detailed chunks from extracted FAISS for selected summary documents.
        Returns a list of (document_name, chunk_text, score) tuples.
        """
        all_chunks = []

        for doc in selected_summary_docs:
            doc_name = doc["document"]
            index_path, meta_path = self.find_extracted_files(doc_name)
            if not index_path or not meta_path:
                continue

            # Load extracted FAISS
            try:
                self.retriever.load_index(index_path, meta_path)
            except Exception as e:
                print(f"⚠️ Could not load extracted index for {doc_name}: {e}")
                continue

            # Retrieve top_k_per_doc chunks
            retrieved_chunks = self.retriever.flat_search(query, top_k=top_k_per_doc)
            for chunk, score in retrieved_chunks:
                all_chunks.append({
                    "document": doc_name,
                    "chunk": chunk,
                    "score": score
                })

        return all_chunks

    def search(self, query, top_summary_docs=3, top_k_per_doc=10):
        """
        Full two-stage pipeline:
        1) Summary retrieval → select documents
        2) Extracted retrieval → collect all chunks from selected documents
        3) Return top final chunks based on FAISS scores (no reranking)
        """
        print(f"\n🔎 Running search for query: {query}")

        # Stage 1: select top summary documents
        selected_docs = self.summary_selector.select_documents(query, top_k_docs=top_summary_docs, top_k_chunks=5)

        print("\n=== 📝 Selected Summary Documents ===")
        for doc in selected_docs:
            print(f"{doc['document']} → Score: {doc['score']:.4f}")

        # Stage 2: retrieve detailed chunks from extracted FAISS
        detailed_chunks = self.retrieve_detailed_chunks(query, selected_docs, top_k_per_doc=top_k_per_doc)
        if not detailed_chunks:
            print("⚠️ No detailed chunks retrieved.")
            return []

        # Stage 3: sort all retrieved chunks by FAISS similarity score
        detailed_chunks.sort(key=lambda x: x["score"], reverse=True)

        return detailed_chunks


if __name__ == "__main__":
    retriever = MainRetriever()
    query = "What is the police arrest procedure?"

    results = retriever.search(query, top_summary_docs=2, top_k_per_doc=10)

    print("\n\n=== TOP DETAILED CHUNKS ===")
    for res in results[:5]:  # Show only top 5 by score
        print(f"Document: {res['document']}, Score: {res['score']:.4f}")
        print(f"Chunk: {res['chunk']}")
        print("-" * 50)
