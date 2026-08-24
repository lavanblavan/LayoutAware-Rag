# app.py
import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Retrieve_Documents.main_retrieve import MainRetriever  # adjust path if needed

# --- FastAPI app ---
app = FastAPI(title="Document QA Retriever")

# Initialize the retriever once
retriever = MainRetriever()


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str
    top_summary_docs: int = 1
    top_k_per_doc: int = 5
    # final_top_k: int = 3


class ChunkResult(BaseModel):
    document: str
    chunk: str
    score: float


class QueryResponse(BaseModel):
    question: str
    results: List[ChunkResult]


# --- API endpoint ---
@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """
    Ask a question and retrieve top detailed chunks from selected documents.
    """
    results = retriever.search(
        query=req.question,
        top_summary_docs=req.top_summary_docs,
        top_k_per_doc=req.top_k_per_doc,
        
    )

    # Convert to Pydantic model format
    results_out = [ChunkResult(
        document=r["document"],
        chunk=r["chunk"],
        score=r["score"]
    ) for r in results]

    return QueryResponse(
        question=req.question,
        results=results_out
    )


# --- Run the API ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
