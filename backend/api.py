"""

Chatbot backend API — library mode with MiniLM vs BGE comparison.

"""

import os

import sys

import traceback

from typing import List, Optional



from fastapi import FastAPI, HTTPException, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import uvicorn



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from services.library_pipeline import (

    load_library_state,

    process_library,

    list_library_pdfs,

)

from services.compare_chat import compare_answer
from services.finetuned_chat import (
    finetuned_compare_answer,
    finetuned_retrieve,
    finetuned_single_answer,
    get_finetuned_store,
    warmup_finetuned_chat_async,
)

from services.retrieval_cache import warmup_retrieval, get_index_store, invalidate_retrieval_cache

from services.finetune_store import (

    add_question,

    export_training_pairs,

    load_corpus,

    stats as finetune_stats,

    update_question_labels,

    write_questions_template,

)





app = FastAPI(title="DocChat Library Backend", version="2.0")


@app.on_event("startup")
def _startup_warmup():
    state = load_library_state()
    if state and state.get("status") == "ready":
        try:
            warmup_retrieval()
        except Exception as e:
            print(f"Retrieval warmup skipped: {e}")
        warmup_finetuned_chat_async()


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)





class HistoryTurn(BaseModel):

    question: str

    minilm_answer: str = ""

    bge_answer: str = ""


class FinetunedHistoryTurn(BaseModel):
    question: str
    base_answer: str = ""
    finetuned_answer: str = ""


class ChatRequest(BaseModel):

    question: str

    top_k: int = 8

    history: List[HistoryTurn] = []





class SourceChunk(BaseModel):

    document: str

    chunk: str

    score: float

    chunk_id: Optional[str] = None





class ModelAnswer(BaseModel):

    model: str

    answer: str

    sources: List[SourceChunk] = []





class CompareResponse(BaseModel):

    question: str

    minilm: ModelAnswer

    bge: ModelAnswer

    documents_in_library: List[str] = []


class FinetunedChatRequest(BaseModel):
    question: str
    top_k: int = 8
    history: List[FinetunedHistoryTurn] = []
    with_answers: bool = True


class FinetunedCompareResponse(BaseModel):
    question: str
    base_bge: ModelAnswer
    finetuned_bge: ModelAnswer
    documents_in_library: List[str] = []
    finetuned_model_path: str = ""


class SingleChatResponse(BaseModel):
    question: str
    model: str
    answer: str
    documents_in_library: List[str] = []





class StatusResponse(BaseModel):

    status: str

    message: str

    documents: List[str] = []


def _to_model_answer(data: dict) -> ModelAnswer:
    return ModelAnswer(
        model=data["model"],
        answer=data["answer"],
        sources=[SourceChunk(**s) for s in data.get("sources", [])],
    )





class FinetuneQuestionRequest(BaseModel):

    question: str

    gold_chunk_ids: List[str] = []

    split: str = "train"





class FinetuneLabelRequest(BaseModel):

    gold_chunk_ids: List[str]

    split: Optional[str] = None





def _run_library_build(skip_summary: bool = False):

    try:

        process_library(skip_summary=skip_summary)

        warmup_retrieval(force=True)
        warmup_finetuned_chat_async(force=True)

    except Exception as e:

        invalidate_retrieval_cache()

        from services.library_pipeline import save_library_state



        save_library_state({

            "status": "error",

            "message": str(e),

            "documents": list_library_pdfs(),

        })

        traceback.print_exc()





@app.get("/")

def root():

    return {

        "service": "docchat-library",

        "port": 9091,

        "message": "Library compare chat — MiniLM vs BGE",

        "endpoints": [
            "/health",
            "/status",
            "/process",
            "/chat/compare",
            "/chat/finetuned",
            "/chat/finetuned/ask",
            "/chat/finetuned/retrieve",
            "/chat/finetuned/status",
            "/finetune/stats",
            "/finetune/export",
        ],

    }





@app.get("/health")

def health():

    return {"ok": True, "service": "docchat-library", "port": 9091}





@app.get("/status", response_model=StatusResponse)

def status():

    state = load_library_state()

    pdfs = list_library_pdfs()

    if not state:

        return StatusResponse(

            status="idle",

            message=f"{len(pdfs)} PDF(s) in library folder. Run POST /process to index.",

            documents=pdfs,

        )

    return StatusResponse(

        status=state.get("status", "idle"),

        message=state.get("message", ""),

        documents=state.get("documents", pdfs),

    )





@app.post("/process", response_model=StatusResponse)

def process(background_tasks: BackgroundTasks, skip_summary: bool = False):

    pdfs = list_library_pdfs()

    if not pdfs:

        raise HTTPException(

            status_code=400,

            detail=f"No PDFs in {os.path.join('Documents', 'documents')}. Add files first.",

        )

    background_tasks.add_task(_run_library_build, skip_summary)

    return StatusResponse(

        status="processing",

        message=f"Indexing {len(pdfs)} document(s) with layout chunks + MiniLM + BGE…",

        documents=pdfs,

    )





@app.post("/chat/compare", response_model=CompareResponse)

def chat_compare(req: ChatRequest):

    state = load_library_state()

    if not state:

        raise HTTPException(

            status_code=400,

            detail="Library not indexed. POST /process first.",

        )

    if state.get("status") == "processing":

        raise HTTPException(status_code=409, detail="Library is still processing.")

    if state.get("status") != "ready":

        raise HTTPException(status_code=400, detail=state.get("message", "Library error."))



    question = (req.question or "").strip()

    if not question:

        raise HTTPException(status_code=400, detail="Question cannot be empty.")



    try:

        result = compare_answer(
            question,
            top_k=req.top_k,
            history=[h.model_dump() for h in req.history],
        )

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))



    return CompareResponse(

        question=result["question"],

        minilm=_to_model_answer(result["minilm"]),

        bge=_to_model_answer(result["bge"]),

        documents_in_library=result.get("documents_in_library", []),

    )





@app.get("/chat/finetuned/status")
def finetuned_chat_status():
    store = get_finetuned_store()
    return store.stats()


@app.post("/chat/finetuned/retrieve", response_model=FinetunedCompareResponse)
def chat_finetuned_retrieve(req: FinetunedChatRequest):
    """Fast retrieval-only — sources appear in ~1–2 seconds (no LLM)."""
    state = load_library_state()
    if not state or state.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Library not indexed or not ready.")

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = finetuned_retrieve(
            question,
            top_k=req.top_k,
            history=[h.model_dump() for h in req.history],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return FinetunedCompareResponse(
        question=result["question"],
        base_bge=_to_model_answer(result["base_bge"]),
        finetuned_bge=_to_model_answer(result["finetuned_bge"]),
        documents_in_library=result.get("documents_in_library", []),
        finetuned_model_path=result.get("finetuned_model_path", ""),
    )


@app.post("/chat/finetuned", response_model=FinetunedCompareResponse)
def chat_finetuned(req: FinetunedChatRequest):
    state = load_library_state()
    if not state:
        raise HTTPException(status_code=400, detail="Library not indexed. POST /process first.")
    if state.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Library is still processing.")
    if state.get("status") != "ready":
        raise HTTPException(status_code=400, detail=state.get("message", "Library error."))

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        history_payload = [h.model_dump() for h in req.history]
        result = finetuned_compare_answer(
            question,
            top_k=req.top_k,
            history=history_payload,
            with_answers=req.with_answers,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return FinetunedCompareResponse(
        question=result["question"],
        base_bge=_to_model_answer(result["base_bge"]),
        finetuned_bge=_to_model_answer(result["finetuned_bge"]),
        documents_in_library=result.get("documents_in_library", []),
        finetuned_model_path=result.get("finetuned_model_path", ""),
    )


@app.post("/chat/finetuned/ask", response_model=SingleChatResponse)
def chat_finetuned_single(req: FinetunedChatRequest):
    """Fine-tuned retriever + LLM only — for the single-model chat page."""
    state = load_library_state()
    if not state or state.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Library not indexed or not ready.")

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = finetuned_single_answer(
            question,
            top_k=req.top_k,
            history=[h.model_dump() for h in req.history],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return SingleChatResponse(
        question=result["question"],
        model=result["model"],
        answer=result.get("answer") or "",
        documents_in_library=result.get("documents_in_library", []),
    )

@app.get("/finetune/stats")

def finetune_status():

    return finetune_stats()





@app.get("/finetune/corpus")

def finetune_corpus(limit: int = 50, offset: int = 0):

    rows = load_corpus()

    return {"total": len(rows), "chunks": rows[offset : offset + limit]}





@app.post("/finetune/questions")

def finetune_add_question(req: FinetuneQuestionRequest):

    rec = add_question(req.question, req.gold_chunk_ids, split=req.split)

    return rec





@app.patch("/finetune/questions/{qid}")

def finetune_label_question(qid: str, req: FinetuneLabelRequest):

    try:

        return update_question_labels(qid, req.gold_chunk_ids, split=req.split)

    except KeyError as e:

        raise HTTPException(status_code=404, detail=str(e))





@app.post("/finetune/export")

def finetune_export():

    result = export_training_pairs()

    if result.get("status") == "no_labels":

        write_questions_template()

    return result





if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=9091)


