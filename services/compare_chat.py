"""

Library chat: compare MiniLM vs BGE retrieval + Ollama answers.

Uses cached embedding models and FAISS indexes (loaded once at startup).

Supports up to 5 prior turns for follow-up continuity.

"""

from __future__ import annotations



import os

import re

import sys

from typing import Optional



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from settings.Settings import Config

from services.library_pipeline import load_library_state

from services.finetune_store import enrich_sources, log_retrieval_run

from services.retrieval_cache import get_llm_client, get_index_store, warmup_retrieval
from utils.session_log import get_logger

log = get_logger(__name__)



MAX_HISTORY_TURNS = 5

_FOLLOWUP_PRONOUNS = re.compile(

    r"\b(it|its|they|them|their|this|that|those|these|which|what about)\b",

    re.I,

)





def _normalize_history(history: Optional[list]) -> list[dict]:

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

            "minilm_answer": (item.get("minilm_answer") or item.get("answer") or "").strip(),

            "bge_answer": (item.get("bge_answer") or item.get("answer") or "").strip(),

        })

    return turns





def _retrieval_query(question: str, history: list[dict]) -> str:

    """Include recent user questions and topic hints so follow-ups retrieve relevant chunks."""

    if not history:

        return question



    parts = [h["question"] for h in history[-3:]] + [question]



    if _FOLLOWUP_PRONOUNS.search(question):

        last = history[-1]

        hint = last["question"]

        for key in ("bge_answer", "minilm_answer"):

            answer = (last.get(key) or "").strip()

            if answer:

                hint += " " + answer[:160]

                break

        parts.insert(0, hint)



    return " ".join(parts).strip()





def _history_block(history: list[dict], answer_key: str) -> str:

    if not history:

        return ""

    lines = []

    for i, turn in enumerate(history, start=1):

        answer = turn.get(answer_key) or turn.get("minilm_answer") or turn.get("bge_answer") or ""

        lines.append(f"Turn {i}\nUser: {turn['question']}\nAssistant: {answer}")

    return (

        "Previous conversation (for context — resolve pronouns like 'it', 'that', 'they'):\n\n"

        + "\n\n".join(lines)

        + "\n\n"

    )





def _is_missing_model(error: Exception) -> bool:
    msg = str(error).lower()
    return "model_not_found" in msg or "does not exist" in msg or "not found" in msg


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    resp = getattr(error, "response", None)
    return resp is not None and getattr(resp, "status_code", None) == 429


def _clean_model_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


def _extract_message_text(message) -> str:
    text = (getattr(message, "content", None) or "").strip()
    if not text:
        text = (getattr(message, "reasoning", None) or "").strip()
    return _clean_model_output(text)


def _completion_tokens(model: str) -> int:
    return 2048


def _llm_answer(
    question: str,
    chunks: list[dict],
    label: str,
    history: list[dict],
    answer_key: str,
) -> str:

    if not chunks:

        return f"[{label}] No relevant passages found in the library."



    context = "\n\n---\n\n".join(

        f"[{c['document']} | score={c['score']:.3f}]\n{c['chunk']}"

        for c in chunks

    )

    client = get_llm_client()

    history_text = _history_block(history, answer_key)

    user_prompt = (

        f"Retriever: {label}\n\n"

        f"{history_text}"

        f"Excerpts:\n{context}\n\n"

        f"Current question: {question}\n\n"

        "Answer:"

    )

    system_prompt = (

        "You are a helpful assistant for reseach related question  from research paper.\n"

        "Answer ONLY from the excerpts below. If the answer is not supported, say so clearly.\n"

        "If the user asks a follow-up, use the conversation history to interpret the question.\n"

        "Return the answer text directly — no analysis, planning, or thinking tags."

    )



    last_error = None

    for model in dict.fromkeys(Config.LLM_MODEL_FALLBACKS):

        tokens = _completion_tokens(model)

        for attempt in range(3):

            try:

                response = client.chat.completions.create(

                    model=model,

                    messages=[

                        {"role": "system", "content": system_prompt},

                        {"role": "user", "content": user_prompt},

                    ],

                    temperature=0.2,

                    max_tokens=tokens,

                )

                text = _extract_message_text(response.choices[0].message)

                if text:

                    return text

                last_error = RuntimeError(f"{model} returned empty content")

                if attempt < 2:

                    tokens = min(8000, tokens + 2000)

                    continue

                break

            except Exception as e:
                last_error = e
                if _is_missing_model(e) or _is_rate_limit(e):
                    break
                raise



    fallback = (

        f"[{label}] Could not generate an answer right now. "

        "Please retry — retrieved passages are listed under Sources."

    )

    if last_error:

        log.warning("LLM answer failed (%s): %s", label, last_error)

    return fallback





def compare_answer(

    question: str,

    top_k: int = 8,

    history: Optional[list] = None,

) -> dict:

    state = load_library_state()

    if not state or state.get("status") != "ready":

        raise RuntimeError(

            "Library not ready. Run: python services/library_pipeline.py"

        )



    question = (question or "").strip()

    if not question:

        raise ValueError("Question cannot be empty.")



    history = _normalize_history(history)

    search_query = _retrieval_query(question, history)



    store = get_index_store()

    if not store.ready:

        warmup_retrieval()



    results = {}

    for model_key, model_label, answer_key in (

        ("minilm", "MiniLM (all-MiniLM-L6-v2)", "minilm_answer"),

        ("bge", "BGE (bge-small-en-v1.5)", "bge_answer"),

    ):

        chunks = store.search(

            model_key,

            search_query,

            top_k_per_doc=max(4, top_k // 2),

        )

        top_chunks = enrich_sources(chunks[:top_k])

        answer = _llm_answer(question, top_chunks, model_label, history, answer_key)

        results[model_key] = {

            "model": model_label,

            "answer": answer,

            "sources": top_chunks[:5],

        }



    log_retrieval_run(

        question,

        results["minilm"]["sources"],

        results["bge"]["sources"],

    )



    return {

        "question": question,

        "minilm": results["minilm"],

        "bge": results["bge"],

        "documents_in_library": state.get("documents", []),

        "history_turns_used": len(history),

    }


def collect_retrieval(
    question: str,
    top_k: int = 8,
    history: Optional[list] = None,
) -> dict:
    """Run MiniLM + BGE retrieval only (no LLM). For finetune data collection."""
    state = load_library_state()
    if not state or state.get("status") != "ready":
        raise RuntimeError("Library not ready. Run: python services/library_pipeline.py")

    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    history = _normalize_history(history)
    search_query = _retrieval_query(question, history)

    store = get_index_store()
    if not store.ready:
        warmup_retrieval()

    results = {}
    for model_key, model_label in (
        ("minilm", "MiniLM (all-MiniLM-L6-v2)"),
        ("bge", "BGE (bge-small-en-v1.5)"),
    ):
        chunks = store.search(
            model_key,
            search_query,
            top_k_per_doc=max(4, top_k // 2),
        )
        top_chunks = enrich_sources(chunks[:top_k])
        results[model_key] = {
            "model": model_label,
            "sources": top_chunks[:5],
        }

    log_retrieval_run(
        question,
        results["minilm"]["sources"],
        results["bge"]["sources"],
    )

    return {
        "question": question,
        "minilm": results["minilm"],
        "bge": results["bge"],
        "history_turns_used": len(history),
    }

