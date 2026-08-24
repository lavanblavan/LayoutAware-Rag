"""
Run a batch of retrieval queries and save finetune data (no Groq — fast, no rate limits).

Each question:
  - logs to Documents/finetune/retrieval_runs.jsonl
  - adds to Documents/finetune/questions.jsonl with BGE top-1 chunk as provisional gold label

Usage:
  python services/batch_finetune_questions.py
  python services/batch_finetune_questions.py --with-answers   # also call Groq (slow, needs quota)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger
from services.compare_chat import collect_retrieval
from services.finetune_store import add_question, load_questions, stats

log = get_logger(__name__)

API = os.getenv("RAG_API", "http://localhost:9091")

QUESTIONS = [
    ("What is Corrective RAG (CRAG)?", "train"),
    ("What failure modes does CRAG address when retrieval returns bad documents?", "train"),
    ("How does the CRAG retrieval evaluator assign Correct, Incorrect, or Ambiguous?", "train"),
    ("Which datasets were used to evaluate CRAG?", "train"),
    ("What is GraphRAG and how does it differ from standard vector RAG?", "train"),
    ("How does GraphRAG build a knowledge graph from source documents?", "train"),
    ("What are the main pipeline stages in GraphRAG?", "train"),
    ("When does GraphRAG outperform baseline RAG on global sensemaking queries?", "train"),
    ("What is Self-RAG and what problem does it solve?", "train"),
    ("What are reflection tokens in Self-RAG?", "train"),
    ("How does Self-RAG decide whether to retrieve additional passages?", "train"),
    ("Which tasks and benchmarks does Self-RAG improve over standard RAG?", "train"),
    ("What are naive RAG, advanced RAG, and modular RAG?", "train"),
    ("What quality scores are used to evaluate RAG systems?", "train"),
    ("Why is RAG robustness important when retrieved documents are noisy?", "train"),
    ("Which RAG tools and frameworks are discussed in the ecosystem survey?", "train"),
    ("How does Lewis et al. combine a neural retriever with a seq2seq generator?", "test"),
    ("What is the difference between RAG-Token and RAG-Sequence?", "test"),
    ("On which generation and QA tasks was the original RAG model evaluated?", "test"),
    ("How does RAG perform on open-domain question answering vs extractive models?", "test"),
]


def _gold_from_bge(sources: list[dict]) -> list[str]:
    if not sources:
        return []
    cid = sources[0].get("chunk_id")
    return [cid] if cid else []


def _compare_via_api(question: str) -> dict:
    body = json.dumps({"question": question, "top_k": 8, "history": []}).encode()
    req = urllib.request.Request(
        f"{API}/chat/compare",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def run_batch(dry_run: bool = False, with_answers: bool = False) -> dict:
    existing = {q["question"].strip().lower() for q in load_questions()}
    results = []

    for i, (question, split) in enumerate(QUESTIONS, start=1):
        key = question.strip().lower()
        if key in existing:
            log.info("[%02d/20] skip (already stored): %s…", i, question[:60])
            results.append({"question": question, "status": "skipped"})
            continue

        log.info("[%02d/20] %s", i, question)
        if dry_run:
            results.append({"question": question, "status": "dry_run"})
            continue

        t0 = time.time()
        try:
            if with_answers:
                out = _compare_via_api(question)
            else:
                out = collect_retrieval(question, top_k=8, history=[])

            bge_sources = out["bge"]["sources"]
            gold = _gold_from_bge(bge_sources)
            rec = add_question(
                question,
                gold_chunk_ids=gold,
                split=split,
                source="auto_bge_top1",
            )
            elapsed = round(time.time() - t0, 1)
            log.info("         ok (%ss) gold=%s", elapsed, gold[0] if gold else "none")
            results.append({
                "question": question,
                "status": "ok",
                "id": rec["id"],
                "gold_chunk_ids": gold,
                "elapsed_s": elapsed,
            })
        except Exception as e:
            log.error("         ERROR: %s", e)
            results.append({"question": question, "status": "error", "error": str(e)})

    summary = stats()
    summary["batch"] = {
        "total_planned": len(QUESTIONS),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "mode": "full" if with_answers else "retrieval_only",
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch finetune question collection")
    parser.add_argument("--dry-run", action="store_true", help="List questions only")
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Call Groq for answers (default: retrieval only)",
    )
    args = parser.parse_args()

    summary = run_batch(dry_run=args.dry_run, with_answers=args.with_answers)
    log.info("--- Finetune store ---")
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    configure_logging()
    main()
