"""
Collect 40 questions (8 per document) with MiniLM + BGE retrieval.

Saves:
  Documents/finetune/questions.jsonl       — question stubs (gold empty for now)
  Documents/finetune/retrieval_runs.jsonl  — full retrieval logs
  Documents/finetune/export/collect_40.json — rich bundle for labeling

Usage:
  python services/collect_40_questions.py
  python services/collect_40_questions.py --with-answers   # also call Groq (slow)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.compare_chat import collect_retrieval
from services.finetune_store import enrich_sources, log_retrieval_run, save_questions

API = os.getenv("RAG_API", "http://localhost:9091")
EXPORT_PATH = os.path.join(Config.FINETUNE_EXPORT_PATH, "collect_40.json")

# 40 questions — 8 per document
QUESTIONS_40 = [
    # 01 Foundational RAG (Lewis et al.) — 8
    ("01_Foundational_RAG_Lewis_2020", "How does Lewis et al. combine a neural retriever with a seq2seq generator?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "What is the difference between RAG-Token and RAG-Sequence?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "How does RAG perform on open-domain question answering compared to extractive models?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "What retriever does the original RAG paper use for Wikipedia passages?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "How does RAG perform on FEVER fact verification?", "test"),
    ("01_Foundational_RAG_Lewis_2020", "What generator model does the original RAG paper use?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "How does RAG marginalize over retrieved documents during generation?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "Can RAG update its knowledge by replacing the non-parametric memory?", "test"),
    # 02 RAG Survey — 8
    ("02_RAG_Survey_Gao_2023", "What are naive RAG, advanced RAG, and modular RAG?", "train"),
    ("02_RAG_Survey_Gao_2023", "What quality scores are used to evaluate RAG systems?", "train"),
    ("02_RAG_Survey_Gao_2023", "Why is RAG robustness important when retrieved documents are noisy?", "train"),
    ("02_RAG_Survey_Gao_2023", "Which RAG tools and frameworks are discussed in the ecosystem survey?", "train"),
    ("02_RAG_Survey_Gao_2023", "What are retrieval, generation, and augmentation in the RAG framework?", "test"),
    ("02_RAG_Survey_Gao_2023", "What are the main steps in naive RAG indexing and retrieval?", "train"),
    ("02_RAG_Survey_Gao_2023", "How does advanced RAG optimize pre-retrieval and post-retrieval?", "train"),
    ("02_RAG_Survey_Gao_2023", "What future research directions does the RAG survey propose?", "test"),
    # 03 Self-RAG — 8
    ("03_Self_RAG_Asai_2023", "What is Self-RAG and what problem does it solve?", "train"),
    ("03_Self_RAG_Asai_2023", "What are reflection tokens in Self-RAG?", "train"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG decide whether to retrieve additional passages?", "train"),
    ("03_Self_RAG_Asai_2023", "What is the retrieve-generate-critique loop in Self-RAG?", "train"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG improve factuality over standard RAG?", "test"),
    ("03_Self_RAG_Asai_2023", "What is adaptive retrieval in Self-RAG?", "train"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG critique the relevance of retrieved passages?", "train"),
    ("03_Self_RAG_Asai_2023", "On which benchmarks was Self-RAG evaluated?", "test"),
    # 04 Corrective RAG — 8
    ("04_Corrective_RAG_CRAG_2024", "What is Corrective RAG (CRAG)?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "What failure modes does CRAG address when retrieval returns bad documents?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "How does the CRAG retrieval evaluator assign Correct, Incorrect, or Ambiguous?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "Which datasets were used to evaluate CRAG?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "What is the decompose-then-recompose algorithm in CRAG?", "test"),
    ("04_Corrective_RAG_CRAG_2024", "When does CRAG trigger web search instead of internal retrieval?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "What is knowledge refinement in CRAG?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "How does Self-CRAG combine CRAG with Self-RAG?", "test"),
    # 05 GraphRAG — 8
    ("05_GraphRAG_Microsoft_2024", "What is GraphRAG and how does it differ from standard vector RAG?", "train"),
    ("05_GraphRAG_Microsoft_2024", "How does GraphRAG build a knowledge graph from source documents?", "train"),
    ("05_GraphRAG_Microsoft_2024", "What are the main pipeline stages in GraphRAG?", "train"),
    ("05_GraphRAG_Microsoft_2024", "When does GraphRAG outperform baseline RAG on global sensemaking queries?", "train"),
    ("05_GraphRAG_Microsoft_2024", "What evaluation criteria does GraphRAG use for global sensemaking answers?", "test"),
    ("05_GraphRAG_Microsoft_2024", "How does GraphRAG generate community summaries from graph communities?", "train"),
    ("05_GraphRAG_Microsoft_2024", "What is adaptive benchmarking for GraphRAG evaluation?", "train"),
    ("05_GraphRAG_Microsoft_2024", "How does GraphRAG extract entities and relationships from text chunks?", "test"),
]


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


def _trim_sources(sources: list[dict], n: int = 8) -> list[dict]:
    out = []
    for s in sources[:n]:
        out.append({
            "document": s.get("document"),
            "chunk_id": s.get("chunk_id"),
            "score": s.get("score"),
            "heading": (s.get("chunk") or "").split("\n", 1)[0][:120],
            "chunk_preview": (s.get("chunk") or "")[:400],
        })
    return out


def collect(with_answers: bool = False, top_k: int = 8) -> dict:
    os.makedirs(Config.FINETUNE_EXPORT_PATH, exist_ok=True)

    question_rows = []
    bundle = []
    ok = err = 0

    for i, (doc_stem, question, split) in enumerate(QUESTIONS_40, start=1):
        print(f"[{i:02d}/40] {question[:72]}…")
        t0 = time.time()
        try:
            if with_answers:
                out = _compare_via_api(question)
                minilm_src = enrich_sources(
                    [{"document": s["document"], "chunk": s["chunk"], "score": s["score"]}
                     for s in out.get("minilm", {}).get("sources", [])]
                )
                bge_src = enrich_sources(
                    [{"document": s["document"], "chunk": s["chunk"], "score": s["score"]}
                     for s in out.get("bge", {}).get("sources", [])]
                )
                minilm_answer = out.get("minilm", {}).get("answer", "")
                bge_answer = out.get("bge", {}).get("answer", "")
            else:
                out = collect_retrieval(question, top_k=top_k, history=[])
                minilm_src = out["minilm"]["sources"]
                bge_src = out["bge"]["sources"]
                minilm_answer = ""
                bge_answer = ""

            log_retrieval_run(question, minilm_src, bge_src)

            qid = f"q_{i:04d}"
            question_rows.append({
                "id": qid,
                "question": question,
                "expected_document": doc_stem,
                "gold_chunk_ids": [],
                "positive_chunk_ids": [],
                "neutral_chunk_ids": [],
                "negative_chunk_ids": [],
                "split": split,
                "source": "collect_40",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

            entry = {
                "id": qid,
                "question": question,
                "expected_document": doc_stem,
                "split": split,
                "minilm_answer": minilm_answer,
                "bge_answer": bge_answer,
                "minilm_sources": _trim_sources(minilm_src, top_k),
                "bge_sources": _trim_sources(bge_src, top_k),
                "labeling": {
                    "positive_chunk_ids": [],
                    "neutral_chunk_ids": [],
                    "negative_chunk_ids": [],
                    "notes": "Fill 2 positive, 1 neutral, 2 negative after reviewing sources/answers",
                },
                "elapsed_s": round(time.time() - t0, 1),
            }
            bundle.append(entry)
            ok += 1
            print(
                f"         ok ({entry['elapsed_s']}s) "
                f"bge_top1={bge_src[0].get('chunk_id') if bge_src else 'none'}"
            )
        except Exception as e:
            err += 1
            print(f"         ERROR: {e}")
            bundle.append({
                "id": f"q_{i:04d}",
                "question": question,
                "expected_document": doc_stem,
                "error": str(e),
            })

    save_questions(question_rows)

    # Overwrite retrieval_runs with fresh log from this run only if we want clean slate
    # (log_retrieval_run appends — for 40 fresh questions that's fine)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(QUESTIONS_40),
        "ok": ok,
        "errors": err,
        "with_answers": with_answers,
        "next_step": "Review collect_40.json then run label_40_questions.py",
        "questions": bundle,
    }
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {ok}/40 to:")
    print(f"  {Config.QUESTIONS_JSONL}")
    print(f"  {Config.RETRIEVAL_RUNS_JSONL}")
    print(f"  {EXPORT_PATH}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Collect 40 RAG questions with retrieval")
    parser.add_argument("--with-answers", action="store_true", help="Call Groq via API (needs quota)")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    result = collect(with_answers=args.with_answers, top_k=args.top_k)
    print(json.dumps({"ok": result["ok"], "errors": result["errors"], "export": EXPORT_PATH}, indent=2))


if __name__ == "__main__":
    main()
