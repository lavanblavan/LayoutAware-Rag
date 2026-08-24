"""
Collect 30 NEW questions (6 per paper) — separate from the original 40.

Does NOT overwrite questions.jsonl or the old collect_40.json.

Saves:
  Documents/finetune/export/collect_30_new.json

Usage:
  python services/collect_30_new_questions.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.compare_chat import collect_retrieval
from services.finetune_store import enrich_sources

EXPORT_PATH = os.path.join(Config.FINETUNE_EXPORT_PATH, "collect_30_new.json")

# 30 brand-new questions — 6 per document (none overlap with collect_40_questions.py)
QUESTIONS_30_NEW = [
    # 01 Lewis — 6
    ("01_Foundational_RAG_Lewis_2020", "How does RAG use DPR dense passage retrieval for Wikipedia?"),
    ("01_Foundational_RAG_Lewis_2020", "What Jeopardy question generation results does Lewis et al. report?"),
    ("01_Foundational_RAG_Lewis_2020", "How do human evaluations compare RAG and BART on factuality and specificity?"),
    ("01_Foundational_RAG_Lewis_2020", "What role does non-parametric memory play in the RAG architecture?"),
    ("01_Foundational_RAG_Lewis_2020", "How does RAG-Token select different retrieved documents for each output token?"),
    ("01_Foundational_RAG_Lewis_2020", "What MS-MARCO NLG results does the original RAG paper report?"),
    # 02 Survey — 6
    ("02_RAG_Survey_Gao_2023", "What chunking methods does the RAG survey describe for document indexing?"),
    ("02_RAG_Survey_Gao_2023", "How does query transformation improve RAG before retrieval?"),
    ("02_RAG_Survey_Gao_2023", "What reranking techniques does the RAG survey discuss?"),
    ("02_RAG_Survey_Gao_2023", "How is faithfulness evaluated in RAG systems according to the survey?"),
    ("02_RAG_Survey_Gao_2023", "What fusion methods combine multiple retrievers in advanced RAG?"),
    ("02_RAG_Survey_Gao_2023", "How does the survey classify augmentation methods during RAG generation?"),
    # 03 Self-RAG — 6
    ("03_Self_RAG_Asai_2023", "How is the Self-RAG critic model trained?"),
    ("03_Self_RAG_Asai_2023", "What do IsRel and IsSup reflection tokens mean during Self-RAG inference?"),
    ("03_Self_RAG_Asai_2023", "How does segment-level beam search work in Self-RAG?"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG perform on PubHealth and ASQA datasets?"),
    ("03_Self_RAG_Asai_2023", "What retrieval-on-demand mechanism does Self-RAG use at inference?"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG mark retrieved passages as relevant or irrelevant?"),
    # 04 CRAG — 6
    ("04_Corrective_RAG_CRAG_2024", "What is the T5-based retrieval evaluator in CRAG?"),
    ("04_Corrective_RAG_CRAG_2024", "How does CRAG refine knowledge strips from web search results?"),
    ("04_Corrective_RAG_CRAG_2024", "What PopQA and FactScore improvements does CRAG report over standard RAG?"),
    ("04_Corrective_RAG_CRAG_2024", "When does CRAG invoke query rewriting for ambiguous retrieval?"),
    ("04_Corrective_RAG_CRAG_2024", "What is the oracle upper bound for CRAG with a perfect retrieval evaluator?"),
    ("04_Corrective_RAG_CRAG_2024", "How does CRAG remove incorrect content from retrieved documents before generation?"),
    # 05 GraphRAG — 6
    ("05_GraphRAG_Microsoft_2024", "How does GraphRAG detect communities in the knowledge graph?"),
    ("05_GraphRAG_Microsoft_2024", "What map-reduce approach does GraphRAG use for global search queries?"),
    ("05_GraphRAG_Microsoft_2024", "What types of claims and entities does GraphRAG extract from text?"),
    ("05_GraphRAG_Microsoft_2024", "When does GraphRAG outperform naive RAG on local versus global tasks?"),
    ("05_GraphRAG_Microsoft_2024", "What are community summary reports in the GraphRAG pipeline?"),
    ("05_GraphRAG_Microsoft_2024", "How does GraphRAG use Leiden clustering in its indexing pipeline?"),
]


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


def collect(top_k: int = 8) -> dict:
    os.makedirs(Config.FINETUNE_EXPORT_PATH, exist_ok=True)
    bundle = []
    ok = err = 0

    for i, (doc_stem, question) in enumerate(QUESTIONS_30_NEW, start=1):
        qid = f"nq_{i:04d}"
        print(f"[{i:02d}/30] {question[:72]}…")
        t0 = time.time()
        try:
            out = collect_retrieval(question, top_k=top_k, history=[])
            bge_src = out["bge"]["sources"]
            minilm_src = out["minilm"]["sources"]

            entry = {
                "id": qid,
                "question": question,
                "expected_document": doc_stem,
                "split": "new_eval",
                "minilm_sources": _trim_sources(minilm_src, top_k),
                "bge_sources": _trim_sources(bge_src, top_k),
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
                "id": qid,
                "question": question,
                "expected_document": doc_stem,
                "error": str(e),
            })

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(QUESTIONS_30_NEW),
        "ok": ok,
        "errors": err,
        "note": "30 new questions — not used in fine-tuning. Next: python services/label_30_new_questions.py",
        "questions": bundle,
    }
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {ok}/30 to {EXPORT_PATH}")
    return payload


def main():
    collect()


if __name__ == "__main__":
    main()
