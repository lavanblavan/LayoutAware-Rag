"""
Build BGE fine-tuning dataset from 5 indexed RAG papers.

Steps:
  1. Run BGE retrieval for 25 questions (5 per document)
  2. Validate / correct gold chunk labels against corpus text
  3. Export Documents/finetune/export/bge_finetune.json (+ .jsonl)

Usage:
  python services/build_bge_finetune.py
  python services/build_bge_finetune.py --export-only   # skip retrieval, use questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.compare_chat import collect_retrieval
from services.finetune_store import (
    corpus_by_id,
    enrich_sources,
    load_questions,
    log_retrieval_run,
    save_questions,
    stats,
)
from services.retrieval_cache import get_index_store, warmup_retrieval

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# 25 questions — 5 per document (stem prefix matches finetune corpus)
QUESTION_SET = [
    # 01 Foundational RAG (Lewis et al.)
    ("01_Foundational_RAG_Lewis_2020", "How does Lewis et al. combine a neural retriever with a seq2seq generator?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "What is the difference between RAG-Token and RAG-Sequence?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "How does RAG perform on open-domain question answering compared to extractive models?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "What retriever does the original RAG paper use for Wikipedia passages?", "train"),
    ("01_Foundational_RAG_Lewis_2020", "How does RAG perform on FEVER fact verification?", "test"),
    # 02 RAG Survey (Gao et al.)
    ("02_RAG_Survey_Gao_2023", "What are naive RAG, advanced RAG, and modular RAG?", "train"),
    ("02_RAG_Survey_Gao_2023", "What quality scores are used to evaluate RAG systems?", "train"),
    ("02_RAG_Survey_Gao_2023", "Why is RAG robustness important when retrieved documents are noisy?", "train"),
    ("02_RAG_Survey_Gao_2023", "Which RAG tools and frameworks are discussed in the ecosystem survey?", "train"),
    ("02_RAG_Survey_Gao_2023", "What are retrieval, generation, and augmentation in the RAG framework?", "test"),
    # 03 Self-RAG (Asai et al.)
    ("03_Self_RAG_Asai_2023", "What is Self-RAG and what problem does it solve?", "train"),
    ("03_Self_RAG_Asai_2023", "What are reflection tokens in Self-RAG?", "train"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG decide whether to retrieve additional passages?", "train"),
    ("03_Self_RAG_Asai_2023", "What is the retrieve-generate-critique loop in Self-RAG?", "train"),
    ("03_Self_RAG_Asai_2023", "How does Self-RAG improve factuality over standard RAG?", "test"),
    # 04 Corrective RAG
    ("04_Corrective_RAG_CRAG_2024", "What is Corrective RAG (CRAG)?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "What failure modes does CRAG address when retrieval returns bad documents?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "How does the CRAG retrieval evaluator assign Correct, Incorrect, or Ambiguous?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "Which datasets were used to evaluate CRAG?", "train"),
    ("04_Corrective_RAG_CRAG_2024", "What is the decompose-then-recompose algorithm in CRAG?", "test"),
    # 05 GraphRAG (Microsoft)
    ("05_GraphRAG_Microsoft_2024", "What is GraphRAG and how does it differ from standard vector RAG?", "train"),
    ("05_GraphRAG_Microsoft_2024", "How does GraphRAG build a knowledge graph from source documents?", "train"),
    ("05_GraphRAG_Microsoft_2024", "What are the main pipeline stages in GraphRAG?", "train"),
    ("05_GraphRAG_Microsoft_2024", "When does GraphRAG outperform baseline RAG on global sensemaking queries?", "train"),
    ("05_GraphRAG_Microsoft_2024", "What evaluation criteria does GraphRAG use for global sensemaking answers?", "test"),
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "and", "but", "or",
    "if", "because", "until", "while", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "we", "our", "you", "your", "about", "over",
}


def _keywords(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))
    return words - STOPWORDS


def _score_chunk(question: str, chunk_text: str, stem: str, expected_stem: str, bge_score: float) -> float:
    q_kw = _keywords(question)
    c_kw = _keywords(chunk_text)
    if not q_kw:
        overlap = 0.0
    else:
        overlap = len(q_kw & c_kw) / len(q_kw)
    doc_bonus = 0.25 if stem == expected_stem else -0.15
    return overlap + doc_bonus + (bge_score * 0.05)


def _pick_gold(
    question: str,
    bge_sources: list[dict],
    expected_stem: str,
    by_id: dict[str, dict],
) -> tuple[str | None, dict, float]:
    """Return (chunk_id, chunk_record, validation_score)."""
    best = None
    best_score = -1.0
    for hit in bge_sources:
        cid = hit.get("chunk_id")
        if not cid or cid not in by_id:
            continue
        rec = by_id[cid]
        score = _score_chunk(question, rec["text"], rec["stem"], expected_stem, hit.get("score", 0))
        if score > best_score:
            best_score = score
            best = (cid, rec, score)
    return best or (None, {}, 0.0)


def _hard_negatives(
    positive_id: str,
    bge_sources: list[dict],
    by_id: dict[str, dict],
    k: int = 5,
) -> tuple[list[str], list[str]]:
    texts, ids = [], []
    for hit in bge_sources:
        cid = hit.get("chunk_id")
        if not cid or cid == positive_id or cid not in by_id:
            continue
        texts.append(by_id[cid]["text"])
        ids.append(cid)
        if len(texts) >= k:
            break
    return texts, ids


def _collect_and_label(top_k: int = 10) -> list[dict]:
    store = get_index_store()
    if not store.ready:
        warmup_retrieval()

    by_id = corpus_by_id()
    rows = []

    for i, (expected_stem, question, split) in enumerate(QUESTION_SET, start=1):
        print(f"[{i:02d}/25] {question[:70]}…")
        out = collect_retrieval(question, top_k=top_k, history=[])
        bge_sources = out["bge"]["sources"]

        # Also fetch extra BGE hits for better gold selection
        extra = enrich_sources(
            store.search("bge", question, top_k_per_doc=6)[:top_k]
        )
        seen = {s.get("chunk_id") for s in bge_sources}
        merged = list(bge_sources)
        for h in extra:
            if h.get("chunk_id") not in seen:
                merged.append(h)
                seen.add(h.get("chunk_id"))

        gold_id, gold_rec, val_score = _pick_gold(question, merged, expected_stem, by_id)
        neg_texts, neg_ids = _hard_negatives(gold_id or "", merged, by_id, k=5)

        bge_top1 = bge_sources[0] if bge_sources else {}
        label_ok = gold_rec.get("stem") == expected_stem and val_score >= 0.15

        row = {
            "id": f"q_{i:04d}",
            "question": question,
            "expected_document": expected_stem,
            "gold_chunk_ids": [gold_id] if gold_id else [],
            "split": split,
            "source": "bge_validated",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation": {
                "score": round(val_score, 4),
                "correct_document": gold_rec.get("stem") == expected_stem,
                "passed": label_ok,
                "bge_top1_chunk_id": bge_top1.get("chunk_id"),
                "bge_top1_score": bge_top1.get("score"),
                "gold_heading": gold_rec.get("heading", "")[:120],
            },
        }
        rows.append(row)
        status = "OK" if label_ok else "CHECK"
        print(f"         {status} gold={gold_id} val={val_score:.2f} doc={gold_rec.get('stem', '?')}")

    save_questions(rows)
    return rows


def _build_export_rows(labeled: list[dict], by_id: dict[str, dict]) -> list[dict]:
    pairs = []
    for q in labeled:
        if not q.get("gold_chunk_ids"):
            continue
        cid = q["gold_chunk_ids"][0]
        rec = by_id.get(cid)
        if not rec:
            continue

        # Re-run negatives from stored retrieval if needed — use BGE search
        store = get_index_store()
        hits = enrich_sources(store.search("bge", q["question"], top_k_per_doc=6)[:10])
        neg_texts, neg_ids = _hard_negatives(cid, hits, by_id, k=5)

        pairs.append({
            "id": q["id"],
            "split": q.get("split", "train"),
            "document": q.get("expected_document", rec["stem"]),
            "question": q["question"],
            "query": BGE_QUERY_PREFIX + q["question"],
            "positive": rec["text"],
            "positive_chunk_id": cid,
            "positive_heading": rec.get("heading", ""),
            "negative": neg_texts,
            "negative_chunk_ids": neg_ids,
            "validation": q.get("validation", {}),
        })
    return pairs


def export_bge_json(labeled: list[dict] | None = None) -> dict:
    os.makedirs(Config.FINETUNE_EXPORT_PATH, exist_ok=True)
    by_id = corpus_by_id()
    labeled = labeled or load_questions()

    pairs = _build_export_rows(labeled, by_id)
    train = [p for p in pairs if p["split"] != "test"]
    test = [p for p in pairs if p["split"] == "test"]

    out_all = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_finetune.json")
    out_train = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_train.json")
    out_test = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_test.json")
    out_jsonl = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_train.jsonl")

    payload = {
        "model": "BAAI/bge-small-en-v1.5",
        "query_prefix": BGE_QUERY_PREFIX,
        "documents": sorted({q[0] for q in QUESTION_SET}),
        "total_pairs": len(pairs),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs,
    }

    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps({
                "query": p["query"],
                "pos": [p["positive"]],
                "neg": p["negative"],
            }, ensure_ascii=False) + "\n")

    passed = sum(1 for q in labeled if q.get("validation", {}).get("passed"))
    return {
        "status": "ok",
        "questions": len(labeled),
        "export_pairs": len(pairs),
        "validation_passed": passed,
        "train": len(train),
        "test": len(test),
        "paths": {
            "all": out_all,
            "train": out_train,
            "test": out_test,
            "jsonl": out_jsonl,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build BGE finetune dataset (25 Q, 5 docs)")
    parser.add_argument("--export-only", action="store_true", help="Skip retrieval, export from questions.jsonl")
    args = parser.parse_args()

    if args.export_only:
        labeled = load_questions()
    else:
        labeled = _collect_and_label(top_k=10)

    result = export_bge_json(labeled)
    result["store"] = stats()
    print("\n--- BGE finetune export ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
