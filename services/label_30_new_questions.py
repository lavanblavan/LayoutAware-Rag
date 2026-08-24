"""
Auto-label the 30 NEW questions (collect_30_new.json).

Exports eval file for base vs fine-tuned comparison.

Usage:
  python services/label_30_new_questions.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.finetune_store import corpus_by_id

COLLECT_PATH = os.path.join(Config.FINETUNE_EXPORT_PATH, "collect_30_new.json")
LABELED_PATH = os.path.join(Config.FINETUNE_EXPORT_PATH, "collect_30_new_labeled.json")
EVAL_JSON = os.path.join(Config.FINETUNE_EXPORT_PATH, "new30_eval.json")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "what",
    "which", "who", "how", "when", "where", "why", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "and", "but", "or", "not",
}


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower())) - STOPWORDS


def _relevance(question: str, chunk_text: str, stem: str, expected: str, bge_score: float) -> float:
    qk = _keywords(question)
    ck = _keywords(chunk_text)
    overlap = (len(qk & ck) / len(qk)) if qk else 0.0
    doc_bonus = 0.30 if stem == expected else -0.20
    return overlap + doc_bonus + bge_score * 0.08


def _classify(question: str, expected: str, sources: list[dict], by_id: dict) -> dict:
    scored = []
    for s in sources:
        cid = s.get("chunk_id")
        if not cid or cid not in by_id:
            continue
        rec = by_id[cid]
        rel = _relevance(question, rec["text"], rec["stem"], expected, float(s.get("score", 0)))
        scored.append({"chunk_id": cid, "document": rec["stem"], "relevance": rel, "text": rec["text"]})

    if not scored:
        return {"positive": [], "neutral": [], "negative": []}

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    max_r = scored[0]["relevance"]
    min_r = scored[-1]["relevance"]
    all_same_doc = all(x["document"] == expected for x in scored)

    positive, neutral, negative = [], [], []
    if max_r < 0.18:
        return {"positive": [], "neutral": [], "negative": scored}
    if all_same_doc and min_r >= 0.25:
        return {"positive": scored, "neutral": [], "negative": []}

    for item in scored:
        r = item["relevance"]
        same = item["document"] == expected
        if r >= 0.38 or (same and r >= 0.26):
            positive.append(item)
        elif r >= 0.14 or (same and r >= 0.08):
            neutral.append(item)
        else:
            negative.append(item)

    if not positive and scored[0]["relevance"] >= 0.12:
        positive.append(scored[0])

    return {"positive": positive, "neutral": neutral, "negative": negative}


def label_all() -> dict:
    if not os.path.exists(COLLECT_PATH):
        raise FileNotFoundError(f"Run collect_30_new_questions.py first. Missing {COLLECT_PATH}")

    with open(COLLECT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    by_id = corpus_by_id()
    labeled = []
    eval_rows = []

    for q in data.get("questions", []):
        if q.get("error"):
            continue
        labels = _classify(q["question"], q["expected_document"], q.get("bge_sources", []), by_id)
        pos_ids = [x["chunk_id"] for x in labels["positive"]]
        entry = {
            **q,
            "positive_chunk_ids": pos_ids,
            "neutral_chunk_ids": [x["chunk_id"] for x in labels["neutral"]],
            "negative_chunk_ids": [x["chunk_id"] for x in labels["negative"]],
        }
        labeled.append(entry)
        if pos_ids:
            eval_rows.append({
                "id": q["id"],
                "question": q["question"],
                "expected_document": q["expected_document"],
                "split": "new_eval",
                "positive_chunk_ids": pos_ids,
            })
        print(f"{q['id']}  +{len(pos_ids)} labels  {q['question'][:55]}…")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(labeled),
        "eval_questions": len(eval_rows),
        "questions": labeled,
    }
    with open(LABELED_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved labeled → {LABELED_PATH}")
    print(f"Saved eval    → {EVAL_JSON}")
    return {"eval_questions": len(eval_rows), "eval_json": EVAL_JSON}


def main():
    print(json.dumps(label_all(), indent=2))


if __name__ == "__main__":
    main()
