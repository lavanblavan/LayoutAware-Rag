"""
Auto-label 40 collected questions by reading BGE retrieval results.

Each chunk in top-8 is classified positive / neutral / negative (no fixed count).
  - All strong + same paper → all positive
  - All weak → all negative
  - Otherwise split by relevance score

Usage:
  python services/label_40_questions.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.finetune_store import corpus_by_id, save_questions

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECT_PATH = os.path.join(Config.FINETUNE_EXPORT_PATH, "collect_40.json")
LABELED_PATH = os.path.join(Config.FINETUNE_EXPORT_PATH, "collect_40_labeled.json")
TRAIN_JSONL = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_train_40.jsonl")
TRAIN_JSON = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_train_40.json")
TEST_JSON = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_test_40.json")

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
        scored.append({
            "chunk_id": cid,
            "document": rec["stem"],
            "heading": rec.get("heading", "")[:100],
            "text": rec["text"],
            "bge_score": s.get("score"),
            "relevance": round(rel, 4),
        })

    if not scored:
        return {"positive": [], "neutral": [], "negative": [], "scored": []}

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    max_r = scored[0]["relevance"]
    min_r = scored[-1]["relevance"]
    all_same_doc = all(x["document"] == expected for x in scored)

    positive, neutral, negative = [], [], []

    # All weak retrieval → all negative
    if max_r < 0.18:
        negative = scored
        return _pack(positive, neutral, negative, scored)

    # All strong from expected paper → all positive
    if all_same_doc and min_r >= 0.25:
        positive = scored
        return _pack(positive, neutral, negative, scored)

    for item in scored:
        r = item["relevance"]
        same = item["document"] == expected
        if r >= 0.38 or (same and r >= 0.26):
            positive.append(item)
        elif r >= 0.14 or (same and r >= 0.08):
            neutral.append(item)
        else:
            negative.append(item)

    # Must have at least one positive if anything is usable
    if not positive and scored[0]["relevance"] >= 0.12:
        positive.append(scored[0])
        neutral = [x for x in neutral if x["chunk_id"] != scored[0]["chunk_id"]]
        negative = [x for x in negative if x["chunk_id"] != scored[0]["chunk_id"]]

    # All classified as one type edge cases
    if not neutral and not negative and positive:
        pass
    elif not positive and not neutral and negative:
        pass
    elif not positive and not negative and neutral:
        # all neutral → treat best as positive, rest neutral
        positive = [neutral.pop(0)] if neutral else []

    return _pack(positive, neutral, negative, scored)


def _pack(positive, neutral, negative, scored):
    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "scored": scored,
        "counts": {
            "positive": len(positive),
            "neutral": len(neutral),
            "negative": len(negative),
        },
    }


def label_all() -> dict:
    if not os.path.exists(COLLECT_PATH):
        raise FileNotFoundError(f"Run collect_40_questions.py first. Missing {COLLECT_PATH}")

    with open(COLLECT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    by_id = corpus_by_id()
    labeled_questions = []
    train_rows = []
    test_rows = []
    question_rows = []

    stats = {"positive": 0, "neutral": 0, "negative": 0, "questions": 0}

    for q in data.get("questions", []):
        if q.get("error"):
            continue
        stats["questions"] += 1
        labels = _classify(
            q["question"],
            q["expected_document"],
            q.get("bge_sources", []),
            by_id,
        )
        stats["positive"] += labels["counts"]["positive"]
        stats["neutral"] += labels["counts"]["neutral"]
        stats["negative"] += labels["counts"]["negative"]

        entry = {
            **q,
            "labeling": {
                "positive_chunk_ids": [x["chunk_id"] for x in labels["positive"]],
                "neutral_chunk_ids": [x["chunk_id"] for x in labels["neutral"]],
                "negative_chunk_ids": [x["chunk_id"] for x in labels["negative"]],
                "positive": [{k: v for k, v in x.items() if k != "text"} for x in labels["positive"]],
                "neutral": [{k: v for k, v in x.items() if k != "text"} for x in labels["neutral"]],
                "negative": [{k: v for k, v in x.items() if k != "text"} for x in labels["negative"]],
            },
        }
        labeled_questions.append(entry)

        pos_texts = [x["text"] for x in labels["positive"]]
        neg_texts = [x["text"] for x in labels["negative"]]
        neu_texts = [x["text"] for x in labels["neutral"]]

        if not pos_texts:
            continue

        row = {
            "id": q["id"],
            "split": q.get("split", "train"),
            "question": q["question"],
            "query": BGE_PREFIX + q["question"],
            "pos": pos_texts,
            "neg": neg_texts,
            "neutral": neu_texts,
            "positive_chunk_ids": [x["chunk_id"] for x in labels["positive"]],
            "negative_chunk_ids": [x["chunk_id"] for x in labels["negative"]],
            "neutral_chunk_ids": [x["chunk_id"] for x in labels["neutral"]],
        }
        if q.get("split") == "test":
            test_rows.append(row)
        else:
            train_rows.append(row)

        question_rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_document": q["expected_document"],
            "gold_chunk_ids": [x["chunk_id"] for x in labels["positive"][:1]],
            "positive_chunk_ids": [x["chunk_id"] for x in labels["positive"]],
            "neutral_chunk_ids": [x["chunk_id"] for x in labels["neutral"]],
            "negative_chunk_ids": [x["chunk_id"] for x in labels["negative"]],
            "split": q.get("split", "train"),
            "source": "auto_labeled_40",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        c = labels["counts"]
        print(f"{q['id']}  +{c['positive']} ~{c['neutral']} -{c['negative']}  {q['question'][:55]}…")

    save_questions(question_rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "relevance_score_adaptive",
        "total_questions": stats["questions"],
        "total_labels": stats,
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
        "questions": labeled_questions,
    }
    with open(LABELED_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, indent=2, ensure_ascii=False)
    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=2, ensure_ascii=False)
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps({
                "query": row["query"],
                "pos": row["pos"],
                "neg": row["neg"],
            }, ensure_ascii=False) + "\n")

    summary = {
        "labeled_questions": stats["questions"],
        "total_positive_labels": stats["positive"],
        "total_neutral_labels": stats["neutral"],
        "total_negative_labels": stats["negative"],
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "files": {
            "labeled": LABELED_PATH,
            "train_jsonl": TRAIN_JSONL,
            "train_json": TRAIN_JSON,
            "test_json": TEST_JSON,
            "questions": Config.QUESTIONS_JSONL,
        },
        "train_command": "python services/train_bge.py --train Documents/finetune/export/bge_train_40.jsonl --output Documents/Models/bge-rag-finetuned-40 --epochs 1 --loss mnrl",
    }
    return summary


def main():
    summary = label_all()
    print("\n--- Labeling complete ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
