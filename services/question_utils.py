"""
Shared helpers for the unified question bank (collect + label).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config

BANK_PATH = Path(__file__).resolve().parent / "questions_bank.json"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "what",
    "which", "who", "how", "when", "where", "why", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "and", "but", "or", "not",
}


def export_dir() -> str:
    os.makedirs(Config.FINETUNE_EXPORT_PATH, exist_ok=True)
    return Config.FINETUNE_EXPORT_PATH


def collect_path(set_name: str) -> str:
    return os.path.join(export_dir(), f"collect_{set_name}.json")


def labeled_path(set_name: str) -> str:
    return os.path.join(export_dir(), f"labeled_{set_name}.json")


def eval_path(set_name: str) -> str:
    # Backward-compatible names for known sets
    legacy = {"new30": "new30_eval.json", "train40": "bge_test_40.json"}
    name = legacy.get(set_name, f"eval_{set_name}.json")
    return os.path.join(export_dir(), name)


def train_jsonl_path(set_name: str) -> str:
    legacy = {"train40": "bge_train_40.jsonl"}
    name = legacy.get(set_name, f"bge_train_{set_name}.jsonl")
    return os.path.join(export_dir(), name)


def train_json_path(set_name: str) -> str:
    legacy = {"train40": "bge_train_40.json"}
    name = legacy.get(set_name, f"bge_train_{set_name}.json")
    return os.path.join(export_dir(), name)


def test_json_path(set_name: str) -> str:
    legacy = {"train40": "bge_test_40.json"}
    name = legacy.get(set_name, f"bge_test_{set_name}.json")
    return os.path.join(export_dir(), name)


def load_bank() -> dict:
    if not BANK_PATH.exists():
        raise FileNotFoundError(f"Question bank not found: {BANK_PATH}")
    with open(BANK_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_sets(bank: dict | None = None) -> list[str]:
    bank = bank or load_bank()
    return sorted(bank.get("sets", {}).keys())


def get_set_config(set_name: str, bank: dict | None = None) -> dict:
    bank = bank or load_bank()
    sets = bank.get("sets", {})
    if set_name not in sets:
        raise KeyError(f"Unknown set '{set_name}'. Available: {', '.join(sorted(sets))}")
    cfg = dict(sets[set_name])
    cfg["name"] = set_name
    cfg.setdefault("id_prefix", "q")
    cfg.setdefault("default_split", "train")
    cfg.setdefault("mode", "eval")
    return cfg


def normalize_questions(set_name: str, bank: dict | None = None) -> list[dict]:
    cfg = get_set_config(set_name, bank)
    prefix = cfg["id_prefix"]
    default_split = cfg["default_split"]
    rows = []
    for i, raw in enumerate(cfg.get("questions", []), start=1):
        if not raw.get("question"):
            continue
        qid = raw.get("id") or f"{prefix}_{i:04d}"
        rows.append({
            "id": qid,
            "question": raw["question"].strip(),
            "expected_document": raw.get("document") or raw.get("expected_document", ""),
            "split": raw.get("split") or default_split,
        })
    if not rows:
        raise ValueError(f"Set '{set_name}' has no questions. Edit {BANK_PATH}")
    return rows


def trim_sources(sources: list[dict], n: int = 8) -> list[dict]:
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


def keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower())) - STOPWORDS


def relevance(question: str, chunk_text: str, stem: str, expected: str, bge_score: float) -> float:
    qk = keywords(question)
    ck = keywords(chunk_text)
    overlap = (len(qk & ck) / len(qk)) if qk else 0.0
    doc_bonus = 0.30 if stem == expected else -0.20
    return overlap + doc_bonus + bge_score * 0.08


def classify(question: str, expected: str, sources: list[dict], by_id: dict) -> dict:
    scored = []
    for s in sources:
        cid = s.get("chunk_id")
        if not cid or cid not in by_id:
            continue
        rec = by_id[cid]
        rel = relevance(question, rec["text"], rec["stem"], expected, float(s.get("score", 0)))
        scored.append({
            "chunk_id": cid,
            "document": rec["stem"],
            "heading": rec.get("heading", "")[:100],
            "text": rec["text"],
            "bge_score": s.get("score"),
            "relevance": round(rel, 4),
        })

    if not scored:
        return {"positive": [], "neutral": [], "negative": [], "scored": [], "counts": {"positive": 0, "neutral": 0, "negative": 0}}

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    max_r = scored[0]["relevance"]
    min_r = scored[-1]["relevance"]
    all_same_doc = all(x["document"] == expected for x in scored)

    positive, neutral, negative = [], [], []

    if max_r < 0.18:
        negative = scored
        return _pack(positive, neutral, negative, scored)

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

    if not positive and scored[0]["relevance"] >= 0.12:
        positive.append(scored[0])
        neutral = [x for x in neutral if x["chunk_id"] != scored[0]["chunk_id"]]
        negative = [x for x in negative if x["chunk_id"] != scored[0]["chunk_id"]]

    if not positive and not negative and neutral:
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
