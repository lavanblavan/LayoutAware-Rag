"""
Compare retrieval BEFORE vs AFTER BGE fine-tuning.

Encodes the same corpus chunks with each model (fair comparison).
Reports Recall@1/@3/@5 and saves side-by-side JSON.

Usage:
  python services/compare_bge_models.py              # 5 test questions
  python services/compare_bge_models.py --split all  # all 25 questions

Requires fine-tuned model at Documents/Models/bge-rag-finetuned
(train first: python services/train_bge.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.finetune_store import load_corpus
from utils.torch_win import bootstrap_torch
from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_FINETUNED = os.path.normpath(
    os.path.join(os.path.dirname(Config.FINETUNE_EXPORT_PATH), "..", "Models", "bge-rag-finetuned")
)
EXPORT_COMPARE = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_before_after_compare.json")


def _load_pairs(split: str | None) -> list[dict]:
    path = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_finetune.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    if split:
        pairs = [p for p in pairs if p.get("split") == split]
    return pairs


def _encode_queries(model, questions: list[str]) -> np.ndarray:
    texts = [BGE_PREFIX + q for q in questions]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")


def _encode_corpus(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def _search(query_vec: np.ndarray, chunk_embs: np.ndarray, chunk_ids: list[str], top_k: int) -> list[tuple[str, float]]:
    scores = (chunk_embs @ query_vec.T).flatten()
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]


def _rank_of(gold_id: str, ranked: list[tuple[str, float]]) -> int | None:
    for i, (cid, _) in enumerate(ranked, start=1):
        if cid == gold_id:
            return i
    return None


def compare(
    base_model: str = "BAAI/bge-small-en-v1.5",
    finetuned_path: str = DEFAULT_FINETUNED,
    split: str = "test",
    top_k: int = 8,
) -> dict:
    if not os.path.isdir(finetuned_path):
        raise FileNotFoundError(
            f"Fine-tuned model not found: {finetuned_path}\nTrain first: python services/train_bge.py"
        )

    split_filter = None if split == "all" else split
    pairs = _load_pairs(split_filter)
    if not pairs:
        raise ValueError(f"No questions for split={split}")

    corpus = load_corpus()
    chunk_ids = [c["id"] for c in corpus]
    chunk_texts = [c["text"] for c in corpus]
    id_to_heading = {c["id"]: c.get("heading", "") for c in corpus}

    bootstrap_torch()
    from sentence_transformers import SentenceTransformer

    log.info("Corpus: %s chunks from %s documents", len(corpus), len({c["stem"] for c in corpus}))
    log.info("Base:       %s", base_model)
    log.info("Fine-tuned: %s", finetuned_path)
    log.info("Evaluating %s questions (split=%s)", len(pairs), split)

    base = SentenceTransformer(base_model)
    tuned = SentenceTransformer(finetuned_path)

    log.info("Encoding corpus with BASE model…")
    base_embs = _encode_corpus(base, chunk_texts)
    log.info("Encoding corpus with FINE-TUNED model…")
    tuned_embs = _encode_corpus(tuned, chunk_texts)

    questions = [p["question"] for p in pairs]
    base_q = _encode_queries(base, questions)
    tuned_q = _encode_queries(tuned, questions)

    rows = []
    metrics = {m: {"base": 0, "tuned": 0} for m in ("r1", "r3", "r5")}

    for i, p in enumerate(pairs):
        gold_id = p.get("positive_chunk_id")
        before = _search(base_q[i], base_embs, chunk_ids, top_k)
        after = _search(tuned_q[i], tuned_embs, chunk_ids, top_k)

        b_rank = _rank_of(gold_id, before)
        a_rank = _rank_of(gold_id, after)

        for k, key in ((1, "r1"), (3, "r3"), (5, "r5")):
            if b_rank and b_rank <= k:
                metrics[key]["base"] += 1
            if a_rank and a_rank <= k:
                metrics[key]["tuned"] += 1

        row = {
            "id": p.get("id"),
            "question": p["question"],
            "document": p.get("document"),
            "gold_chunk_id": gold_id,
            "gold_heading": id_to_heading.get(gold_id, ""),
            "before": {
                "gold_rank": b_rank,
                "top1_chunk_id": before[0][0] if before else None,
                "top1_score": round(before[0][1], 4) if before else None,
                "top3": [{"chunk_id": c, "score": round(s, 4)} for c, s in before[:3]],
            },
            "after": {
                "gold_rank": a_rank,
                "top1_chunk_id": after[0][0] if after else None,
                "top1_score": round(after[0][1], 4) if after else None,
                "top3": [{"chunk_id": c, "score": round(s, 4)} for c, s in after[:3]],
            },
            "improved": (a_rank or 99) < (b_rank or 99),
            "regressed": (a_rank or 99) > (b_rank or 99),
        }
        rows.append(row)

        arrow = " ↑" if row["improved"] else (" ↓" if row["regressed"] else " =")
        log.info("%s  gold rank %s → %s%s", p.get("id"), b_rank, a_rank, arrow)
        log.info("  %s", p["question"][:75])

    n = len(pairs)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "finetuned_model": finetuned_path,
        "split": split,
        "questions": n,
        "metrics": {
            "base": {f"recall@{k}": round(metrics[f"r{k}"]["base"] / n, 3) for k in (1, 3, 5)},
            "finetuned": {f"recall@{k}": round(metrics[f"r{k}"]["tuned"] / n, 3) for k in (1, 3, 5)},
            "delta_recall@1": round((metrics["r1"]["tuned"] - metrics["r1"]["base"]) / n, 3),
            "improved": sum(1 for r in rows if r["improved"]),
            "regressed": sum(1 for r in rows if r["regressed"]),
        },
        "comparisons": rows,
    }

    os.makedirs(Config.FINETUNE_EXPORT_PATH, exist_ok=True)
    with open(EXPORT_COMPARE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 55)
    log.info("%18s %12s %12s", "", "Base BGE", "Fine-tuned")
    log.info("%-18s %12.3f %12.3f", "Recall@1", summary["metrics"]["base"]["recall@1"], summary["metrics"]["finetuned"]["recall@1"])
    log.info("%-18s %12.3f %12.3f", "Recall@3", summary["metrics"]["base"]["recall@3"], summary["metrics"]["finetuned"]["recall@3"])
    log.info("%-18s %12.3f %12.3f", "Recall@5", summary["metrics"]["base"]["recall@5"], summary["metrics"]["finetuned"]["recall@5"])
    log.info("Improved: %s  Regressed: %s", summary["metrics"]["improved"], summary["metrics"]["regressed"])
    log.info("Report: %s", EXPORT_COMPARE)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--finetuned", default=DEFAULT_FINETUNED)
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    compare(args.base, args.finetuned, args.split, args.top_k)


if __name__ == "__main__":
    configure_logging()
    main()
