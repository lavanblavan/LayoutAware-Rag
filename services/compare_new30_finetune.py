"""
Compare base BGE vs fine-tuned BGE on 30 NEW questions (never seen in training).

Usage:
  python services/collect_30_new_questions.py
  python services/label_30_new_questions.py
  python services/compare_new30_finetune.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.rerank_retrieval import DEFAULT_BASE, DEFAULT_FINETUNED, RetrievalEngine

EVAL_JSON = os.path.join(Config.FINETUNE_EXPORT_PATH, "new30_eval.json")
REPORT_JSON = os.path.join(Config.FINETUNE_EXPORT_PATH, "new30_base_vs_finetuned.json")


def _best_gold_rank(ranked_ids: list[str], gold_ids: list[str]) -> int | None:
    best = None
    for gid in gold_ids:
        try:
            rank = ranked_ids.index(gid) + 1
        except ValueError:
            continue
        best = rank if best is None else min(best, rank)
    return best


def compare(finetuned_path: str = DEFAULT_FINETUNED, top_k: int = 5) -> dict:
    if not os.path.exists(EVAL_JSON):
        raise FileNotFoundError(
            f"Missing {EVAL_JSON}\n"
            "Run: python services/collect_30_new_questions.py && python services/label_30_new_questions.py"
        )

    with open(EVAL_JSON, encoding="utf-8") as f:
        items = json.load(f)

    engine = RetrievalEngine(
        base_model=DEFAULT_BASE,
        finetuned_path=finetuned_path,
        load_reranker=False,
    )

    rows = []
    base_hits = {1: 0, 3: 0, 5: 0}
    tuned_hits = {1: 0, 3: 0, 5: 0}
    improved = regressed = same = 0

    print(f"\nComparing {len(items)} new questions (base vs fine-tuned)…\n")

    for item in items:
        question = item["question"]
        gold_ids = item["positive_chunk_ids"]

        base_hits_list = engine.retrieve(question, "base_bge", top_k=top_k)
        tuned_hits_list = engine.retrieve(question, "finetuned_bge", top_k=top_k)

        base_ids = [h.chunk_id for h in base_hits_list]
        tuned_ids = [h.chunk_id for h in tuned_hits_list]
        b_rank = _best_gold_rank(base_ids, gold_ids)
        t_rank = _best_gold_rank(tuned_ids, gold_ids)

        for k in (1, 3, 5):
            if b_rank and b_rank <= k:
                base_hits[k] += 1
            if t_rank and t_rank <= k:
                tuned_hits[k] += 1

        if (t_rank or 99) < (b_rank or 99):
            improved += 1
            arrow = "↑"
        elif (t_rank or 99) > (b_rank or 99):
            regressed += 1
            arrow = "↓"
        else:
            same += 1
            arrow = "="

        print(f"{item['id']}  base {b_rank} → tuned {t_rank} {arrow}  {question[:60]}…")

        rows.append({
            "id": item["id"],
            "question": question,
            "expected_document": item.get("expected_document"),
            "gold_chunk_ids": gold_ids,
            "base": {
                "gold_rank": b_rank,
                "top1_chunk_id": base_ids[0] if base_ids else None,
                "top3": base_ids[:3],
            },
            "finetuned": {
                "gold_rank": t_rank,
                "top1_chunk_id": tuned_ids[0] if tuned_ids else None,
                "top3": tuned_ids[:3],
            },
            "improved": (t_rank or 99) < (b_rank or 99),
            "regressed": (t_rank or 99) > (b_rank or 99),
        })

    n = len(items)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "30 new questions (not in original 40 training set)",
        "questions": n,
        "base_model": DEFAULT_BASE,
        "finetuned_model": finetuned_path,
        "metrics": {
            "base": {f"recall@{k}": round(base_hits[k] / n, 3) for k in (1, 3, 5)},
            "finetuned": {f"recall@{k}": round(tuned_hits[k] / n, 3) for k in (1, 3, 5)},
            "delta_recall@1": round((tuned_hits[1] - base_hits[1]) / n, 3),
            "improved": improved,
            "regressed": regressed,
            "unchanged": same,
        },
        "comparisons": rows,
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    m = report["metrics"]
    print("\n" + "=" * 55)
    print(f"{'':18} {'Base BGE':>12} {'Fine-tuned':>12}")
    print(f"{'Recall@1':18} {m['base']['recall@1']:>12.3f} {m['finetuned']['recall@1']:>12.3f}")
    print(f"{'Recall@3':18} {m['base']['recall@3']:>12.3f} {m['finetuned']['recall@3']:>12.3f}")
    print(f"{'Recall@5':18} {m['base']['recall@5']:>12.3f} {m['finetuned']['recall@5']:>12.3f}")
    print(f"\nImproved: {improved}  Regressed: {regressed}  Unchanged: {same}")
    print(f"Report: {REPORT_JSON}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", default=DEFAULT_FINETUNED)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    compare(args.finetuned, args.top_k)


if __name__ == "__main__":
    main()
