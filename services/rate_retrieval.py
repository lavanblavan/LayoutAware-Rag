"""
Compare and rate retrieval: base BGE vs fine-tuned BGE vs base BGE + reranker.

Uses the 40-question dataset (proper train/test split) or ad-hoc new questions.

Usage:
  python services/rate_retrieval.py --split test
  python services/rate_retrieval.py --split test --finetuned Documents/Models/bge-rag-finetuned-40-1ep
  python services/rate_retrieval.py --question "What is GraphRAG community detection?"
  python services/rate_retrieval.py --questions-file Documents/finetune/export/bge_test_40.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.rerank_retrieval import (
    DEFAULT_BASE,
    DEFAULT_FINETUNED,
    DEFAULT_RERANKER,
    Method,
    RetrievalEngine,
)

EXPORT_DIR = Config.FINETUNE_EXPORT_PATH
DEFAULT_REPORT = os.path.join(EXPORT_DIR, "retrieval_rating_report.json")
TRAIN_40 = os.path.join(EXPORT_DIR, "bge_train_40.json")
TEST_40 = os.path.join(EXPORT_DIR, "bge_test_40.json")

METHODS: tuple[Method, ...] = ("base_bge", "finetuned_bge", "base_bge_rerank")
METHOD_LABELS = {
    "base_bge": "Base BGE",
    "finetuned_bge": "Fine-tuned BGE",
    "base_bge_rerank": "Base BGE + Reranker",
}


def _load_eval_questions(split: str | None, questions_file: str | None) -> list[dict]:
    if questions_file:
        with open(questions_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("questions") or data.get("pairs") or []

    path = TEST_40 if split == "test" else TRAIN_40 if split == "train" else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    if split == "all":
        rows = []
        for p in (TRAIN_40, TEST_40):
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    rows.extend(json.load(f))
        return rows

    raise FileNotFoundError(
        f"No questions for split={split}. Run label_40_questions.py or pass --questions-file."
    )


def _gold_ids(item: dict) -> list[str]:
    ids = item.get("positive_chunk_ids") or []
    if ids:
        return list(ids)
    single = item.get("positive_chunk_id") or item.get("gold_chunk_id")
    return [single] if single else []


def _best_gold_rank(ranked_ids: list[str], gold_ids: list[str]) -> int | None:
    if not gold_ids:
        return None
    best = None
    for gid in gold_ids:
        try:
            rank = ranked_ids.index(gid) + 1
        except ValueError:
            continue
        best = rank if best is None else min(best, rank)
    return best


def _recall_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def _quality_label(rank: int | None) -> str:
    if rank is None:
        return "poor"
    if rank == 1:
        return "excellent"
    if rank <= 3:
        return "good"
    if rank <= 5:
        return "ok"
    return "poor"


def _rate_question(engine: RetrievalEngine, item: dict, top_k: int = 5) -> dict:
    question = item["question"]
    gold_ids = _gold_ids(item)
    methods_out = {}

    for method in METHODS:
        if method == "finetuned_bge" and engine.finetuned is None:
            methods_out[method] = {"skipped": True, "reason": "fine-tuned model not loaded"}
            continue

        hits = engine.retrieve(question, method, top_k=top_k)
        ranked_ids = [h.chunk_id for h in hits]
        rank = _best_gold_rank(ranked_ids, gold_ids) if gold_ids else None

        methods_out[method] = {
            "gold_rank": rank,
            "quality": _quality_label(rank) if gold_ids else "unlabeled",
            "recall@1": _recall_at_k(rank, 1) if gold_ids else None,
            "recall@3": _recall_at_k(rank, 3) if gold_ids else None,
            "recall@5": _recall_at_k(rank, 5) if gold_ids else None,
            "top1": {
                "chunk_id": hits[0].chunk_id if hits else None,
                "score": round(hits[0].score, 4) if hits else None,
                "heading": hits[0].heading if hits else "",
                "document": hits[0].document if hits else "",
            },
            "top_k": [
                {
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 4),
                    "bi_score": round(h.bi_score, 4) if h.bi_score is not None else None,
                    "rerank_score": round(h.rerank_score, 4) if h.rerank_score is not None else None,
                    "heading": h.heading,
                    "document": h.document,
                }
                for h in hits
            ],
        }

    # Pick winner among methods that have a gold rank
    scored = [
        (m, methods_out[m]["gold_rank"])
        for m in METHODS
        if not methods_out[m].get("skipped") and methods_out[m].get("gold_rank") is not None
    ]
    winner = None
    if scored:
        winner = min(scored, key=lambda x: x[1])[0]

    return {
        "id": item.get("id"),
        "question": question,
        "expected_document": item.get("expected_document") or item.get("document"),
        "gold_chunk_ids": gold_ids,
        "split": item.get("split"),
        "winner": winner,
        "methods": methods_out,
    }


def _aggregate(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("gold_chunk_ids")]
    if not labeled:
        return {"labeled_questions": 0}

    summary = {"labeled_questions": len(labeled), "methods": {}}
    for method in METHODS:
        usable = [
            r for r in labeled
            if not r["methods"][method].get("skipped")
        ]
        if not usable:
            continue
        n = len(usable)
        summary["methods"][method] = {
            "recall@1": round(sum(1 for r in usable if r["methods"][method]["recall@1"]) / n, 3),
            "recall@3": round(sum(1 for r in usable if r["methods"][method]["recall@3"]) / n, 3),
            "recall@5": round(sum(1 for r in usable if r["methods"][method]["recall@5"]) / n, 3),
            "excellent": sum(1 for r in usable if r["methods"][method]["quality"] == "excellent"),
            "good": sum(1 for r in usable if r["methods"][method]["quality"] == "good"),
            "ok": sum(1 for r in usable if r["methods"][method]["quality"] == "ok"),
            "poor": sum(1 for r in usable if r["methods"][method]["quality"] == "poor"),
            "wins": sum(1 for r in usable if r.get("winner") == method),
        }
    return summary


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    n = summary.get("labeled_questions", 0)
    print("\n" + "=" * 72)
    print(f"Retrieval rating — {n} labeled question(s)")
    print("=" * 72)
    print(f"{'Method':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'Wins':>6}  Quality (ex/gd/ok/poor)")
    print("-" * 72)
    for method in METHODS:
        m = summary.get("methods", {}).get(method)
        if not m:
            continue
        q = f"{m['excellent']}/{m['good']}/{m['ok']}/{m['poor']}"
        print(
            f"{METHOD_LABELS[method]:<22} "
            f"{m['recall@1']:>6.3f} {m['recall@3']:>6.3f} {m['recall@5']:>6.3f} "
            f"{m['wins']:>6}  {q}"
        )

    print("\nPer question:")
    for row in report["questions"]:
        qid = row.get("id") or "new"
        print(f"\n  [{qid}] {row['question'][:70]}")
        if not row.get("gold_chunk_ids"):
            print("    (no gold labels — showing top-1 only)")
        for method in METHODS:
            md = row["methods"][method]
            if md.get("skipped"):
                print(f"    {METHOD_LABELS[method]:<22} skipped")
                continue
            rank = md.get("gold_rank")
            rank_s = str(rank) if rank is not None else "—"
            top1 = md["top1"]["chunk_id"] or "—"
            qual = md.get("quality", "")
            mark = " *" if row.get("winner") == method else ""
            print(f"    {METHOD_LABELS[method]:<22} rank {rank_s:<3} [{qual:<9}] top1={top1}{mark}")

    print(f"\nReport: {report['report_path']}")


def rate(
    split: str = "test",
    questions_file: str | None = None,
    single_question: str | None = None,
    base_model: str = DEFAULT_BASE,
    finetuned_path: str = DEFAULT_FINETUNED,
    reranker_model: str = DEFAULT_RERANKER,
    retrieve_n: int = 30,
    top_k: int = 5,
    report_path: str = DEFAULT_REPORT,
) -> dict:
    if single_question:
        items = [{"id": "adhoc", "question": single_question.strip(), "positive_chunk_ids": []}]
    else:
        items = _load_eval_questions(split if not questions_file else None, questions_file)
        if split and not questions_file and split != "all":
            items = [q for q in items if q.get("split") == split or not q.get("split")]

    if not items:
        raise ValueError("No questions to evaluate")

    engine = RetrievalEngine(
        base_model=base_model,
        finetuned_path=finetuned_path,
        reranker_model=reranker_model,
        retrieve_n=retrieve_n,
    )

    print(f"\nRating {len(items)} question(s)…\n")
    rows = [_rate_question(engine, item, top_k=top_k) for item in items]
    summary = _aggregate(rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split if not single_question else "adhoc",
        "base_model": base_model,
        "finetuned_model": finetuned_path if os.path.isdir(finetuned_path) else None,
        "reranker_model": reranker_model,
        "retrieve_n": retrieve_n,
        "top_k": top_k,
        "summary": summary,
        "questions": rows,
        "report_path": os.path.abspath(report_path),
    }

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_summary(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Rate base vs fine-tuned vs reranked retrieval")
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    parser.add_argument("--questions-file", help="JSON list of questions (with optional positive_chunk_ids)")
    parser.add_argument("--question", help="Single ad-hoc question (retrieval only, no gold rating)")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--finetuned", default=DEFAULT_FINETUNED)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--retrieve-n", type=int, default=30, help="Bi-encoder pool size before rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    rate(
        split=args.split,
        questions_file=args.questions_file,
        single_question=args.question,
        base_model=args.base,
        finetuned_path=args.finetuned,
        reranker_model=args.reranker,
        retrieve_n=args.retrieve_n,
        top_k=args.top_k,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
