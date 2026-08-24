"""
Auto-label collected questions for any set in services/questions_bank.json.

Usage:
  python services/label_questions.py --list
  python services/label_questions.py --set train40
  python services/label_questions.py --set new30
  python services/label_questions.py --set custom

Requires: python services/collect_questions.py --set <name> first
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger
from settings.Settings import Config
from services.finetune_store import corpus_by_id, save_questions
from services.question_utils import (
    BGE_PREFIX,
    BANK_PATH,
    classify,
    collect_path,
    eval_path,
    get_set_config,
    labeled_path,
    list_sets,
    load_bank,
    test_json_path,
    train_json_path,
    train_jsonl_path,
)

log = get_logger(__name__)


def label_set(set_name: str) -> dict:
    cfg = get_set_config(set_name)
    src = collect_path(set_name)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Missing {src}\nRun: python services/collect_questions.py --set {set_name}"
        )

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    by_id = corpus_by_id()
    labeled_questions = []
    train_rows = []
    test_rows = []
    eval_rows = []
    question_rows = []
    stats = {"positive": 0, "neutral": 0, "negative": 0, "questions": 0}

    for q in data.get("questions", []):
        if q.get("error"):
            continue
        stats["questions"] += 1
        labels = classify(
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
            },
        }
        labeled_questions.append(entry)

        pos_ids = entry["labeling"]["positive_chunk_ids"]
        if pos_ids and cfg.get("mode") == "eval":
            eval_rows.append({
                "id": q["id"],
                "question": q["question"],
                "expected_document": q["expected_document"],
                "split": q.get("split", cfg.get("default_split", "eval")),
                "positive_chunk_ids": pos_ids,
            })

        pos_texts = [x["text"] for x in labels["positive"]]
        if not pos_texts:
            continue

        row = {
            "id": q["id"],
            "split": q.get("split", "train"),
            "question": q["question"],
            "query": BGE_PREFIX + q["question"],
            "pos": pos_texts,
            "neg": [x["text"] for x in labels["negative"]],
            "neutral": [x["text"] for x in labels["neutral"]],
            "positive_chunk_ids": pos_ids,
            "negative_chunk_ids": entry["labeling"]["negative_chunk_ids"],
            "neutral_chunk_ids": entry["labeling"]["neutral_chunk_ids"],
        }
        split = q.get("split", "train")
        if split == "test":
            test_rows.append(row)
        elif split in ("train",):
            train_rows.append(row)

        if cfg.get("mode") == "finetune":
            question_rows.append({
                "id": q["id"],
                "question": q["question"],
                "expected_document": q["expected_document"],
                "gold_chunk_ids": pos_ids[:1],
                "positive_chunk_ids": pos_ids,
                "neutral_chunk_ids": entry["labeling"]["neutral_chunk_ids"],
                "negative_chunk_ids": entry["labeling"]["negative_chunk_ids"],
                "split": split,
                "source": f"auto_labeled_{set_name}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        c = labels["counts"]
        log.info("%s  +%s ~%s -%s  %s…", q["id"], c["positive"], c["neutral"], c["negative"], q["question"][:55])

    if question_rows:
        save_questions(question_rows)

    labeled_out = labeled_path(set_name)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "set": set_name,
        "method": "relevance_score_adaptive",
        "total_questions": stats["questions"],
        "total_labels": stats,
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
        "eval_pairs": len(eval_rows),
        "questions": labeled_questions,
    }
    with open(labeled_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    files = {"labeled": labeled_out}

    if train_rows:
        tjson = train_json_path(set_name)
        tjsonl = train_jsonl_path(set_name)
        with open(tjson, "w", encoding="utf-8") as f:
            json.dump(train_rows, f, indent=2, ensure_ascii=False)
        with open(tjsonl, "w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps({"query": row["query"], "pos": row["pos"], "neg": row["neg"]}, ensure_ascii=False) + "\n")
        files["train_json"] = tjson
        files["train_jsonl"] = tjsonl

    if test_rows:
        tpath = test_json_path(set_name)
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump(test_rows, f, indent=2, ensure_ascii=False)
        files["test_json"] = tpath

    if eval_rows:
        epath = eval_path(set_name)
        with open(epath, "w", encoding="utf-8") as f:
            json.dump(eval_rows, f, indent=2, ensure_ascii=False)
        files["eval_json"] = epath

    summary = {
        "set": set_name,
        "labeled_questions": stats["questions"],
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "eval_rows": len(eval_rows),
        "files": files,
        "bank": str(BANK_PATH),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Label collected question sets")
    parser.add_argument("--set", help="Set name (train40, new30, custom, …)")
    parser.add_argument("--list", action="store_true", help="List available sets")
    args = parser.parse_args()

    if args.list:
        for name in list_sets():
            cfg = load_bank()["sets"][name]
            log.info("  %-12s  mode=%s  — %s", name, cfg.get("mode", "eval"), cfg.get("title", ""))
        log.info("Bank file: %s", BANK_PATH)
        return

    if not args.set:
        parser.error("Pass --set NAME or --list")

    summary = label_set(args.set)
    log.info("--- Labeling complete ---")
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    configure_logging()
    main()
