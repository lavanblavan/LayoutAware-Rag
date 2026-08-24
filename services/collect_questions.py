"""
Collect retrieval results for any question set in services/questions_bank.json.

Usage:
  python services/collect_questions.py --list
  python services/collect_questions.py --set train40
  python services/collect_questions.py --set new30
  python services/collect_questions.py --set custom
  python services/collect_questions.py --all

Edit questions in services/questions_bank.json under sets.<name>.questions
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

from utils.session_log import configure_logging, get_logger
from settings.Settings import Config
from services.compare_chat import collect_retrieval
from services.finetune_store import enrich_sources, log_retrieval_run, save_questions
from services.question_utils import (
    BANK_PATH,
    collect_path,
    get_set_config,
    list_sets,
    load_bank,
    normalize_questions,
    trim_sources,
)

log = get_logger(__name__)

API = os.getenv("RAG_API", "http://localhost:9091")


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


def collect_set(set_name: str, with_answers: bool = False, top_k: int = 8) -> dict:
    cfg = get_set_config(set_name)
    questions = normalize_questions(set_name)
    out_path = collect_path(set_name)

    bundle = []
    question_rows = []
    ok = err = 0
    total = len(questions)

    for i, q in enumerate(questions, start=1):
        log.info("[%s] [%02d/%d] %s…", set_name, i, total, q["question"][:72])
        t0 = time.time()
        try:
            if with_answers:
                out = _compare_via_api(q["question"])
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
                out = collect_retrieval(q["question"], top_k=top_k, history=[])
                minilm_src = out["minilm"]["sources"]
                bge_src = out["bge"]["sources"]
                minilm_answer = ""
                bge_answer = ""

            if cfg.get("mode") == "finetune":
                log_retrieval_run(q["question"], minilm_src, bge_src)
                question_rows.append({
                    "id": q["id"],
                    "question": q["question"],
                    "expected_document": q["expected_document"],
                    "gold_chunk_ids": [],
                    "positive_chunk_ids": [],
                    "neutral_chunk_ids": [],
                    "negative_chunk_ids": [],
                    "split": q["split"],
                    "source": f"collect_{set_name}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

            entry = {
                "id": q["id"],
                "question": q["question"],
                "expected_document": q["expected_document"],
                "split": q["split"],
                "minilm_answer": minilm_answer,
                "bge_answer": bge_answer,
                "minilm_sources": trim_sources(minilm_src, top_k),
                "bge_sources": trim_sources(bge_src, top_k),
                "elapsed_s": round(time.time() - t0, 1),
            }
            bundle.append(entry)
            ok += 1
            top1 = bge_src[0].get("chunk_id") if bge_src else "none"
            log.info("         ok (%ss) bge_top1=%s", entry["elapsed_s"], top1)
        except Exception as e:
            err += 1
            log.error("         ERROR: %s", e)
            bundle.append({**q, "error": str(e)})

    if question_rows:
        save_questions(question_rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "set": set_name,
        "title": cfg.get("title", set_name),
        "total": total,
        "ok": ok,
        "errors": err,
        "with_answers": with_answers,
        "bank": str(BANK_PATH),
        "next_step": f"python services/label_questions.py --set {set_name}",
        "questions": bundle,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Saved %s/%s → %s", ok, total, out_path)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Collect retrieval for question bank sets")
    parser.add_argument("--set", help="Set name in questions_bank.json (e.g. train40, new30, custom)")
    parser.add_argument("--all", action="store_true", help="Run every set in the bank")
    parser.add_argument("--list", action="store_true", help="List available sets")
    parser.add_argument("--with-answers", action="store_true", help="Also call Groq via API (slow)")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if args.list:
        bank = load_bank()
        for name in list_sets(bank):
            cfg = bank["sets"][name]
            n = len(cfg.get("questions", []))
            log.info("  %-12s  %3d questions  — %s", name, n, cfg.get("title", ""))
        log.info("Bank file: %s", BANK_PATH)
        return

    if args.all:
        for name in list_sets():
            collect_set(name, with_answers=args.with_answers, top_k=args.top_k)
        return

    if not args.set:
        parser.error("Pass --set NAME, --all, or --list")

    result = collect_set(args.set, with_answers=args.with_answers, top_k=args.top_k)
    log.info(json.dumps({"set": args.set, "ok": result["ok"], "errors": result["errors"]}, indent=2))


if __name__ == "__main__":
    configure_logging()
    main()
