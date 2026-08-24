"""
Build / export fine-tuning data from indexed library chunks.

Usage:
  python services/export_finetune.py backfill   # corpus from Documents/chunks/*.json
  python services/export_finetune.py stats
  python services/export_finetune.py export     # train.json after labeling questions
  python services/export_finetune.py template
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from services.finetune_store import (
    export_training_pairs,
    rebuild_corpus,
    save_document_chunks,
    stats,
    write_questions_template,
)


def backfill_from_chunks_dir():
    """Load existing Documents/chunks/*_chunks.json into finetune corpus."""
    chunks_dir = Config.CHUNKS_PATH
    if not os.path.isdir(chunks_dir):
        print(f"No chunks folder: {chunks_dir}")
        print("Run: python services/library_pipeline.py")
        return 0

    count = 0
    for name in sorted(os.listdir(chunks_dir)):
        if not name.endswith("_chunks.json"):
            continue
        path = os.path.join(chunks_dir, name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stem = data.get("stem") or name.replace("_chunks.json", "")
        pdf = data.get("document") or f"{stem}.pdf"
        texts = data.get("chunks", [])
        if texts and isinstance(texts[0], dict):
            texts = [c.get("text", "") for c in texts]
        save_document_chunks(stem, pdf, texts)
        count += 1
        print(f"  {stem}: {len(texts)} chunks")

    rebuild_corpus()
    print(f"Backfilled {count} document(s) → {Config.CORPUS_JSONL}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Fine-tune dataset export")
    parser.add_argument(
        "command",
        choices=["backfill", "stats", "export", "template"],
        help="backfill corpus | show stats | export train/test | write questions template",
    )
    args = parser.parse_args()

    if args.command == "backfill":
        backfill_from_chunks_dir()
        write_questions_template()
    elif args.command == "stats":
        print(json.dumps(stats(), indent=2))
    elif args.command == "export":
        print(json.dumps(export_training_pairs(), indent=2))
    elif args.command == "template":
        path = write_questions_template()
        print(f"Template written: {path}")


if __name__ == "__main__":
    main()
