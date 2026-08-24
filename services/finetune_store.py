"""
Save and export data for bi-encoder fine-tuning (MiniLM / BGE).

Files under Documents/finetune/:
  corpus.jsonl          — every chunk with stable id
  chunks/{stem}.json    — per-document chunk records
  questions.jsonl       — questions + gold_chunk_ids (you label these)
  retrieval_runs.jsonl  — auto-saved each compare-chat query
  export/train.json     — sentence-transformers training pairs
  export/test.json      — held-out pairs
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config


def _ensure_finetune_dirs():
    os.makedirs(Config.FINETUNE_PATH, exist_ok=True)
    os.makedirs(Config.FINETUNE_CHUNKS_PATH, exist_ok=True)
    os.makedirs(Config.FINETUNE_EXPORT_PATH, exist_ok=True)


def chunk_id(stem: str, index: int) -> str:
    return f"{stem}::{index}"


def _heading_from_text(text: str) -> str:
    line = (text or "").strip().split("\n", 1)[0].strip()
    if line.startswith("[TABLE]"):
        return line[:120]
    if len(line) > 200:
        return line[:200] + "…"
    return line


def _chunk_type(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("[TABLE]") or "\n[TABLE]" in t[:80]:
        return "table"
    first = t.split("\n", 1)[0]
    if re.match(r"^\d+[A-Za-z]?\.\s", first) or first.isupper() and len(first) < 80:
        return "section"
    return "paragraph"


def build_chunk_records(stem: str, pdf_name: str, chunk_texts: list[str]) -> list[dict]:
    records = []
    for i, text in enumerate(chunk_texts):
        text = (text or "").strip()
        if not text:
            continue
        records.append({
            "id": chunk_id(stem, i),
            "stem": stem,
            "document": pdf_name,
            "index": i,
            "heading": _heading_from_text(text),
            "type": _chunk_type(text),
            "text": text,
            "char_count": len(text),
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        })
    return records


def save_document_chunks(stem: str, pdf_name: str, chunk_texts: list[str]) -> list[dict]:
    """Save per-doc chunk file and refresh master corpus."""
    _ensure_finetune_dirs()
    records = build_chunk_records(stem, pdf_name, chunk_texts)
    path = os.path.join(Config.FINETUNE_CHUNKS_PATH, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stem": stem, "document": pdf_name, "chunks": records}, f, indent=2, ensure_ascii=False)
    rebuild_corpus()
    return records


def rebuild_corpus():
    """Merge all per-document chunk files into corpus.jsonl."""
    _ensure_finetune_dirs()
    all_records = []
    if os.path.isdir(Config.FINETUNE_CHUNKS_PATH):
        for name in sorted(os.listdir(Config.FINETUNE_CHUNKS_PATH)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(Config.FINETUNE_CHUNKS_PATH, name), encoding="utf-8") as f:
                data = json.load(f)
            all_records.extend(data.get("chunks", []))
    with open(Config.CORPUS_JSONL, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return all_records


def load_corpus() -> list[dict]:
    if not os.path.exists(Config.CORPUS_JSONL):
        return rebuild_corpus()
    records = []
    with open(Config.CORPUS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def corpus_by_id() -> dict[str, dict]:
    return {r["id"]: r for r in load_corpus()}


def resolve_chunk_id(document_stem: str, chunk_text: str) -> Optional[str]:
    """Map retrieved chunk text back to corpus id."""
    text = (chunk_text or "").strip()
    if not text:
        return None
    path = os.path.join(Config.FINETUNE_CHUNKS_PATH, f"{document_stem}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.get("chunks", []):
            if rec.get("text", "").strip() == text:
                return rec["id"]
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    for rec in load_corpus():
        if rec.get("stem") == document_stem and rec.get("text_hash") == h:
            return rec["id"]
        if rec.get("text", "").strip() == text:
            return rec["id"]
    return None


def enrich_sources(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        stem = s.get("document", "")
        text = s.get("chunk", "")
        cid = resolve_chunk_id(stem, text)
        out.append({**s, "chunk_id": cid})
    return out


def _append_jsonl(path: str, record: dict):
    _ensure_finetune_dirs()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_retrieval_run(question: str, minilm_sources: list[dict], bge_sources: list[dict]):
    """Auto-save each compare-chat query for later labeling / hard negatives."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question.strip(),
        "minilm": enrich_sources(minilm_sources),
        "bge": enrich_sources(bge_sources),
    }
    _append_jsonl(Config.RETRIEVAL_RUNS_JSONL, record)


def load_questions() -> list[dict]:
    if not os.path.exists(Config.QUESTIONS_JSONL):
        return []
    rows = []
    with open(Config.QUESTIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_questions(rows: list[dict]):
    _ensure_finetune_dirs()
    with open(Config.QUESTIONS_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_question(
    question: str,
    gold_chunk_ids: Optional[list[str]] = None,
    split: str = "train",
    source: str = "manual",
) -> dict:
    rows = load_questions()
    qid = f"q_{len(rows)+1:04d}"
    rec = {
        "id": qid,
        "question": question.strip(),
        "gold_chunk_ids": gold_chunk_ids or [],
        "split": split,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    rows.append(rec)
    save_questions(rows)
    return rec


def update_question_labels(qid: str, gold_chunk_ids: list[str], split: Optional[str] = None) -> dict:
    rows = load_questions()
    for row in rows:
        if row.get("id") == qid:
            row["gold_chunk_ids"] = gold_chunk_ids
            if split:
                row["split"] = split
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_questions(rows)
            return row
    raise KeyError(f"Question not found: {qid}")


def _hard_negatives(question: str, positive_ids: set[str], model_key: str = "minilm", k: int = 5) -> list[str]:
    from services.retrieval_cache import get_index_store, warmup_retrieval

    store = get_index_store()
    if not store.ready:
        warmup_retrieval()
    hits = store.search(question, model_key=model_key, top_k_per_doc=10)
    negs = []
    for h in enrich_sources(hits[:20]):
        cid = h.get("chunk_id")
        if cid and cid not in positive_ids:
            negs.append(cid)
        if len(negs) >= k:
            break
    return negs


def export_training_pairs(test_ratio: float = 0.2) -> dict:
    """
    Export sentence-transformers JSON:
    [{"anchor": q, "positive": text, "negative": [text,...], "split": train|test}]
    """
    _ensure_finetune_dirs()
    by_id = corpus_by_id()
    questions = load_questions()
    labeled = [q for q in questions if q.get("gold_chunk_ids")]

    if not labeled:
        # Seed template from retrieval runs if no labels yet
        return {
            "status": "no_labels",
            "message": "Add gold_chunk_ids to questions.jsonl (see template). Retrieval runs are saved automatically.",
            "corpus_chunks": len(by_id),
            "questions": len(questions),
        }

    pairs = []
    for q in labeled:
        pos_texts = []
        for cid in q["gold_chunk_ids"]:
            rec = by_id.get(cid)
            if rec:
                pos_texts.append(rec["text"])
        if not pos_texts:
            continue
        positive = pos_texts[0]
        neg_ids = _hard_negatives(
            q["question"],
            set(q["gold_chunk_ids"]),
            model_key="minilm",
            k=5,
        )
        neg_texts = [by_id[n]["text"] for n in neg_ids if n in by_id]
        pairs.append({
            "id": q["id"],
            "anchor": q["question"],
            "positive": positive,
            "negative": neg_texts,
            "split": q.get("split", "train"),
            "gold_chunk_ids": q["gold_chunk_ids"],
        })

    train = [p for p in pairs if p["split"] != "test"]
    test = [p for p in pairs if p["split"] == "test"]

    train_path = os.path.join(Config.FINETUNE_EXPORT_PATH, "train.json")
    test_path = os.path.join(Config.FINETUNE_EXPORT_PATH, "test.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    # Also write JSONL for scripts
    st_path = os.path.join(Config.FINETUNE_EXPORT_PATH, "train.jsonl")
    with open(st_path, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps({
                "anchor": p["anchor"],
                "positive": p["positive"],
                "negative": p["negative"],
            }, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "train_pairs": len(train),
        "test_pairs": len(test),
        "train_path": train_path,
        "test_path": test_path,
        "train_jsonl": st_path,
    }


def write_questions_template():
    """Empty template for labeling — copy gold_chunk_ids from corpus ids."""
    _ensure_finetune_dirs()
    corpus = load_corpus()
    template = {
        "instructions": (
            "Add one object per line to questions.jsonl. "
            "gold_chunk_ids must match ids from corpus.jsonl (e.g. LK_Police_Ordinance::12). "
            "split: train or test."
        ),
        "example": {
            "id": "q_0001",
            "question": "What is the police arrest procedure?",
            "gold_chunk_ids": ["LK_Police_Ordinance::45"],
            "split": "train",
            "source": "manual",
        },
        "sample_chunk_ids": [c["id"] for c in corpus[:15]],
    }
    path = os.path.join(Config.FINETUNE_PATH, "questions_template.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    return path


def stats() -> dict:
    corpus = load_corpus()
    questions = load_questions()
    runs = 0
    if os.path.exists(Config.RETRIEVAL_RUNS_JSONL):
        with open(Config.RETRIEVAL_RUNS_JSONL, encoding="utf-8") as f:
            runs = sum(1 for line in f if line.strip())
    labeled = sum(1 for q in questions if q.get("gold_chunk_ids"))
    return {
        "corpus_chunks": len(corpus),
        "documents": len({c["stem"] for c in corpus}),
        "questions": len(questions),
        "labeled_questions": labeled,
        "retrieval_runs_logged": runs,
        "paths": {
            "corpus": Config.CORPUS_JSONL,
            "questions": Config.QUESTIONS_JSONL,
            "retrieval_runs": Config.RETRIEVAL_RUNS_JSONL,
            "export_dir": Config.FINETUNE_EXPORT_PATH,
        },
    }
