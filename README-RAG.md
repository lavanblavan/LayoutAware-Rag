# PaperRAG — How It Works

Architecture, retrieval, fine-tuning, validation, and the main failures we hit (and how we fixed them).

Setup instructions: **[README.md](README.md)**

---

## Problem

A **research-paper Q&A system** over five ML/RAG PDFs (Lewis 2020, Gao survey, Self-RAG, CRAG, GraphRAG). The system must retrieve the **right paper and section**, then generate an answer from those passages.

Similar papers share vocabulary (“RAG”, “retrieval”, “factuality”), so a general retriever often returns the **wrong document at rank 1**.

---

## End-to-end pipeline

```
PDF
  → layout detection (PubLayNet) + OCR (Tesseract)
  → layout-tagged text ([TITLE], [PARAGRAPH], [TABLE])
  → section chunks (one chunk per heading/table)
  → embeddings (MiniLM + BGE) + FAISS index
  → user question
  → retriever (top-k chunks)
  → Groq LLM (answer from excerpts only)
```

### 1. Extraction & layout

- PDF pages → images → **PubLayNet** detects Title, Text, List, Table regions.
- **Tesseract** OCR inside each region in reading order.
- Output: `Documents/Extracted_texts/*.txt` + `*_layout.json`.

Falls back to Tesseract paragraph blocks if PubLayNet is unavailable.

### 2. Chunking (fix #1 — layout-aware)

**Failure:** **HDBSCAN** on sentence embeddings grouped similar-sounding sentences from *different sections*. Chunks had no headings and started mid-paragraph.

**Fix:** Chunk **after** layout — one chunk per section heading or table.

| Layout chunk | HDBSCAN chunk |
|--------------|---------------|
| `2.1 Models` — RAG-Token vs RAG-Sequence | *"sequence-to-sequence models We endow…"* (mid-paragraph) |

~**406 section chunks** across five papers.

Demo: `python services/demo_chunk_compare.py` → `Documents/finetune/export/chunk_compare_layout_vs_hdbscan.md`

### 3. Indexing & retrieval

**FAISS** (`IndexFlatIP`, normalized vectors = cosine similarity).

| Model | Role |
|-------|------|
| `all-MiniLM-L6-v2` | Baseline (main compare page) |
| `BAAI/bge-small-en-v1.5` | Base BGE retriever |
| Fine-tuned BGE (1 epoch) | Domain-tuned retriever |

BGE query prefix: `Represent this sentence for searching relevant passages: <question>`

Flow: embed question → FAISS top-k → pass top 3–5 chunks to Groq.

### 4. Generation (Groq)

**Groq** (`openai/gpt-oss-20b` + fallbacks) answers from retrieved excerpts only. Prompt: *answer only from excerpts; say if unsupported*.

No separate automated faithfulness judge yet (planned next step).

---

## Question bank & labeling

All evaluation questions are in **`services/questions_bank.json`** — one file, multiple sets:

| Set | Count | Split | Mode | Use |
|-----|-------|-------|------|-----|
| `train40` | 40 | train/test | finetune | BGE fine-tuning |
| `new30` | 30 | new_eval | eval | Unseen generalization test |
| `custom` | yours | eval | eval | Add your own questions |

**Workflow:**

```powershell
python services/collect_questions.py --set <name>   # run MiniLM + BGE retrieval
python services/label_questions.py --set <name>       # auto-label positive/neutral/negative chunks
```

Labeling uses keyword overlap + BGE score + expected-document bonus (`services/question_utils.py`).

Outputs per set in `Documents/finetune/export/`:

| File | Content |
|------|---------|
| `collect_{set}.json` | Retrieval results |
| `labeled_{set}.json` | Auto-labels |
| `bge_train_40.jsonl` | MNRL training pairs (train40) |
| `new30_eval.json` | Gold chunks for eval (new30) |

---

## Fine-tuning (fix #2 — cross-paper confusion)

**Failure:** Base BGE returned wrong paper at rank 1 on overlapping terminology.

| Question | Expected | Base rank |
|----------|----------|-----------|
| Self-RAG factuality | Self-RAG | CRAG (#2) |
| Non-parametric memory | Lewis 2020 | Self-RAG (#2) |

**Fix:** 40 domain Q→chunk pairs, **MNRL**, **1 epoch** → `Documents/Models/bge-rag-finetuned-40-1ep`

| Eval set | Base Recall@1 | Fine-tuned Recall@1 |
|----------|---------------|---------------------|
| 10 held-out test Q | 80% | 80% (wins on hard Qs) |
| 30 new unseen Q | 93.3% | **96.7%** (0 regressions) |

Hard win: `q_0021` — rank **2 → 1**.

---

## Validation design

1. **Train/test split** — 30 train + 10 test from `train40` set.
2. **Independent eval** — `new30` set never used in training.
3. **Multi-gold labels** — several valid chunk IDs per question.
4. **Metrics** — Recall@1 / @3 / @5 on chunk IDs.

Scripts: `rate_retrieval.py`, `compare_new30_finetune.py`, `collect_questions.py`, `label_questions.py`.

---

## Three failures & lessons

### 1 — Dense HDBSCAN chunks mixed sections
- **Fix:** Layout-first chunking (PubLayNet → section chunks).
- **Lesson:** Chunk structure matters as much as the embedding model.

### 2 — Cross-paper retrieval confusion
- **Fix:** Domain fine-tuning (1 epoch MNRL on 40 pairs).
- **Lesson:** General embedders need adaptation on small specialized corpora.

### 3 — Overfitting & leaky evaluation
- **Overfitting:** 3 epochs regressed on held-out Qs → **1 epoch** chosen.
- **Leaky eval:** Early 5-Q test had 4/5 train questions → false 40% Recall@1 → fixed with proper splits + `new30`.

### Also tried — reranker hurt metrics
`bge-reranker-base`: Recall@1 **80% → 70%** on 10-Q test. Not deployed.

---

## Risks & limitations

| Risk | Status |
|------|--------|
| Train/test leakage | Mitigated with `new30` independent set |
| Overfitting | 1 epoch; per-question regression checks |
| OCR noise | Layout helps; some garbled text remains |
| Small training set | 40 pairs — corpus-specific gains |
| LLM hallucination | Prompt grounding only |
| Deployment | Groq API dependency; not production-hardened |

---

## Web UI

| Page | Retrieval | Use case |
|------|-----------|----------|
| `/` | MiniLM vs BGE | Baseline comparison |
| `/chat.html` | Fine-tuned BGE | Single chatbot demo |
| `/finetuned.html` | Base vs fine-tuned | Assessment side-by-side |

---

## Key files

| Path | Description |
|------|-------------|
| `services/questions_bank.json` | All question sets (edit here) |
| `services/collect_questions.py` | Unified retrieval collection |
| `services/label_questions.py` | Unified auto-labeling |
| `services/question_utils.py` | Shared bank + classify logic |
| `services/library_pipeline.py` | PDF → layout → chunks → FAISS |
| `Extractor_storing/create_chunks.py` | Layout + HDBSCAN chunking |
| `Summarizer/layout_extract.py` | PubLayNet + OCR |
| `services/compare_chat.py` | MiniLM/BGE + Groq |
| `services/finetuned_chat.py` | Fine-tuned store + chat |
| `services/train_bge.py` | BGE fine-tuning |
| `services/demo_chunk_compare.py` | Layout vs HDBSCAN demo export |

---

## Reproduce headline result

```powershell
python services/collect_questions.py --set new30
python services/label_questions.py --set new30
python services/compare_new30_finetune.py
```

Expected: **93.3% → 96.7% Recall@1** → `Documents/finetune/export/new30_base_vs_finetuned.json`

---

## Next experiments

1. LLM faithfulness judge (answer supported by retrieved chunks?)
2. Expand question bank beyond 40 train pairs
3. Hybrid BM25 + dense retrieval
4. Split oversized layout sections without losing headings
