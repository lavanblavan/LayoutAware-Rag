# PaperRAG — How It Works

This document explains the RAG pipeline: ingestion, chunking, retrieval, fine-tuning, generation, validation, and the main failures we hit (and how we fixed them).

---

## Problem

We built a **research-paper Q&A system** over five ML/RAG PDFs (Lewis et al. 2020, Gao survey, Self-RAG, CRAG, GraphRAG). Users ask natural-language questions; the system must retrieve the **right paper and section**, then generate an answer grounded in those passages.

Similar papers share vocabulary (“RAG”, “retrieval”, “factuality”), so a generic retriever often returns the **wrong document at rank 1**.

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

- PDF pages are rendered to images.
- **PubLayNet** (via LayoutParser) detects regions: Title, Text, List, Table.
- **Tesseract** OCR runs inside each region in reading order.
- Output is saved as tagged text in `Documents/Extracted_texts/*.txt` and region metadata in `*_layout.json`.

If the neural layout model is unavailable, the pipeline falls back to Tesseract paragraph blocks.

### 2. Chunking (fix #1 — layout-aware)

**Old approach (failure):** sentence embeddings + **HDBSCAN** clustering. Sentences that *sound similar* were grouped together, even from different sections. Chunks had no stable headings and often started mid-paragraph.

**Fix:** chunk **after** layout detection — one chunk per section heading or table. Example from Lewis 2020:

| Layout chunk | HDBSCAN chunk |
|--------------|---------------|
| `2.1 Models` — RAG-Token vs RAG-Sequence | Starts mid-sentence: *"sequence-to-sequence models We endow…"* |

Result: **406 section-level chunks** across five papers, stored in `Documents/finetune/chunks/`.

Demo export: run `python services/demo_chunk_compare.py` → `Documents/finetune/export/chunk_compare_layout_vs_hdbscan.md`.

### 3. Indexing & retrieval

Each chunk is embedded and stored in **FAISS** (`IndexFlatIP` with normalized vectors = cosine similarity).

| Model | Role |
|-------|------|
| `all-MiniLM-L6-v2` | Baseline retriever (main compare page) |
| `BAAI/bge-small-en-v1.5` | Stronger base retriever |
| Fine-tuned BGE (1 epoch) | Domain-tuned retriever for RAG papers |

BGE queries use the prefix:  
`Represent this sentence for searching relevant passages: <question>`

Retrieval flow:

1. Embed the question.
2. FAISS top-k search over all library chunks (~406).
3. Pass top 3–5 chunks to the LLM as context.

Fine-tuned retrieval uses the same FAISS structure but embeddings from the fine-tuned model (`services/finetuned_chat.py`).

### 4. Generation (Groq)

The LLM (**Groq**, `openai/gpt-oss-20b` with fallbacks) receives only retrieved excerpts. System prompt instructs: *answer only from the excerpts; say if unsupported*.

This connects retrieval → generation but does **not** include a separate automated faithfulness judge (planned as a next step).

---

## Fine-tuning (fix #2 — cross-paper confusion)

**Failure:** Base BGE confused papers with overlapping terminology.

| Question | Expected | Base top-1 |
|----------|----------|------------|
| How does Self-RAG improve factuality? | Self-RAG paper | **CRAG** (rank 2) |
| What role does non-parametric memory play? | Lewis 2020 | **Self-RAG** (rank 2) |

**Fix:**

1. **40 domain questions** (5 per paper), auto-labeled from BGE retrieval.
2. Train **BGE-small** with **MNRL** (Multiple Negatives Ranking Loss).
3. Use **1 epoch** (see failure #3 below).
4. Model: `Documents/Models/bge-rag-finetuned-40-1ep`.

**Results:**

| Eval set | Base Recall@1 | Fine-tuned Recall@1 |
|----------|---------------|---------------------|
| 10 held-out test questions | 80% | 80% (wins on hard cases) |
| 30 new unseen questions | 93.3% | **96.7%** (0 regressions) |

Hard win: `q_0021` (Self-RAG factuality) — rank **2 → 1**.

---

## Validation design

To avoid misleading metrics:

1. **Train / test split** — 30 train + 10 test from the original 40 questions (`bge_test_40.json`).
2. **Independent eval** — 30 **new** questions never used in training (`compare_new30_finetune.py`).
3. **Multi-gold labels** — several valid chunk IDs per question (not a single gold span).
4. **Metrics** — Recall@1, @3, @5 on chunk IDs.

Scripts: `services/rate_retrieval.py`, `services/compare_new30_finetune.py`.

---

## Three real failures & lessons

### Failure 1 — Dense HDBSCAN chunks mixed sections

- **Symptom:** Retrieval returned blobs with no section boundary; wrong context for precise questions.
- **Fix:** Layout-first chunking (PubLayNet → section chunks).
- **Lesson:** Structure of chunks matters as much as the embedding model.

### Failure 2 — Cross-paper retrieval confusion

- **Symptom:** Wrong paper at rank 1 when terminology overlaps.
- **Fix:** Domain fine-tuning on 40 Q→chunk pairs (1 epoch MNRL).
- **Lesson:** General-purpose embedders need domain adaptation on small specialized corpora.

### Failure 3 — Overfitting & leaky evaluation

**Overfitting:** Training **3 epochs** on ~40 pairs caused regressions (e.g. GraphRAG eval question dropped out of top results). **Fix:** **1 epoch**, keep checkpoint `bge-rag-finetuned-40-1ep`.

**Leaky eval:** Early 5-question compare used 4/5 **training** questions → both models showed 40% Recall@1, hiding real gains. **Fix:** proper held-out test + 30 independent questions.

### Also tried — reranker did not help

`BAAI/bge-reranker-base` on the 10-Q test: Recall@1 **80% → 70%**. Not deployed; fine-tuning was the better fix for this corpus.

---

## Risks & limitations

| Risk | Mitigation / status |
|------|---------------------|
| **Train/test leakage** | Separate 30-Q eval set; document splits in export JSON |
| **Overfitting** | 1 epoch; monitor per-question regressions |
| **OCR noise** | Layout regions help; some garbled text remains |
| **Small training set** | 40 pairs — gains are real but corpus-specific |
| **LLM hallucination** | Prompt grounding only; no automated faithfulness scorer yet |
| **Deployment** | Groq API dependency; not production-hardened |

---

## Web UI (three modes)

| Page | Retrieval | Use case |
|------|-----------|----------|
| `/` | MiniLM vs BGE | Compare baseline embedders |
| `/chat.html` | Fine-tuned BGE only | Customer-style single chatbot |
| `/finetuned.html` | Base vs fine-tuned | Demo / assessment side-by-side |

---

## Key files

| Path | Description |
|------|-------------|
| `services/library_pipeline.py` | PDF → layout → chunks → FAISS |
| `Extractor_storing/create_chunks.py` | Layout + HDBSCAN chunking |
| `Summarizer/layout_extract.py` | PubLayNet layout + OCR |
| `services/compare_chat.py` | MiniLM/BGE compare + Groq answers |
| `services/finetuned_chat.py` | Fine-tuned corpus store + compare/single chat |
| `services/train_bge.py` | BGE fine-tuning |
| `Documents/finetune/export/` | Eval reports, chunk compare, train JSONL |

---

## Next experiments

1. **LLM faithfulness judge** — score whether answers are supported by retrieved chunks.
2. **More training pairs** — expand beyond 40 questions with human or LLM silver labels.
3. **Hybrid retrieval** — BM25 + dense for entity-heavy queries (helpful on FEVER-style questions).
4. **Smarter chunk splitting** — split oversized sections (e.g. long Introduction blocks) without losing headings.

---

## Reproduce headline result

```powershell
python services/compare_new30_finetune.py
```

Expected: **Base 93.3% → Fine-tuned 96.7% Recall@1** on 30 unseen questions (`Documents/finetune/export/new30_base_vs_finetuned.json`).
