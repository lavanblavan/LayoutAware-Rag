# Layout-Aware RAG (PaperRAG)

Layout-aware RAG for research papers — fine-tuned BGE retrieval + Groq answers over a PDF library.

**Repo:** [github.com/lavanblavan/Layout-Aware-Rag](https://github.com/lavanblavan/Layout-Aware-Rag)

Retrieval uses **MiniLM** and **BGE** (base + fine-tuned). Answers are generated with **Groq** (cloud LLM).

For pipeline details, failures, and validation design, see **[README-RAG.md](README-RAG.md)**.

---

## Requirements

- **Python 3.10+** (tested on 3.11)
- **Tesseract OCR** — [Windows install](https://github.com/UB-Mannheim/tesseract/wiki)  
  Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Poppler** (for `pdf2image`) — on PATH or set `POPPLER_PATH`
- **Groq API key** — [console.groq.com](https://console.groq.com/)
- GPU optional (CPU works; first model load ~1–2 min)

**PubLayNet** layout weights (~315 MB) download automatically from Hugging Face on first run. If layout detection fails, the pipeline falls back to Tesseract block layout.

---

## Install

```powershell
git clone https://github.com/lavanblavan/Layout-Aware-Rag.git
cd Layout-Aware-Rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Detectron2 + LayoutParser (Windows example — adjust for your CUDA/CPU setup):

```powershell
pip install torch torchvision
pip install layoutparser detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
```

---

## Configure Groq

```powershell
copy settings\.env.example settings\.env
```

Edit `settings/.env` and set `GROQ_API_KEY`. Never commit this file.

---

## Add PDFs

Place research PDFs in:

```
Documents/documents/
```

PDFs, FAISS indexes, chunks, and model weights are **generated locally** (not in the repo). Re-run indexing after adding papers.

---

## Run

**Terminal 1 — backend (port 9091)**

```powershell
python backend/api.py
```

**Terminal 2 — frontend (port 9010)**

```powershell
cd frontend
python server.py
```

| Page | URL | Purpose |
|------|-----|---------|
| Main compare | http://localhost:9010/ | MiniLM vs BGE |
| Fine-tuned chat | http://localhost:9010/chat.html | Single chatbot (fine-tuned BGE) |
| Side-by-side demo | http://localhost:9010/finetuned.html | Base vs fine-tuned BGE |

Click **Rebuild index** on the main page if the library is not ready.

**CLI indexing:**

```powershell
python -c "from services.library_pipeline import process_library; process_library()"
# or
curl -X POST http://localhost:9091/process
```

---

## Questions & evaluation

All questions live in **one file**: `services/questions_bank.json`

| Set | Questions | Purpose |
|-----|-----------|---------|
| `train40` | 40 | Fine-tuning (8 per paper) |
| `new30` | 30 | Unseen eval set |
| `custom` | yours | Add your own questions here |

### Add your own questions

Edit `services/questions_bank.json` under `sets.custom.questions`:

```json
{
  "document": "03_Self_RAG_Asai_2023",
  "question": "How does Self-RAG improve factuality over standard RAG?",
  "split": "eval"
}
```

Then collect retrieval and auto-label:

```powershell
python services/collect_questions.py --list
python services/collect_questions.py --set custom
python services/label_questions.py --set custom
```

### Built-in eval workflow

```powershell
# Fine-tune set
python services/collect_questions.py --set train40
python services/label_questions.py --set train40

# 30 unseen questions
python services/collect_questions.py --set new30
python services/label_questions.py --set new30
python services/compare_new30_finetune.py

# 10-Q held-out test (base vs fine-tuned vs reranker)
python services/rate_retrieval.py --split test

# Layout vs HDBSCAN chunk demo
python services/demo_chunk_compare.py
```
Reports → `Documents/finetune/export/`

Legacy wrappers (`collect_40_questions.py`, `collect_30_new_questions.py`, etc.) still work but call the unified scripts above.

---

## Fine-tune BGE

After labeling `train40`:

```powershell
python services/train_bge.py --epochs 1 --output Documents/Models/bge-rag-finetuned-40-1ep
```

Set `FINETUNED_BGE_PATH` in `settings/.env`, restart the backend.

Training data: `Documents/finetune/export/bge_train_40.jsonl`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Backend offline | Run `python backend/api.py` on port **9091** |
| Empty LLM answers | Check `GROQ_API_KEY` in `settings/.env`; restart backend |
| Fine-tuned chat loading | First start embeds corpus (~1–2 min); wait for **Ready** |
| Tesseract not found | Install Tesseract; update path in `Summarizer/layout_extract.py` |
| Library not ready | **Rebuild index** or `POST /process` |
| Unknown question set | Run `python services/collect_questions.py --list` |

---

## Project layout

```
backend/api.py              FastAPI server
frontend/                   PaperRAG web UI (3 pages)
services/
  questions_bank.json       ← edit questions here
  collect_questions.py      Run retrieval for any set
  label_questions.py        Auto-label gold chunks
  library_pipeline.py       PDF → layout → chunks → FAISS
  train_bge.py              Fine-tune BGE
Documents/documents/        Input PDFs (local)
Documents/finetune/export/  Eval reports (local)
Documents/Models/           Fine-tuned weights (local)
Database/                   FAISS indexes (local)
settings/.env               Groq key (local, not in repo)
```

---

## Headline result

```powershell
python services/compare_new30_finetune.py
```

**Base 93.3% → Fine-tuned 96.7% Recall@1** on 30 unseen questions.

See [README-RAG.md](README-RAG.md) for architecture, failures, and validation design.
