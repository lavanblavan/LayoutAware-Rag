# PaperRAG — Setup & Run

Research-paper question-answering RAG over a PDF library. Retrieval uses **MiniLM** and **BGE** (base + fine-tuned). Answers are generated with **Groq** (cloud LLM).

---

## Requirements

- **Python 3.10+** (tested on 3.11)
- **Tesseract OCR** — [install for Windows](https://github.com/UB-Mannheim/tesseract/wiki)  
  Default path expected: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Poppler** (for `pdf2image`) — add to PATH or set `POPPLER_PATH` if needed
- **Groq API key** — [console.groq.com](https://console.groq.com/)
- GPU optional (CPU works; first model load takes ~1–2 min)

Layout extraction uses **PubLayNet** (Detectron2 via LayoutParser). On first run, weights are **downloaded automatically** from Hugging Face (~315 MB). If the neural layout model fails to load, the pipeline falls back to Tesseract block layout.

---

## Install

```powershell
cd path\to\New_Rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install Detectron2 + LayoutParser if not already present (Windows example):

```powershell
pip install torch torchvision
pip install layoutparser detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
```

Adjust the Detectron2 wheel URL for your CUDA/CPU setup if this command fails.

---

## Configure Groq

Create `settings/.env` from the example (never commit this file):

```powershell
copy settings\.env.example settings\.env
```

Then edit `settings/.env` and set your Groq API key:

---

## Add PDFs

Place research PDFs in:

```
Documents/documents/
```

The project ships with indexing scripts and eval exports under `services/`. PDFs, FAISS indexes, chunks, and fine-tuned weights are **generated locally** (not in the repo). Re-run indexing after adding PDFs.

---

## Run

Open **two terminals**:

**Terminal 1 — backend (port 9091)**

```powershell
cd path\to\New_Rag
python backend/api.py
```

**Terminal 2 — frontend (port 9010)**

```powershell
cd path\to\New_Rag\frontend
python server.py
```

Then open in a browser:

| Page | URL | Purpose |
|------|-----|---------|
| Main compare | http://localhost:9010/ | MiniLM vs BGE answers |
| Fine-tuned chat | http://localhost:9010/chat.html | Single chatbot (fine-tuned BGE only) |
| Side-by-side demo | http://localhost:9010/finetuned.html | Base BGE vs fine-tuned BGE |

On the main page, click **Rebuild index** if the library is not ready yet.

---

## Index the library (CLI alternative)

```powershell
python -c "from services.library_pipeline import process_library; process_library()"
```

Or trigger via API:

```powershell
curl -X POST http://localhost:9091/process
```

---

## Evaluation scripts

```powershell
# 10-question held-out test (base vs fine-tuned vs reranker)
python services/rate_retrieval.py --split test

# 30 unseen questions — base vs fine-tuned
python services/compare_new30_finetune.py

# Layout vs HDBSCAN chunk comparison (for demos)
python services/demo_chunk_compare.py
```

Reports are written to `Documents/finetune/export/`.

---

## Fine-tune BGE (optional)

Training data is under `Documents/finetune/export/bge_train_40.jsonl`.

```powershell
python services/train_bge.py --epochs 1 --output Documents/Models/bge-rag-finetuned-40-1ep
```

Point `FINETUNED_BGE_PATH` in `.env` at the saved model folder, then restart the backend.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Backend offline | Ensure `python backend/api.py` is running on port **9091** |
| Empty LLM answers | Check `GROQ_API_KEY` in `settings/.env`; restart backend |
| Fine-tuned chat loading forever | First start embeds the full corpus (~1–2 min); wait for status **Ready** |
| Tesseract not found | Install Tesseract and update path in `Summarizer/layout_extract.py` if needed |
| Library not ready | Click **Rebuild index** or run `POST /process` |

---

## Project layout (short)

```
backend/api.py          FastAPI server
frontend/               Web UI (PaperRAG)
services/               Pipeline, retrieval, fine-tune, eval
Documents/documents/    Input PDFs
Documents/finetune/     Chunks, questions, eval exports
Documents/Models/       Fine-tuned BGE weights
Database/               FAISS indexes (MiniLM + BGE)
settings/.env           Groq key (local only)
```

For how retrieval, chunking, and fine-tuning work, see **[README-RAG.md](README-RAG.md)**.
