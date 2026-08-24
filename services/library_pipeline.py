"""
Offline library pipeline for downloaded research papers PDFs.

Flow per PDF:
  preprocess → layout extract → layout .txt
  → strict chunks (one chunk per heading/table)
  → Groq summary (batched sections with titles)
  → FAISS indexes for MiniLM and BGE (extracted + summary chunks)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.torch_win import bootstrap_torch

bootstrap_torch()

# Preload PubLayNet before any PIL/cv2 imports in downstream modules.
from Summarizer.publaynet_model import load_publaynet_model  # noqa: E402

from settings.Settings import Config
from utils.session_log import get_logger

log = get_logger(__name__)


def _ensure_dirs():
    for path in (
        Config.EXTRACTED_TEXT_PATH,
        Config.CHUNKS_PATH,
        Config.SUMMARY_OUTPUT_PATH,
        Config.FINETUNE_PATH,
        Config.FINETUNE_CHUNKS_PATH,
        Config.FINETUNE_EXPORT_PATH,
    ):
        os.makedirs(path, exist_ok=True)
    for model_key in Config.EMBEDDING_MODELS:
        os.makedirs(Config.faiss_root(model_key, "extracted"), exist_ok=True)
        os.makedirs(Config.faiss_root(model_key, "summary"), exist_ok=True)


def save_library_state(data: dict):
    _ensure_dirs()
    with open(Config.LIBRARY_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_library_state():
    if not os.path.exists(Config.LIBRARY_STATE_PATH):
        return None
    with open(Config.LIBRARY_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def list_library_pdfs():
    folder = Config.PDF_FOLDER_PATH
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    )


def extract_pdf_layout(pdf_path: str) -> tuple[str, list, str]:
    from Summarizer.preprocess import DocumentPreprocessor
    from Summarizer.Extraction import TextExtractor

    preprocessor = DocumentPreprocessor()
    extractor = TextExtractor()
    images = preprocessor.pdf_to_images(pdf_path)
    total_text, layout_blocks = extractor.images_to_layout_text(images)
    if not (total_text or "").strip():
        from PIL import Image
        import numpy as np

        preprocessed = preprocessor.process_pdf(pdf_path)
        pages = []
        for img in preprocessed:
            if isinstance(img, np.ndarray):
                pages.append(Image.fromarray(img))
            else:
                pages.append(img)
        page_texts = extractor.images_to_texts(pages)
        tagged = []
        for i, page in enumerate(page_texts, start=1):
            body = (page or "").strip()
            if not body:
                continue
            tagged.append(f"======== PAGE {i} ========\n\n[PARAGRAPH]\n{body}")
        total_text = "\n\n".join(tagged)
        layout_blocks = []
    stem = Path(pdf_path).stem
    return total_text, layout_blocks, stem


def chunk_layout_text(text: str) -> tuple[list[str], list]:
    from Extractor_storing.create_chunks import SemanticChunker

    chunker = SemanticChunker()
    result = chunker.run(text, strict_layout=True)
    if not result or not result[2]:
        result = chunker.run(text, strict_layout=False)
    _, _, fine_chunks, groups = result
    return fine_chunks, groups


def summary_section_chunks(summary_text: str) -> list[str]:
    """Split Groq summary output into section chunks for indexing."""
    parts = re.split(r"(?=^###\s+Section\s+\d+)", summary_text, flags=re.MULTILINE)
    chunks = [p.strip() for p in parts if p.strip()]
    if chunks:
        return chunks
    return [summary_text.strip()] if summary_text.strip() else []


def index_chunks(chunks: list[str], model_name: str, index_path: str, meta_path: str):
    from Extractor_storing.embedd_chunks import EmbedChunks

    embedder = EmbedChunks(model_name=model_name)
    groups = [[c] for c in chunks]
    embedder.build_index_from_chunks(chunks, groups)
    embedder.save_index(index_path, meta_path)


def process_one_pdf(pdf_name: str, skip_summary: bool = False) -> dict:
    pdf_path = os.path.join(Config.PDF_FOLDER_PATH, pdf_name)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    log.info("%s", f"\n{'='*60}\n📄 {pdf_name}\n{'='*60}")
    total_text, layout_blocks, stem = extract_pdf_layout(pdf_path)

    extracted_path = os.path.join(Config.EXTRACTED_TEXT_PATH, f"{stem}.txt")
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(total_text)

    layout_path = os.path.join(Config.EXTRACTED_TEXT_PATH, f"{stem}_layout.json")
    serializable = []
    for block in layout_blocks:
        item = dict(block)
        if "box" in item:
            item["box"] = [float(x) for x in item.get("box", ())]
        serializable.append(item)
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    doc_chunks, chunk_groups = chunk_layout_text(total_text)
    chunks_path = os.path.join(Config.CHUNKS_PATH, f"{stem}_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({"document": pdf_name, "stem": stem, "chunks": doc_chunks}, f, indent=2, ensure_ascii=False)

    from services.finetune_store import save_document_chunks, write_questions_template

    finetune_records = save_document_chunks(stem, pdf_name, doc_chunks)
    log.info("%s layout chunks saved (+ %s finetune ids)", len(doc_chunks), len(finetune_records))

    summary_path = os.path.join(Config.SUMMARY_OUTPUT_PATH, f"{stem}_summary.txt")
    if skip_summary and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
        log.info("Using existing summary")
    else:
        from Summarizer.Summary_creator import summary_create

        summarizer = summary_create()
        summary_text = summarizer.find_minititles(
            total_text,
            document_title=pdf_name,
            sections=doc_chunks,
        )
        summarizer.put_summary(summary_path, summary_text)

    summary_chunks = summary_section_chunks(summary_text)
    log.info("%s summary sections for indexing", len(summary_chunks))

    index_info = {}
    for model_key, model_name in Config.EMBEDDING_MODELS.items():
        extract_dir = Config.faiss_root(model_key, "extracted")
        summary_dir = Config.faiss_root(model_key, "summary")
        extract_index = os.path.join(extract_dir, f"{stem}_faiss.index")
        extract_meta = os.path.join(extract_dir, f"{stem}_meta.npz")
        summary_index = os.path.join(summary_dir, f"{stem}_summary_faiss.index")
        summary_meta = os.path.join(summary_dir, f"{stem}_summary_meta.npz")

        log.info("Indexing %s (%s)...", model_key, model_name)
        index_chunks(doc_chunks, model_name, extract_index, extract_meta)
        index_chunks(summary_chunks, model_name, summary_index, summary_meta)
        index_info[model_key] = {
            "extract_index": extract_index,
            "extract_meta": extract_meta,
            "summary_index": summary_index,
            "summary_meta": summary_meta,
        }

    return {
        "pdf": pdf_name,
        "stem": stem,
        "extracted_path": extracted_path,
        "layout_path": layout_path,
        "chunks_path": chunks_path,
        "summary_path": summary_path,
        "num_doc_chunks": len(doc_chunks),
        "num_summary_chunks": len(summary_chunks),
        "indexes": index_info,
    }


def process_library(skip_summary: bool = False) -> dict:
    pdfs = list_library_pdfs()
    if not pdfs:
        raise RuntimeError(
            f"No PDFs in {Config.PDF_FOLDER_PATH}. Add  PDFs there first."
        )

    _ensure_dirs()
    log.info("Loading PubLayNet layout model (once for all PDFs)…")
    load_publaynet_model()
    save_library_state({
        "status": "processing",
        "message": f"Processing {len(pdfs)} documents…",
        "documents": pdfs,
    })

    processed = []
    errors = []
    for pdf in pdfs:
        try:
            processed.append(process_one_pdf(pdf, skip_summary=skip_summary))
        except Exception as e:
            log.error("Failed %s: %s", pdf, e)
            errors.append({"pdf": pdf, "error": str(e)})

    final = {
        "status": "ready" if processed else "error",
        "message": f"Library ready — {len(processed)} document(s) indexed with MiniLM + BGE.",
        "documents": [p["pdf"] for p in processed],
        "processed": processed,
        "errors": errors,
        "models": Config.EMBEDDING_MODELS,
    }
    save_library_state(final)
    from services.finetune_store import write_questions_template

    write_questions_template()
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build library indexes from PDFs")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Reuse existing summary files (skip Groq calls)",
    )
    args = parser.parse_args()
    result = process_library(skip_summary=args.skip_summary)
    log.info("%s", json.dumps({"status": result["status"], "documents": result["documents"]}, indent=2))
