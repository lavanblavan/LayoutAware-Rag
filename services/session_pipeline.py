"""
Process a single uploaded PDF into an isolated session store.
Old / library documents are never used for chat answers.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.torch_win import bootstrap_torch

bootstrap_torch()

from Summarizer.publaynet_model import load_publaynet_model  # noqa: E402

from settings.Settings import Config


SESSION_FOLDERS = [
    Config.SESSION_UPLOAD_PATH,
    Config.SESSION_EXTRACTED_PATH,
    Config.SESSION_SUMMARY_PATH,
    Config.SESSION_FAISS_EXTRACTED,
    Config.SESSION_FAISS_SUMMARY,
]


def _ensure_dirs():
    for path in SESSION_FOLDERS:
        os.makedirs(path, exist_ok=True)


def _clear_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def clear_session():
    """Wipe previous upload + indexes so only the new document is available."""
    _ensure_dirs()
    for path in SESSION_FOLDERS:
        _clear_folder(path)
    if os.path.exists(Config.SESSION_STATE_PATH):
        os.remove(Config.SESSION_STATE_PATH)


def save_state(data: dict):
    _ensure_dirs()
    with open(Config.SESSION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_state():
    if not os.path.exists(Config.SESSION_STATE_PATH):
        return None
    with open(Config.SESSION_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def store_upload(filename: str, content: bytes) -> str:
    """
    Replace any previous session document with this upload.
    Returns saved file path.
    """
    clear_session()
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    dest = os.path.join(Config.SESSION_UPLOAD_PATH, safe_name)
    with open(dest, "wb") as f:
        f.write(content)

    save_state({
        "filename": safe_name,
        "status": "uploaded",
        "message": "Document saved. Processing…",
        "pdf_path": dest,
    })
    return dest


def process_session_document() -> dict:
    """Extract → summarize → embed the active uploaded PDF only."""
    from PIL import Image
    import numpy as np
    from Summarizer.Extraction import TextExtractor
    from Summarizer.preprocess import DocumentPreprocessor
    from Summarizer.Summary_creator import summary_create
    from Extractor_storing.embedd_chunks import EmbedChunks

    state = load_state()
    if not state or not state.get("pdf_path"):
        raise RuntimeError("No document uploaded yet.")

    pdf_path = state["pdf_path"]
    if not os.path.exists(pdf_path):
        raise RuntimeError("Uploaded PDF is missing from session storage.")

    filename = state["filename"]
    stem = Path(filename).stem

    save_state({**state, "status": "extracting", "message": "Extracting text from PDF…"})

    load_publaynet_model()

    preprocessor = DocumentPreprocessor()
    extractor = TextExtractor()
    summarizer = summary_create()
    embedder = EmbedChunks()

    images = preprocessor.pdf_to_images(pdf_path)
    total_text, layout_blocks = extractor.images_to_layout_text(images)
    if not (total_text or "").strip():
        preprocessed = preprocessor.process_pdf(pdf_path)
        fallback_images = []
        for img in preprocessed:
            if isinstance(img, np.ndarray):
                fallback_images.append(Image.fromarray(img))
            else:
                fallback_images.append(img)
        pages = extractor.images_to_texts(fallback_images)
        tagged = []
        for i, page in enumerate(pages, start=1):
            body = (page or "").strip()
            if not body:
                continue
            tagged.append(f"======== PAGE {i} ========\n\n[PARAGRAPH]\n{body}")
        total_text = "\n\n".join(tagged)
        layout_blocks = []

    extracted_path = os.path.join(Config.SESSION_EXTRACTED_PATH, f"{stem}.txt")
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(total_text)

    layout_path = os.path.join(Config.SESSION_EXTRACTED_PATH, f"{stem}_layout.json")
    serializable = []
    for block in layout_blocks:
        item = dict(block)
        item["box"] = [float(x) for x in item.get("box", ())]
        serializable.append(item)
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    save_state({
        **state,
        "status": "summarizing",
        "message": "Building document summary…",
        "extracted_path": extracted_path,
    })

    print("🧠 Chunking document for summary and index...")
    _, _, fine_chunks, chunk_groups = embedder.chunker.run(total_text)
    summary = summarizer.find_minititles(
        total_text,
        document_title=filename,
        sections=fine_chunks,
    )
    summary_path = os.path.join(Config.SESSION_SUMMARY_PATH, f"{stem}_summary.txt")
    summarizer.put_summary(summary_path, summary)

    save_state({
        **state,
        "status": "embedding",
        "message": "Indexing document for chat…",
        "extracted_path": extracted_path,
        "summary_path": summary_path,
    })

    embedder.build_index_from_chunks(fine_chunks, chunk_groups)
    extract_index = os.path.join(Config.SESSION_FAISS_EXTRACTED, f"{stem}_faiss.index")
    extract_meta = os.path.join(Config.SESSION_FAISS_EXTRACTED, f"{stem}_meta.npz")
    embedder.save_index(extract_index, extract_meta)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
    embedder.build_index(summary_text)
    summary_index = os.path.join(Config.SESSION_FAISS_SUMMARY, f"{stem}_summary_faiss.index")
    summary_meta = os.path.join(Config.SESSION_FAISS_SUMMARY, f"{stem}_summary_meta.npz")
    embedder.save_index(summary_index, summary_meta)

    final_state = {
        "filename": filename,
        "status": "ready",
        "message": "Document ready. Ask a question.",
        "pdf_path": pdf_path,
        "extracted_path": extracted_path,
        "layout_path": layout_path,
        "summary_path": summary_path,
        "stem": stem,
        "extract_index": extract_index,
        "extract_meta": extract_meta,
        "summary_index": summary_index,
        "summary_meta": summary_meta,
    }
    save_state(final_state)
    return final_state


def retrieve_from_active_doc(query: str, top_k: int = 8):
    """Search only the active session document indexes."""
    from Retrieve_Documents.Retriever import Retriever

    state = load_state()
    if not state or state.get("status") != "ready":
        raise RuntimeError("No ready document in session. Upload and process a PDF first.")

    retriever = Retriever()
    retriever.load_index(state["extract_index"], state["extract_meta"])
    results = retriever.flat_search(query, top_k=top_k)

    return [
        {"document": state["filename"], "chunk": chunk, "score": float(score)}
        for chunk, score in results
    ]


def answer_question(question: str, top_k: int = 8) -> dict:
    """Retrieve session chunks and generate a chatbot-style answer with Ollama."""
    from services.llm_client import get_llm_client

    chunks = retrieve_from_active_doc(question, top_k=top_k)
    state = load_state()
    if not chunks:
        return {
            "answer": "I could not find relevant content in the uploaded document.",
            "sources": [],
            "document": state.get("filename") if state else None,
        }

    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1} | score={c['score']:.3f}]\n{c['chunk']}"
        for i, c in enumerate(chunks)
    )

    client = get_llm_client()
    prompt = (
        "You are a helpful document assistant. Answer the user's question using ONLY "
        "the excerpts from the uploaded document below. If the answer is not in the "
        "excerpts, say you cannot find it in this document. Be clear and concise.\n\n"
        f"Document name: {state.get('filename')}\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    last_error = None
    answer = None
    for model in dict.fromkeys(Config.LLM_MODEL_FALLBACKS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError(f"Ollama model {model} returned an empty answer.")
            break
        except Exception as e:
            last_error = e
            message = str(e).lower()
            if "model_not_found" not in message and "does not exist" not in message and "not found" not in message:
                raise
    if not answer:
        raise RuntimeError(f"Ollama chat failed: {last_error}") from last_error

    return {
        "answer": answer,
        "sources": chunks[:5],
        "document": state.get("filename"),
    }
