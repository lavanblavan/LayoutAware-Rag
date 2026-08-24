from dotenv import load_dotenv
import os

_SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SETTINGS_DIR, ".."))
load_dotenv(os.path.join(_SETTINGS_DIR, ".env"))


class Config:
    # LLM: Groq (fast, cloud) or Ollama (local)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_MODEL_FALLBACKS = [
        m.strip()
        for m in os.getenv(
            "LLM_MODEL_FALLBACKS",
            "openai/gpt-oss-20b,openai/gpt-oss-120b,groq/compound-mini",
        ).split(",")
        if m.strip()
    ] or [LLM_MODEL]

    # Backward-compatible aliases
    GROQ_MODEL = LLM_MODEL
    GROQ_MODEL_FALLBACKS = LLM_MODEL_FALLBACKS

    PDF_FOLDER_PATH = os.path.join(_PROJECT_ROOT, "Documents", "documents")
    EXTRACTED_TEXT_PATH = os.path.join(_PROJECT_ROOT, "Documents", "Extracted_texts")
    CHUNKS_PATH = os.path.join(_PROJECT_ROOT, "Documents", "chunks")
    SUMMARY_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "Documents", "Summarry")
    LIBRARY_STATE_PATH = os.path.join(_PROJECT_ROOT, "Documents", "library_state.json")

    # Fine-tuning dataset (corpus, questions, retrieval logs, exports)
    FINETUNE_PATH = os.path.join(_PROJECT_ROOT, "Documents", "finetune")
    FINETUNE_CHUNKS_PATH = os.path.join(FINETUNE_PATH, "chunks")
    FINETUNE_EXPORT_PATH = os.path.join(FINETUNE_PATH, "export")
    CORPUS_JSONL = os.path.join(FINETUNE_PATH, "corpus.jsonl")
    QUESTIONS_JSONL = os.path.join(FINETUNE_PATH, "questions.jsonl")
    RETRIEVAL_RUNS_JSONL = os.path.join(FINETUNE_PATH, "retrieval_runs.jsonl")

    # Legacy single-model paths (MiniLM)
    Extracted_Faiss = os.path.join(_PROJECT_ROOT, "Database", "extracted")
    Summarry_Faiss = os.path.join(_PROJECT_ROOT, "Database", "summary")

    # Dual embedding models for retrieval comparison
    EMBEDDING_MODELS = {
        "minilm": "all-MiniLM-L6-v2",
        "bge": "BAAI/bge-small-en-v1.5",
    }

    FINETUNED_BGE_PATH = os.getenv(
        "FINETUNED_BGE_PATH",
        os.path.join(_PROJECT_ROOT, "Documents", "Models", "bge-rag-finetuned-40-1ep"),
    )

    @staticmethod
    def faiss_root(model_key: str, index_kind: str) -> str:
        """index_kind: extracted | summary"""
        return os.path.join(_PROJECT_ROOT, "Database", model_key, index_kind)

    # Active chat session — only the currently uploaded document is used
    SESSION_ROOT = os.path.join(_PROJECT_ROOT, "Documents", "session")
    SESSION_UPLOAD_PATH = os.path.join(SESSION_ROOT, "upload")
    SESSION_EXTRACTED_PATH = os.path.join(SESSION_ROOT, "extracted")
    SESSION_SUMMARY_PATH = os.path.join(SESSION_ROOT, "summary")
    SESSION_FAISS_EXTRACTED = os.path.join(_PROJECT_ROOT, "Database", "session", "extracted")
    SESSION_FAISS_SUMMARY = os.path.join(_PROJECT_ROOT, "Database", "session", "summary")
    SESSION_STATE_PATH = os.path.join(SESSION_ROOT, "state.json")
