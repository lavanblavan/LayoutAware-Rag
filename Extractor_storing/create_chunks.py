import re
import os
import sys
import nltk
from nltk.tokenize import sent_tokenize
import hdbscan
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

nltk.download('punkt')

LAYOUT_TAG_RE = re.compile(
    r"^\[(TITLE|TEXT|PARAGRAPH|LIST|TABLE|FIGURE)\]\s*(.*)$"
)
PAGE_BANNER_RE = re.compile(r"^={4,}\s*PAGE\s+\d+\s*={4,}$", re.I)
ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,.'()\-]{3,}$")
NUMBERED_SECTION = re.compile(r"^\s*\d+[A-Za-z]?\.\s+\S+")


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_sentences_per_chunk: int = 10,
        max_chars_per_chunk: int = 1400,
    ):
        self.model_name = model_name
        self._model = None
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk

    @property
    def model(self):
        """Load SentenceTransformer only when HDBSCAN fallback needs embeddings."""
        if self._model is None:
            from utils.torch_win import bootstrap_torch

            bootstrap_torch()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def preprocess_text(self, text):
        """
        Clean and split text into well-formed sentences.
        """
        # Remove extra spaces, weird chars
        text = re.sub(r'\s+', ' ', text).strip()
        # Sentence tokenization
        sentences = sent_tokenize(text)
        # Clean sentences
        cleaned_sentences = []
        seen = set()
        for s in sentences:
            s = s.strip()
            s = re.sub(r'\s+', ' ', s)
            if len(s) > 5 and s not in seen:
                cleaned_sentences.append(s)
                seen.add(s)
        return cleaned_sentences

    def cluster_sentences(self, sentences, min_cluster_size=2):
        """
        Cluster sentences into semantic chunks automatically.
        Uses HDBSCAN to discover the number of clusters.
        """
        if not sentences:
            return [], []

        # Compute embeddings
        embeddings = self.model.encode(sentences, convert_to_tensor=False, normalize_embeddings=True)

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)

        # Group sentences by cluster label
        clusters = {}
        for sentence, label in zip(sentences, labels):
            if label == -1:
                # Treat outliers as their own clusters
                label = f"outlier_{sentence[:15]}"
            clusters.setdefault(label, []).append(sentence)

        # Sort clusters by first occurrence in original text
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: min(sentences.index(s) for s in x[1])
        )

        # Create coarse chunks (full cluster joined)
        coarse_chunks = [' '.join(group) for _, group in sorted_clusters]
        return coarse_chunks, sorted_clusters

    def create_chunks(self, sorted_clusters):
        """
        Break clusters into sub-chunks of max_sentences_per_chunk sentences each.
        """
        all_chunks = []
        for _, group in sorted_clusters:
            for i in range(0, len(group), self.max_sentences_per_chunk):
                sub_chunk = group[i:i + self.max_sentences_per_chunk]
                chunk_text = ' '.join(sub_chunk)
                if len(chunk_text) > 30:  # skip too short noise
                    all_chunks.append(chunk_text)
        return all_chunks
    def create_chunks_group(self, sorted_clusters):
        """
        Break clusters into sub-chunks of max_sentences_per_chunk sentences each.
        """
        all_chunks = []
        all_chunk_groups = []
        for _, group in sorted_clusters:
            for i in range(0, len(group), self.max_sentences_per_chunk):
                sub_chunk = group[i:i + self.max_sentences_per_chunk]
                chunk_text = ' '.join(sub_chunk)
                  # skip too short noise
                all_chunks.append(chunk_text)
            all_chunk_groups.append(all_chunks)
            all_chunks = []

        return all_chunk_groups

    def _split_overlong(self, text):
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.max_chars_per_chunk:
            return [text]
        parts = []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 1 > self.max_chars_per_chunk:
                parts.append(buf.strip())
                buf = para
            else:
                buf = f"{buf} {para}".strip() if buf else para
        if buf.strip():
            parts.append(buf.strip())
        return [p for p in parts if len(p) > 30]

    def _pack_sections(self, sections, split_long=True):
        """sections: list of (heading, body) -> coarse, fine, groups."""
        coarse_chunks = []
        fine_chunks = []
        all_chunk_groups = []
        sentences = []
        for heading, body in sections:
            heading = (heading or "").strip()
            body = (body or "").strip()
            if heading and body:
                coarse = f"{heading}\n{body}"
            else:
                coarse = heading or body
            if not coarse or len(coarse) < 8:
                continue
            coarse_chunks.append(coarse)
            if split_long:
                pieces = self._split_overlong(coarse)
            else:
                pieces = [coarse]
            group = []
            for piece in pieces:
                fine_chunks.append(piece)
                group.append(piece)
                sentences.extend(sent_tokenize(piece) if piece else [])
            if group:
                all_chunk_groups.append(group)
        return sentences, coarse_chunks, fine_chunks, all_chunk_groups

    def _parse_layout_blocks(self, text):
        """Parse [TITLE] / [PARAGRAPH] blocks, including tag-on-its-own-line format."""
        blocks = []
        kind = None
        buf = []

        def flush():
            nonlocal kind, buf
            if not kind:
                buf = []
                return
            payload = "\n".join(buf).strip()
            if payload:
                blocks.append((kind, payload))
            kind = None
            buf = []

        for raw in text.splitlines():
            line = raw.strip()
            if not line or PAGE_BANNER_RE.match(line):
                continue
            match = LAYOUT_TAG_RE.match(line)
            if match:
                flush()
                kind = match.group(1)
                if kind == "PARAGRAPH":
                    kind = "TEXT"
                rest = (match.group(2) or "").strip()
                buf = [rest] if rest else []
                continue
            if kind:
                buf.append(raw.rstrip())
        flush()
        return blocks

    def run_layout(self, text):
        """Chunk LayoutParser / tagged output: keep title + following body together."""
        parsed = self._parse_layout_blocks(text)
        if not parsed:
            return None

        sections = []
        current_heading = ""
        current_body = []

        def flush():
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))

        for kind, payload in parsed:
            if kind == "FIGURE":
                continue
            if kind == "TABLE":
                flush()
                current_heading, current_body = "", []
                sections.append((f"[TABLE] {payload[:80]}", payload))
                continue
            first_line = payload.split("\n", 1)[0]
            starts_section = NUMBERED_SECTION.match(first_line)
            if kind == "TITLE":
                flush()
                current_heading = first_line[:200]
                extra = payload[len(first_line):].strip()
                current_body = [extra] if extra else []
                continue
            if kind == "LIST" or starts_section:
                flush()
                current_heading = first_line[:200]
                current_body = [payload]
                continue
            current_body.append(payload)

        flush()
        result = self._pack_sections(sections, split_long=False)
        if result[2]:
            log.info("Layout chunks: %s sections (1 chunk per heading/table)", len(result[2]))
        return result if result[2] else None

    def run_layout_strict(self, text):
        """Same as run_layout — one chunk per heading+body or table."""
        return self.run_layout(text)

    def run_legal_structure(self, text):
        """Chunk existing OCR dumps by headings and numbered clauses."""
        raw_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in raw_lines if line]
        if len(lines) < 8:
            return None

        sections = []
        heading = ""
        body = []

        def flush():
            if heading or body:
                sections.append((heading, " ".join(body).strip()))

        for line in lines:
            if ALLCAPS_HEADING.match(line) or NUMBERED_SECTION.match(line):
                flush()
                heading = line
                body = []
            else:
                body.append(line)
        flush()

        if len(sections) < 3:
            return None
        result = self._pack_sections(sections, split_long=False)
        if result[2]:
            log.info("Legal-structure chunks: %s fine / %s sections", len(result[2]), len(result[1]))
        return result if result[2] else None

    def run(self, text, strict_layout=True):
        layout_result = self.run_layout(text) if strict_layout else self.run_layout(text)
        if layout_result and layout_result[2]:
            sentences, coarse_chunks, fine_chunks, all_chunk_groups = layout_result
        else:
            legal_result = self.run_legal_structure(text)
            if legal_result and legal_result[2]:
                sentences, coarse_chunks, fine_chunks, all_chunk_groups = legal_result
            else:
                sentences = self.preprocess_text(text)
                coarse_chunks, sorted_clusters = self.cluster_sentences(sentences, min_cluster_size=2)
                log.info("Total sentences after preprocessing: %s", len(sentences))
                log.info("Number of coarse clusters: %s", len(coarse_chunks))
                fine_chunks = self.create_chunks(sorted_clusters)
                all_chunk_groups = self.create_chunks_group(sorted_clusters)

        for i, chunk in enumerate(fine_chunks):
            log.info("Chunk %s: %s%s", i + 1, chunk[:120], "..." if len(chunk) > 120 else "")

        return sentences, coarse_chunks, fine_chunks, all_chunk_groups


if __name__ == "__main__":
    configure_logging()
    file_path = r"C:\Users\Lavan\Desktop\Chatbot\Document_Summarizer\Document_Summarizer\Documents\Police\LK_Police_Ordinance_summary.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunker = SemanticChunker()
    sentences, coarse_chunks, fine_chunks, all_chunk_groups = chunker.run(content)

    # Example: get embeddings for fine chunks for RAG retrieval
    model = SentenceTransformer('all-MiniLM-L6-v2')
    fine_chunk_embeddings = model.encode(fine_chunks, normalize_embeddings=True)

    log.info("Created %s fine-grained chunks ready for indexing.", len(fine_chunks))
