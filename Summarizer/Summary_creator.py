# from fileinput import filename
# import groq
# import os
# import sys
# import time
# import json
# import re
# import threading
# import asyncio
# import nest_asyncio
# import numpy as np
# from pathlib import Path

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# nest_asyncio.apply()

# import logging
# logging.basicConfig(level=logging.INFO)
# from settings.Settings import Config 

# logger = logging.getLogger(__name__)
# class summary_create:
#     def __init__(self):
#         self.api_key = Config.Groq_API_KEY
#         self.client = groq.Client(api_key=self.api_key)
#         self.model ="llama-3.3-70b-versatile"
#         self.array_of_summaries = []
        

    
#     def find_minititles(self,text,document_title="No Title Provided"):
#         """"
#         Find the sub titles which can be used for summarization in future"""
#         try:
#             prompt = (
#                 "You are a summary maker. Read the text below and break it into "
#                 "small, meaningful sections of summary for a page. try to get all the context into summary "
#                 "Each summary should be short, clear, and descriptive. "
#                 "Return only the numbered list of summary.\n\n"
#                 "Before putting summary put the heading of the documents if there is any.\n\n"
#                 f"Document Title: {document_title}\n\n"
#                 "if there is a same title no need to put it again and again just put subtitles\n\n"
#                 f"Text:\n{text}"
#             )

#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.5,
#                 max_tokens=300
#             )

#             reply = response.choices[0].message.content
            
#             print("Subtitles found successfully.",reply)
#             return reply

#         except Exception as e:
#             logger.error(f"Error in structured subtitle maker: {e}")
#             return "Sorry, I couldn't generate subtitles."
#     def put_summary(self,document_name,summary):
#         """""
#         put the summary of the document in json file
        
#         """
#         try:
#             with open(document_name, 'w') as f:
#                 f.write(summary)
#             print(f"Summary saved to {document_name}")
#         except Exception as e:
#             logger.error(f"Error saving array to file: {e}")
#             print("Failed to save array data.")

import os
import sys
import logging
import re
import nest_asyncio
from pathlib import Path

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from settings.Settings import Config
from services.llm_client import get_llm_client


class summary_create:
    CHUNK_WORDS = 1200
    MAX_SECTIONS_PER_BATCH = 8
    MAX_COMPLETION_TOKENS = 1200

    def __init__(self):
        self.client = get_llm_client()
        self.model = Config.LLM_MODEL
        self._model_candidates = list(dict.fromkeys(Config.LLM_MODEL_FALLBACKS))

    def chunk_text_by_tokens(self, text, max_token_words=None):
        """Approximate token chunking using word count."""
        if max_token_words is None:
            max_token_words = self.CHUNK_WORDS
        words = text.split()
        chunks = []
        current = []

        for word in words:
            current.append(word)
            if len(current) >= max_token_words:
                chunks.append(" ".join(current))
                current = []

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _is_missing_model(self, error):
        message = str(error).lower()
        return "model_not_found" in message or "does not exist" in message or "not found" in message

    def _clean_model_output(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if "### Section" in text:
            text = text[text.index("### Section") :]
        return text.strip()

    def _extract_message_text(self, message) -> str:
        text = (getattr(message, "content", None) or "").strip()
        if not text:
            text = (getattr(message, "reasoning", None) or "").strip()
        return self._clean_model_output(text)

    def _completion_tokens(self, model: str, section_count: int) -> int:
        base = max(self.MAX_COMPLETION_TOKENS, 220 * max(1, section_count))
        return min(4096, base)

    def _complete(self, prompt, max_tokens=None, model=None):
        last_error = None
        models = [model] if model else self._model_candidates
        models = [m for m in models if m]
        for candidate in models:
            tokens = max_tokens or self.MAX_COMPLETION_TOKENS
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=candidate,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Return only the requested summary text. "
                                    "No analysis, planning, or thinking tags."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=tokens,
                    )
                    text = self._extract_message_text(response.choices[0].message)
                    if not text:
                        raise RuntimeError(
                            f"Ollama model {candidate} returned empty summary content."
                        )
                    if candidate != self.model:
                        logger.info(f"Switched summarizer model to {candidate}")
                        self.model = candidate
                        self._model_candidates = [candidate] + [
                            m for m in self._model_candidates if m != candidate
                        ]
                    return text
                except Exception as e:
                    last_error = e
                    logger.error(f"Ollama chunk error with {candidate}: {e}")
                    if self._is_missing_model(e):
                        break
                    message = str(e).lower()
                    if "empty summary content" in message and attempt < 2:
                        tokens = min(4096, tokens + 1000)
                        logger.warning(
                            f"Retrying {candidate} with max_tokens={tokens}."
                        )
                        continue
                    break
        raise RuntimeError(f"Could not summarize chunk with Ollama: {last_error}") from last_error

    def summarize_chunk(self, chunk, document_title):
        words = chunk.split()
        if len(words) > self.CHUNK_WORDS * 1.3:
            parts = self.chunk_text_by_tokens(chunk, self.CHUNK_WORDS)
            summaries = [self.summarize_chunk(part, document_title) for part in parts]
            return "\n".join(summaries)

        prompt = (
            "You are an expert document summarizer.\n"
            "Summarize the section of text below into clear bullet points.\n"
            "Summary must preserve factual details.\n\n"
            f"Document Title: {document_title}\n\n"
            f"Text:\n{chunk}\n\n"
            "Return **ONLY** the summary.\n"
        )
        return self._complete(prompt)

    def _layout_sections(self, text):
        """Reuse the same layout/section chunks used for FAISS indexing."""
        from Extractor_storing.create_chunks import SemanticChunker

        chunker = SemanticChunker()
        result = chunker.run(text, strict_layout=True)
        if result and result[2]:
            return result[2]
        return None

    def _batch_sections(self, sections):
        """Pack numbered document chunks into LLM-safe batches."""
        batches = []
        current = []
        current_words = 0
        for i, section in enumerate(sections, start=1):
            body = (section or "").strip()
            if not body:
                continue
            labeled = f"[Section {i}]\n{body}"
            words = len(labeled.split())
            too_many_sections = len(current) >= self.MAX_SECTIONS_PER_BATCH
            too_many_words = current and current_words + words > self.CHUNK_WORDS
            if current and (too_many_sections or too_many_words):
                batches.append(current)
                current = []
                current_words = 0
            current.append((i, labeled))
            current_words += words
        if current:
            batches.append(current)
        return batches

    def summarize_sections(self, sections, document_title="No Title Provided"):
        """Summarize every layout/index chunk, batched for local model context limits."""
        batches = self._batch_sections(sections)
        logger.info(
            f"Summarizing {len(sections)} document chunks in {len(batches)} Ollama batches."
        )
        final_summary = []
        for batch_i, batch in enumerate(batches):
            start_n, end_n = batch[0][0], batch[-1][0]
            print(f"Summarizing document chunks {start_n}-{end_n} ({batch_i+1}/{len(batches)} Ollama calls)...")
            packed = "\n\n".join(item[1] for item in batch)
            prompt = (
                "You are an expert document summarizer.\n"
                "The document is already split into numbered sections.\n"
                "Summarize EVERY section separately. Do not skip, merge, or drop sections.\n"
                "Keep the section number and a short heading.\n"
                "Use bullet points and preserve factual details.\n"
                "If a section is only a heading or table fragment, still include it briefly.\n\n"
                f"Document Title: {document_title}\n\n"
                f"{packed}\n\n"
                "Return ONLY the section summaries in this format:\n"
                "### Section N: <heading>\n"
                "- bullet\n"
            )
            out_tokens = self._completion_tokens(self.model, len(batch))
            final_summary.append(self._complete(prompt, max_tokens=out_tokens))
        return "\n\n".join(final_summary)

    def find_minititles(self, text, document_title="No Title Provided", sections=None):
        """Summarize layout/index chunks. Word-split only if those chunks are missing."""
        try:
            if not sections:
                sections = self._layout_sections(text)
            if sections:
                return self.summarize_sections(sections, document_title)

            chunks = self.chunk_text_by_tokens(text)
            logger.info(f"No layout chunks found. Word-split into {len(chunks)} Ollama batches.")
            final_summary = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                final_summary.append(f"\n--- Summary for Chunk {i+1} ---\n")
                final_summary.append(self.summarize_chunk(chunk, document_title))
            return "\n".join(final_summary)

        except Exception as e:
            logger.error(f"Error in summarizer pipeline: {e}")
            raise

    # ---------------------------------------------------------------
    # SAVE SUMMARY TO FILE
    # ---------------------------------------------------------------
    def put_summary(self, document_name, summary):
        try:
            with open(document_name, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to {document_name}")

        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            print("Failed to save summary.")
      