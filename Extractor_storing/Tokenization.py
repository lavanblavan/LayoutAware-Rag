from sentence_transformers import SentenceTransformer
from typing import List, Union
from nltk.tokenize import sent_tokenize
import nltk
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

nltk.download('punkt')

class SentenceTokenizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load sentence-transformer model
        self.model = SentenceTransformer(model_name)

    def tokenize(self, text: str):
        """
        Tokenize text into sentences.
        """
        sentences = sent_tokenize(text)
        return sentences

    def embed(self, text: str):
        """
        Tokenize text into sentences and get embeddings.
        """
        sentences = self.tokenize(text)
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return sentences, embeddings
    
    def embed_list(self, texts: List[str]):
        """
        Tokenize a list of texts into sentences and get embeddings.
        """
        all_sentences = []
        for text in texts:
            sentences = self.tokenize(text)
            all_sentences.extend(sentences)
        
        embeddings = self.model.encode(all_sentences, convert_to_tensor=True)
        return all_sentences, embeddings


if __name__ == "__main__":
    configure_logging()
    document = """
Machine learning is a field of artificial intelligence.
It allows computers to learn from data without being explicitly programmed.
Sentence Transformers are great for creating vector embeddings.
"""

    tokenizer = SentenceTokenizer()
    sentences, embeddings = tokenizer.embed(document)

    log.info("Tokenized sentences:")
    for s in sentences:
        log.info("- %s", s)

    log.info("Embeddings shape: %s", embeddings.shape)
 