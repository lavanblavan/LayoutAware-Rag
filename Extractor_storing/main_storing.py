import os
import sys
import faiss
from pathlib import Path

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.session_log import configure_logging, get_logger
from settings.Settings import Config as settings_module

log = get_logger(__name__)
from Extractor_storing.embedd_chunks import EmbedChunks


class MainStoring:
    def __init__(self):
        # Folder paths from settings
        self.extracted_text_folder = settings_module.EXTRACTED_TEXT_PATH
        self.summary_text_folder = settings_module.SUMMARY_OUTPUT_PATH

        self.extracted_faiss_path = settings_module.Extracted_Faiss
        self.summary_faiss_path = settings_module.Summarry_Faiss

        # Ensure output folders exist
        os.makedirs(self.extracted_faiss_path, exist_ok=True)
        os.makedirs(self.summary_faiss_path, exist_ok=True)

        # Initialize embedding engine
        self.embedder = EmbedChunks()

    def build_faiss_for_folder(self, input_folder, output_folder):
        """
        Build FAISS index for all .txt files from a given folder
        and save to output_folder.
        """
        txt_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith('.txt')
        ]

        if not txt_files:
            log.warning("No .txt files found in %s", input_folder)
            return

        log.info("Found %s text files in %s", len(txt_files), input_folder)

        for filename in txt_files:
            file_path = os.path.join(input_folder, filename)
            log.info("Processing %s ...", filename)

            # Read file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Build FAISS index
            index, fine_chunks, groups = self.embedder.build_index(content)

            # Output filenames
            index_out = os.path.join(output_folder, f"{Path(filename).stem}_faiss.index")
            meta_out = os.path.join(output_folder, f"{Path(filename).stem}_meta.npz")

            # Save index + metadata
            self.embedder.save_index(index_out, meta_out)

            log.info("Saved FAISS index & meta for %s", filename)

    def run(self):
        log.info("BUILDING EXTRACTED TEXT EMBEDDINGS")
        self.build_faiss_for_folder(
            input_folder=self.extracted_text_folder,
            output_folder=self.extracted_faiss_path
        )

        log.info("BUILDING SUMMARY EMBEDDINGS")
        self.build_faiss_for_folder(
            input_folder=self.summary_text_folder,
            output_folder=self.summary_faiss_path
        )

        log.info("All FAISS indexes created successfully!")


if __name__ == "__main__":
    configure_logging()
    MainStoring().run()
