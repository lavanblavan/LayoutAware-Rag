import os
import sys
import faiss
from pathlib import Path

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from settings.Settings import Config as settings_module
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
            print(f"⚠️ No .txt files found in {input_folder}")
            return

        print(f"📄 Found {len(txt_files)} text files in {input_folder}")

        for filename in txt_files:
            file_path = os.path.join(input_folder, filename)
            print(f"\n📘 Processing {filename} ...")

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

            print(f"✅ Saved FAISS index & meta for {filename}")

    def run(self):
        print("\n========================= 📌 BUILDING EXTRACTED TEXT EMBEDDINGS =========================")
        self.build_faiss_for_folder(
            input_folder=self.extracted_text_folder,
            output_folder=self.extracted_faiss_path
        )

        print("\n========================= 📌 BUILDING SUMMARY EMBEDDINGS =========================")
        self.build_faiss_for_folder(
            input_folder=self.summary_text_folder,
            output_folder=self.summary_faiss_path
        )

        print("\n🎉 All FAISS indexes created successfully!")


if __name__ == "__main__":
    MainStoring().run()
