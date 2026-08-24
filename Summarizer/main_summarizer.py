import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.torch_win import bootstrap_torch

bootstrap_torch()
from Summarizer.publaynet_model import load_publaynet_model  # noqa: E402

from settings.Settings import Config as settings_module
from Summarizer.Extraction import TextExtractor
from Summarizer.preprocess import DocumentPreprocessor
from Summarizer.Summary_creator import summary_create
from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

class MainSummarizer:
    def __init__(self, folder_path,extracted_text_path,summary_output_path):
        self.folder_path = folder_path
        self.extracted_text_path = extracted_text_path
        self.summary_output_path = summary_output_path
        os.makedirs(self.extracted_text_path, exist_ok=True)
        os.makedirs(self.summary_output_path, exist_ok=True)
        self.text_extractor = TextExtractor()
        self.preprocessor = DocumentPreprocessor()
        self.summarizer = summary_create()

    def get_pdf_files(self):
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
        return pdf_files
    
    def process_documents(self):
        load_publaynet_model()
        pdf_files = self.get_pdf_files()
        for pdf in pdf_files:
            log.info("Processing %s...", pdf)
            pdf_path = os.path.join(self.folder_path, pdf)
            images = self.preprocessor.pdf_to_images(pdf_path)
            total_text, layout_blocks = self.text_extractor.images_to_layout_text(images)
            if not (total_text or "").strip():
                preprocessed_images = self.preprocessor.process_pdf(pdf_path)
                page_texts = self.text_extractor.images_to_texts(preprocessed_images)
                tagged = []
                for i, page in enumerate(page_texts, start=1):
                    body = (page or "").strip()
                    if not body:
                        continue
                    tagged.append(f"======== PAGE {i} ========\n\n[PARAGRAPH]\n{body}")
                total_text = "\n\n".join(tagged)
                layout_blocks = []
            log.info("Extracted layout text from %s (%s regions).", pdf, len(layout_blocks))
            txt_filename = Path(self.extracted_text_path) / Path(pdf).with_suffix('.txt')
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(total_text)
            log.info("Saved extracted text to %s", txt_filename)
            summary = self.summarizer.find_minititles(total_text,document_title=pdf)        
            summary_filename = Path(self.summary_output_path) / f"{Path(pdf).stem}_summary.txt"

            self.summarizer.put_summary(summary_filename, summary)
            log.info("Saved summary to %s", summary_filename)

if __name__ == "__main__":
    configure_logging()
    folder_path = settings_module.PDF_FOLDER_PATH
    extracted_text_path = settings_module.EXTRACTED_TEXT_PATH
    summary_output_path = settings_module.SUMMARY_OUTPUT_PATH

    main_summarizer = MainSummarizer(folder_path, extracted_text_path, summary_output_path)
    main_summarizer.process_documents()