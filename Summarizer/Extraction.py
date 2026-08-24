import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# PubLayNet / PyTorch must init before PIL in this process (Windows).
from Summarizer import publaynet_model  # noqa: F401
from Summarizer.layout_extract import LayoutExtractor


class TextExtractor:
    def __init__(self, lang="eng", use_layout=True):
        self.lang = lang
        self.use_layout = use_layout
        self.layout_extractor = LayoutExtractor(lang=lang) if use_layout else None

    def image_to_text(self, image) -> str:
        import pytesseract
        from PIL import Image

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not isinstance(image, Image.Image):
            from PIL import Image as PILImage
            import numpy as np

            image = PILImage.fromarray(np.asarray(image))
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text

    def images_to_texts(self, images):
        return [self.image_to_text(img) for img in images]

    def images_to_layout_text(self, images):
        if not self.layout_extractor:
            pages = self.images_to_texts(images)
            return "\n".join(pages), []
        return self.layout_extractor.extract_document(images)
