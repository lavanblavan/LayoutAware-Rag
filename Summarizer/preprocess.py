from pdf2image import convert_from_path
import numpy as np
from PIL import Image

class DocumentPreprocessor:
    def __init__(self, dpi=150):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path):
        """
        Convert PDF to list of images (one per page)
        """
        images = convert_from_path(pdf_path, dpi=self.dpi)
        return images

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for better OCR results
        Steps:
        - Convert to grayscale
        - Apply binary thresholding
        - Remove noise with morphological operations
        """
        import cv2

        # Convert PIL image to OpenCV format
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply binary thresholding
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove noise with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


        return img

    def process_pdf(self, pdf_path):
        """
        Full pipeline: PDF -> Images -> Preprocessed Images
        Returns list of preprocessed images (as numpy arrays)
        """
        images = self.pdf_to_images(pdf_path)
        preprocessed_images = [self.preprocess_image(img) for img in images]
        return preprocessed_images