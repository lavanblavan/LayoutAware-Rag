"""
PubLayNet layout extraction — thin CLI over the shared pipeline.

  python Summarizer/run_publaynet.py Documents/documents/01-1990_E.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Summarizer.publaynet_model import load_publaynet_model
from Summarizer.layout_extract import LayoutExtractor
from settings.Settings import Config


def main():
    parser = argparse.ArgumentParser(description="PubLayNet layout + OCR")
    parser.add_argument("input", help="PDF or image path")
    parser.add_argument("--score", type=float, default=0.55, help="Detection threshold")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print("File not found:", src)
        sys.exit(1)

    load_publaynet_model(score_thresh=args.score)
    extractor = LayoutExtractor(lang="eng", score_thresh=args.score)

    if src.suffix.lower() == ".pdf":
        from Summarizer.preprocess import DocumentPreprocessor

        images = DocumentPreprocessor().pdf_to_images(str(src))
    else:
        from PIL import Image

        images = [Image.open(src).convert("RGB")]

    print(f"Processing {len(images)} page(s)…")
    text, blocks = extractor.extract_document(images)

    out_dir = Path(Config.EXTRACTED_TEXT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"Done — {len(blocks)} blocks → {out_path}")
    if blocks:
        print(f"Sample [{blocks[0]['type']}]: {blocks[0]['text'][:120]}…")


if __name__ == "__main__":
    main()
