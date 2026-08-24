"""
Layout-aware PDF extraction via PubLayNet (titles, paragraphs, lists, tables).

Uses Summarizer/publaynet_model.py — same loader as run_publaynet.py.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List

import numpy as np
from PIL import Image
import pytesseract

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Summarizer.publaynet_model import PUBLAYNET_LABELS, load_publaynet_model
from utils.session_log import get_logger

log = get_logger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

KEEP_TYPES = {"Title", "Text", "List", "Table"}
SECTION_START = re.compile(
    r"^\s*(?:\d+[A-Za-z]?\.\s+|\(\d+\)\s+|[A-Z][A-Z\s]{4,}$)"
)


def _to_rgb_array(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        return img
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    raise TypeError(f"Unsupported image type: {type(image)}")


def _clip_box(x1, y1, x2, y2, h, w, pad=4):
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _ocr_crop(rgb: np.ndarray, box, lang="eng") -> str:
    clipped = _clip_box(*box, rgb.shape[0], rgb.shape[1])
    if clipped is None:
        return ""
    x1, y1, x2, y2 = clipped
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    text = pytesseract.image_to_string(Image.fromarray(crop), lang=lang)
    return re.sub(r"[ \t]+", " ", text).strip()


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _drop_overlaps(blocks: List[dict], iou_thresh=0.5) -> List[dict]:
    ranked = sorted(blocks, key=lambda b: b.get("score", 0.0), reverse=True)
    kept = []
    for block in ranked:
        if any(_iou(block["box"], k["box"]) > iou_thresh for k in kept):
            continue
        kept.append(block)
    kept.sort(key=lambda b: (b["page"], b["box"][1], b["box"][0]))
    return kept


DISPLAY_TAGS = {
    "Title": "TITLE",
    "Text": "PARAGRAPH",
    "List": "LIST",
    "Table": "TABLE",
    "Figure": "FIGURE",
}


def blocks_to_structured_text(blocks: List[dict]) -> str:
    lines = []
    current_page = None
    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        kind = block.get("type") or "Text"
        tag = DISPLAY_TAGS.get(kind, kind.upper())
        if tag == "TEXT":
            tag = "PARAGRAPH"
        page = block.get("page")
        if page != current_page:
            current_page = page
            if lines:
                lines.append("")
            lines.append(f"======== PAGE {page} ========")
            lines.append("")
        if tag != "TABLE":
            text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
        else:
            text = text.replace("\r\n", "\n").strip()
        lines.append(f"[{tag}]")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


class LayoutExtractor:
    """PubLayNet layout detection + Tesseract OCR per region."""

    def __init__(self, lang="eng", score_thresh=0.55):
        self.lang = lang
        self.score_thresh = score_thresh
        self._model = None
        self._model_failed = False
        if os.getenv("USE_NEURAL_LAYOUT", "1").lower() not in ("0", "false", "no"):
            self._load_model()

    def _load_model(self):
        if self._model is not None or self._model_failed:
            return self._model
        try:
            self._model = load_publaynet_model(score_thresh=self.score_thresh)
        except Exception as e:
            log.warning("PubLayNet unavailable (%s).", e)
            if os.getenv("ALLOW_TESSERACT_LAYOUT_FALLBACK", "0").lower() in ("1", "true", "yes"):
                log.info("Using Tesseract layout fallback.")
            self._model_failed = True
            self._model = None
        return self._model

    def detect_blocks_neural(self, rgb: np.ndarray, page: int) -> List[dict]:
        model = self._load_model()
        if model is None:
            return []

        layout = model.detect(rgb)
        blocks = []
        for item in layout:
            kind = getattr(item, "type", None) or "Text"
            if kind not in KEEP_TYPES:
                continue
            coord = item.block
            box = (coord.x_1, coord.y_1, coord.x_2, coord.y_2)
            score = float(getattr(item, "score", 1.0) or 1.0)
            text = _ocr_crop(rgb, box, self.lang)
            if not text:
                continue
            if kind == "Text" and SECTION_START.match(text.split("\n", 1)[0]):
                kind = "Title"
            blocks.append(
                {
                    "type": kind,
                    "text": text,
                    "box": box,
                    "score": score,
                    "page": page,
                }
            )
        return _drop_overlaps(blocks)

    def detect_blocks_tesseract(self, rgb: np.ndarray, page: int) -> List[dict]:
        """Slow fallback only when PubLayNet cannot load."""
        data = pytesseract.image_to_data(
            Image.fromarray(rgb),
            lang=self.lang,
            output_type=pytesseract.Output.DICT,
        )
        n = len(data["text"])
        heights = [
            data["height"][i]
            for i in range(n)
            if int(data["conf"][i]) > 30 and data["text"][i].strip()
        ]
        median_h = float(np.median(heights)) if heights else 12.0

        lines = {}
        for i in range(n):
            word = (data["text"][i] or "").strip()
            conf = int(data["conf"][i])
            if not word or conf < 25:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            rec = lines.setdefault(
                key,
                {"words": [], "lefts": [], "tops": [], "rights": [], "bottoms": [], "heights": []},
            )
            left, top = data["left"][i], data["top"][i]
            right, bottom = left + data["width"][i], top + data["height"][i]
            rec["words"].append(word)
            rec["lefts"].append(left)
            rec["tops"].append(top)
            rec["rights"].append(right)
            rec["bottoms"].append(bottom)
            rec["heights"].append(data["height"][i])

        paragraph_lines = {}
        for (block_n, par_n, line_n), rec in lines.items():
            text = " ".join(rec["words"]).strip()
            if not text:
                continue
            box = (
                min(rec["lefts"]),
                min(rec["tops"]),
                max(rec["rights"]),
                max(rec["bottoms"]),
            )
            avg_h = float(np.mean(rec["heights"]))
            xs = sorted(rec["lefts"])
            gaps = [xs[j + 1] - xs[j] for j in range(len(xs) - 1)]
            wide_gaps = sum(1 for g in gaps if g > 55)
            avg_word_len = float(np.mean([len(w) for w in rec["words"]]))
            kind = "Text"
            if wide_gaps >= 4 and len(rec["words"]) >= 8 and avg_word_len <= 5:
                kind = "Table"
            elif SECTION_START.match(text) or (
                text.isupper()
                and len(rec["words"]) <= 8
                and avg_h >= 1.3 * median_h
                and len(text) < 70
            ):
                kind = "Title"
            paragraph_lines.setdefault((block_n, par_n), []).append(
                (line_n, kind, text, box, avg_h)
            )

        blocks = []
        for (_block_n, _par_n), items in paragraph_lines.items():
            items.sort(key=lambda x: x[0])
            if len(items) >= 3 and sum(1 for item in items if item[1] == "Table") >= max(2, len(items) // 2):
                text = "\n".join(item[2] for item in items)
                box = (
                    min(item[3][0] for item in items),
                    min(item[3][1] for item in items),
                    max(item[3][2] for item in items),
                    max(item[3][3] for item in items),
                )
                blocks.append({"type": "Table", "text": text, "box": box, "score": 0.6, "page": page})
                continue

            current = []
            current_kind = "Text"
            current_box = None

            def flush():
                nonlocal current, current_box
                if not current:
                    return
                text = " ".join(current).strip()
                blocks.append(
                    {
                        "type": current_kind,
                        "text": text,
                        "box": current_box,
                        "score": 0.7 if current_kind == "Title" else 0.5,
                        "page": page,
                    }
                )
                current = []

            for _line_n, kind, text, box, _h in items:
                if kind == "Title":
                    flush()
                    current_kind = "Title"
                    current = [text]
                    current_box = box
                    flush()
                    current_kind = "Text"
                    current_box = None
                else:
                    if not current:
                        current_kind = "Text"
                        current_box = box
                    current.append(text)
                    if current_box is None:
                        current_box = box
                    else:
                        current_box = (
                            min(current_box[0], box[0]),
                            min(current_box[1], box[1]),
                            max(current_box[2], box[2]),
                            max(current_box[3], box[3]),
                        )
            flush()

        return _drop_overlaps(blocks, iou_thresh=0.7)

    def extract_page(self, image, page: int = 1) -> List[dict]:
        rgb = _to_rgb_array(image)
        blocks = self.detect_blocks_neural(rgb, page)
        if not blocks and self._model is None:
            allow = os.getenv("ALLOW_TESSERACT_LAYOUT_FALLBACK", "0").lower()
            if allow in ("1", "true", "yes"):
                blocks = self.detect_blocks_tesseract(rgb, page)
        return blocks

    def extract_document(self, images) -> tuple[str, List[dict]]:
        all_blocks: List[dict] = []
        for i, image in enumerate(images, start=1):
            log.info("Layout extract page %s/%s…", i, len(images))
            all_blocks.extend(self.extract_page(image, page=i))
        return blocks_to_structured_text(all_blocks), all_blocks

    def extract_text(self, images) -> str:
        text, _blocks = self.extract_document(images)
        return text
