"""
Shared PubLayNet model loader (Hugging Face mirror).

Import this module before numpy/PIL/cv2 so PyTorch initializes correctly on Windows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.torch_win import bootstrap_torch

bootstrap_torch()

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "PyTorch failed to load. Try: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

import layoutparser as lp
from utils.session_log import get_logger

log = get_logger(__name__)

PUBLAYNET_LABELS = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
MODEL_DIR = Path(__file__).resolve().parent / "models" / "PubLayNet" / "faster_rcnn_R_50_FPN_3x"
HF_BASE = "https://huggingface.co/nlpconnect/PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main"

_cached_model = None


def download_file(name: str, dest: Path, min_size: int) -> Path:
    if dest.exists() and dest.stat().st_size >= min_size:
        head = dest.read_bytes()[:20]
        if not head.startswith(b"<!DOCTYPE"):
            return dest
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading PubLayNet %s…", name)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1024 * 1024):
            f.write(chunk)
    if tmp.stat().st_size < min_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {name} from Hugging Face.")
    with open(tmp, "rb") as f:
        if f.read(15).startswith(b"<!DOCTYPE"):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {name} (got HTML).")
    tmp.replace(dest)
    return dest


def load_publaynet_model(score_thresh: float = 0.55, reuse: bool = True):
    """Load (or reuse) PubLayNet Detectron2 layout model."""
    global _cached_model
    if reuse and _cached_model is not None:
        return _cached_model

    config = download_file("config.yml", MODEL_DIR / "config.yml", min_size=1000)
    weights = download_file("model_final.pth", MODEL_DIR / "model_final.pth", min_size=10_000_000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading PubLayNet on %s…", device)
    model = lp.Detectron2LayoutModel(
        config_path=str(config),
        model_path=str(weights),
        label_map=PUBLAYNET_LABELS,
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            score_thresh,
            "MODEL.DEVICE",
            device,
        ],
    )
    log.info("PubLayNet ready.")
    if reuse:
        _cached_model = model
    return model
