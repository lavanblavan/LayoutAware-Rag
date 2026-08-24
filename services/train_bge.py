"""
Fine-tune BAAI/bge-small-en-v1.5 on Documents/finetune/export/bge_train.jsonl

Uses MultipleNegativesRankingLoss with hard negatives from the export.

Usage:
  python services/train_bge.py
  python services/train_bge.py --epochs 4 --batch-size 8
  python services/train_bge.py --output Models/bge-rag-finetuned

After training, point retrieval at the saved model:
  set EMBEDDING_MODEL_BGE=Models/bge-rag-finetuned   (or update Settings.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger
from settings.Settings import Config

log = get_logger(__name__)

DEFAULT_TRAIN = os.path.join(Config.FINETUNE_EXPORT_PATH, "bge_train.jsonl")
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(Config.FINETUNE_EXPORT_PATH), "..", "Models", "bge-rag-finetuned"
)
DEFAULT_OUTPUT = os.path.normpath(DEFAULT_OUTPUT)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_examples(rows: list[dict], loss_name: str = "mnrl") -> list:
    """Convert query/pos/neg jsonl to sentence-transformers InputExample list."""
    from sentence_transformers import InputExample

    examples = []
    for row in rows:
        query = row["query"]
        pos_list = row["pos"] if isinstance(row["pos"], list) else [row["pos"]]
        negs = row.get("neg") or []
        if loss_name == "triplet":
            pos = pos_list[0]
            if negs:
                examples.append(InputExample(texts=[query, pos, negs[0]]))
            else:
                examples.append(InputExample(texts=[query, pos]))
        else:
            for pos in pos_list:
                examples.append(InputExample(texts=[query, pos]))
    return examples


def train(
    train_path: str = DEFAULT_TRAIN,
    output_path: str = DEFAULT_OUTPUT,
    base_model: str = "BAAI/bge-small-en-v1.5",
    epochs: int = 1,
    batch_size: int = 4,
    warmup_steps: int = 5,
    learning_rate: float = 2e-5,
    loss_name: str = "mnrl",
):
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Run: python services/build_bge_finetune.py"
        )

    rows = load_jsonl(train_path)
    if not rows:
        raise ValueError(f"No training rows in {train_path}")

    log.info("Loaded %s training pairs from %s", len(rows), train_path)
    log.info("Base model: %s", base_model)
    log.info("Output: %s", output_path)
    if len(rows) < 50:
        log.info(
            "Note: only %s pairs — use 1 epoch, low LR, mnrl loss to avoid overfitting.",
            len(rows),
        )

    examples = build_examples(rows, loss_name=loss_name)
    model = SentenceTransformer(base_model)

    loss_name = (loss_name or "mnrl").lower()
    if loss_name == "triplet":
        train_loss = losses.TripletLoss(
            model, distance_metric=losses.TripletDistanceMetric.COSINE
        )
        log.info("Loss: TripletLoss (query, positive, hard negative)")
    else:
        # In-batch negatives — usually more stable on tiny datasets
        train_loss = losses.MultipleNegativesRankingLoss(model)
        log.info("Loss: MultipleNegativesRankingLoss (in-batch negatives)")

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    os.makedirs(output_path, exist_ok=True)

    steps_per_epoch = max(1, len(examples) // batch_size)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=min(warmup_steps, steps_per_epoch),
        optimizer_params={"lr": learning_rate},
        output_path=output_path,
        show_progress_bar=True,
    )

    meta = {
        "base_model": base_model,
        "train_pairs": len(rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss": loss_name,
        "train_file": train_path,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    }
    with open(os.path.join(output_path, "finetune_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved fine-tuned model to: %s", output_path)
    log.info("Next: rebuild FAISS indexes or set bge model path in Settings.py")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE on RAG retrieval pairs")
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="Path to bge_train.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to save the model")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Base embedding model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="Lower LR reduces overfitting on small data")
    parser.add_argument(
        "--loss",
        default="mnrl",
        choices=["mnrl", "triplet"],
        help="mnrl = in-batch negatives (recommended for <50 pairs)",
    )
    args = parser.parse_args()

    train(
        train_path=args.train,
        output_path=args.output,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        learning_rate=args.lr,
        loss_name=args.loss,
    )


if __name__ == "__main__":
    configure_logging()
    main()
