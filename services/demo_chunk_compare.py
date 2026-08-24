"""
Side-by-side demo: layout-aware chunks vs HDBSCAN dense chunks.

Exports a file you can open during the DigitalTurtles screen recording.

Usage:
  python services/demo_chunk_compare.py
  python services/demo_chunk_compare.py --document 01_Foundational_RAG_Lewis_2020
  python services/demo_chunk_compare.py --samples 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings.Settings import Config
from Extractor_storing.create_chunks import SemanticChunker

DEFAULT_DOC = "01_Foundational_RAG_Lewis_2020"
EXPORT_DIR = Config.FINETUNE_EXPORT_PATH
LAYOUT_TAG_RE = re.compile(
    r"^\[(TITLE|TEXT|PARAGRAPH|LIST|TABLE|FIGURE)\]\s*", re.MULTILINE
)


def _strip_layout_tags(text: str) -> str:
    text = LAYOUT_TAG_RE.sub("", text)
    text = re.sub(r"^={4,}\s*PAGE\s+\d+\s*={4,}$", "", text, flags=re.M)
    return text


def _heading(chunk: str) -> str:
    first = (chunk.split("\n", 1)[0] or "").strip()
    return first[:80] if first else "(no heading)"


def _preview(chunk: str, n: int = 220) -> str:
    one_line = " ".join(chunk.split())
    return one_line[:n] + ("…" if len(one_line) > n else "")


def _layout_chunks(chunker: SemanticChunker, text: str) -> list[str]:
    result = chunker.run_layout(text)
    return result[2] if result and result[2] else []


def _hdbscan_chunks(chunker: SemanticChunker, text: str, max_sentences: int | None) -> list[str]:
    plain = _strip_layout_tags(text)
    sentences = chunker.preprocess_text(plain)
    if max_sentences:
        sentences = sentences[:max_sentences]
    _, clusters = chunker.cluster_sentences(sentences, min_cluster_size=2)
    return chunker.create_chunks(clusters)


def _pick_showcase(layout: list[str], dense: list[str], n: int) -> list[dict]:
    """Pick chunks that best illustrate the difference (named sections vs mid-sentence blobs)."""
    rows = []

    # Layout: prefer numbered section headings
    layout_picks = []
    for i, ch in enumerate(layout):
        h = _heading(ch)
        if re.match(r"^\d+(\.\d+)*\s+\S", h) or h in {"Abstract", "Introduction"}:
            layout_picks.append((i, ch))
    if not layout_picks:
        layout_picks = list(enumerate(layout[:n]))

    for idx, ch in layout_picks[:n]:
        rows.append({
            "layout_index": idx,
            "layout_heading": _heading(ch),
            "layout_preview": _preview(ch),
            "layout_chars": len(ch),
        })

    for i, ch in enumerate(dense[:n]):
        if i >= len(rows):
            rows.append({})
        rows[i]["hdbscan_index"] = i
        rows[i]["hdbscan_starts_with"] = _preview(ch, 120)
        rows[i]["hdbscan_chars"] = len(ch)
        rows[i]["hdbscan_has_section_heading"] = bool(
            re.match(r"^\d+(\.\d+)*\s+[A-Z]", ch.strip())
        )

    return rows


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Layout vs HDBSCAN chunk comparison",
        "",
        f"**Document:** `{payload['document']}`  ",
        f"**Generated:** {payload['created_at']}  ",
        f"**Layout chunks:** {payload['counts']['layout']}  ",
        f"**HDBSCAN chunks:** {payload['counts']['hdbscan']}"
        + (
            f" (first {payload['hdbscan_sentence_limit']} sentences)"
            if payload.get("hdbscan_sentence_limit")
            else ""
        ),
        "",
        "Use this file side-by-side with `Documents/finetune/chunks/<document>.json` in your recording.",
        "",
        "---",
        "",
    ]

    for i, row in enumerate(payload["showcase"], start=1):
        lines.extend([
            f"## Example {i}",
            "",
            "### NEW — layout chunk (one section per heading)",
            f"- **Index:** {row.get('layout_index', '—')}",
            f"- **Heading:** `{row.get('layout_heading', '—')}`",
            f"- **Length:** {row.get('layout_chars', '—')} chars",
            f"- **Preview:** {row.get('layout_preview', '—')}",
            "",
            "### OLD — HDBSCAN dense chunk (similar sentences grouped)",
            f"- **Index:** {row.get('hdbscan_index', '—')}",
            f"- **Starts with section heading?** {row.get('hdbscan_has_section_heading', '—')}",
            f"- **Length:** {row.get('hdbscan_chars', '—')} chars",
            f"- **Preview:** {row.get('hdbscan_starts_with', '—')}",
            "",
            "---",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export layout vs HDBSCAN chunk comparison")
    parser.add_argument("--document", default=DEFAULT_DOC, help="Document stem (default: Lewis 2020)")
    parser.add_argument(
        "--samples", type=int, default=6,
        help="Number of side-by-side examples in the export (default: 6)",
    )
    parser.add_argument(
        "--hdbscan-sentences", type=int, default=120,
        help="Limit HDBSCAN to first N sentences for faster demo (default: 120)",
    )
    args = parser.parse_args()

    text_path = Path(Config.EXTRACTED_TEXT_PATH) / f"{args.document}.txt"
    if not text_path.exists():
        raise SystemExit(f"Missing extracted text: {text_path}\nRun library_pipeline first.")

    text = text_path.read_text(encoding="utf-8")
    chunker = SemanticChunker()

    print(f"Reading {text_path.name} …")
    layout = _layout_chunks(chunker, text)
    dense = _hdbscan_chunks(chunker, text, max_sentences=args.hdbscan_sentences)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document": args.document,
        "source_text": str(text_path),
        "new_chunks_file": str(
            Path(Config.FINETUNE_CHUNKS_PATH) / f"{args.document}.json"
        ),
        "hdbscan_sentence_limit": args.hdbscan_sentences,
        "counts": {"layout": len(layout), "hdbscan": len(dense)},
        "showcase": _pick_showcase(layout, dense, args.samples),
        "layout_samples": [
            {"index": i, "heading": _heading(c), "chars": len(c), "preview": _preview(c)}
            for i, c in enumerate(layout[: args.samples + 2])
        ],
        "hdbscan_samples": [
            {
                "index": i,
                "has_section_heading": bool(re.match(r"^\d+(\.\d+)*\s+[A-Z]", c.strip())),
                "chars": len(c),
                "preview": _preview(c),
            }
            for i, c in enumerate(dense[: args.samples + 2])
        ],
    }

    os.makedirs(EXPORT_DIR, exist_ok=True)
    json_path = Path(EXPORT_DIR) / "chunk_compare_layout_vs_hdbscan.json"
    md_path = Path(EXPORT_DIR) / "chunk_compare_layout_vs_hdbscan.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(md_path, payload)

    print()
    print("=" * 60)
    print("  LAYOUT (new)     vs     HDBSCAN (old)")
    print("=" * 60)
    print(f"  {payload['counts']['layout']} chunks          {payload['counts']['hdbscan']} chunks")
    print()
    for i, row in enumerate(payload["showcase"][:4], start=1):
        print(f"--- Example {i} ---")
        print(f"  NEW  [{row.get('layout_index')}] {row.get('layout_heading')}")
        print(f"       {row.get('layout_preview', '')[:100]}…")
        print(f"  OLD  [{row.get('hdbscan_index')}] heading={row.get('hdbscan_has_section_heading')}")
        print(f"       {row.get('hdbscan_starts_with', '')[:100]}…")
        print()

    print("Open these files for your recording:")
    print(f"  {md_path}")
    print(f"  {json_path}")
    print(f"  {payload['new_chunks_file']}")


if __name__ == "__main__":
    main()
