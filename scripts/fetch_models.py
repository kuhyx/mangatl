#!/usr/bin/env python3
"""Download the model weights mangatl needs, idempotently.

Lives in its own file rather than inside a shell heredoc so that ruff and mypy
actually see it. Each subcommand is a no-op when its target already exists, so
the installer can re-run freely.

Usage:
    fetch_models.py ocr
    fetch_models.py detector <models-dir>
    fetch_models.py gguf <models-dir> <repo> <filename>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _download(repo: str, filename: str) -> Path:
    """Fetch one file from the Hub and return its cached path.

    Args:
        repo: Hub repository id.
        filename: File to pull from that repository.

    Returns:
        Path to the cached file.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename))


def fetch_ocr() -> None:
    """Warm the manga-ocr cache so the first real page is not a long download."""
    from huggingface_hub import snapshot_download

    snapshot_download("kha-white/manga-ocr-base")


def fetch_detector(models_dir: Path) -> None:
    """Place the speech-bubble segmentation weights in ``models_dir``.

    Args:
        models_dir: Destination directory for ``bubble-seg.pt``.
    """
    dst = models_dir / "bubble-seg.pt"
    if dst.exists():
        return
    shutil.copy(_download("kitsumed/yolov8m_seg-speech-bubble", "model.pt"), dst)
    print(f"  -> {dst}")  # noqa: T201


def fetch_gguf(models_dir: Path, repo: str, name: str) -> None:
    """Place a GGUF checkpoint in ``models_dir``.

    Args:
        models_dir: Destination directory.
        repo: Hub repository id.
        name: GGUF filename within that repository.
    """
    dst = models_dir / name
    if dst.exists():
        return
    shutil.copy(_download(repo, name), dst)
    print(f"  -> {dst}")  # noqa: T201


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ocr", help="warm the manga-ocr cache")

    detector = sub.add_parser("detector", help="fetch bubble segmentation weights")
    detector.add_argument("models_dir", type=Path)

    gguf = sub.add_parser("gguf", help="fetch a GGUF checkpoint")
    gguf.add_argument("models_dir", type=Path)
    gguf.add_argument("repo")
    gguf.add_argument("name")

    args = parser.parse_args(argv)
    if args.command == "ocr":
        fetch_ocr()
    elif args.command == "detector":
        fetch_detector(args.models_dir)
    else:
        fetch_gguf(args.models_dir, args.repo, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
