"""Command-line entry point.

Two subcommands: ``serve`` starts the web UI, ``batch`` runs a directory of
pages headlessly and carries translated context forward from page to page,
which is what makes a whole chapter read consistently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from mangatl.config import Settings
from mangatl.domain.models import Page
from mangatl.quality.glossary import Glossary
from mangatl.storage.db import Database
from mangatl.web.assembly import build_pipeline

MAX_CARRIED_CONTEXT = 24
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="mangatl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    batch = sub.add_parser("batch", help="translate a directory of pages in order")
    batch.add_argument("directory", type=Path)
    batch.add_argument("--out", type=Path, default=None)
    return parser


def _page_size(path: Path) -> tuple[int, int]:
    """Read image dimensions."""
    with Image.open(path) as img:
        return (img.width, img.height)


def run_batch(directory: Path, settings: Settings, out_dir: Path | None = None) -> int:
    """Translate every image in ``directory``, carrying context forward.

    Args:
        directory: Folder of page images. Processed in sorted filename order,
            which is what chapter rips already use.
        settings: Runtime configuration.
        out_dir: Where renders go; defaults to the configured render dir.

    Returns:
        Process exit code: ``0`` when every page rendered, ``1`` otherwise.
    """
    settings.ensure_dirs()
    target = out_dir if out_dir is not None else settings.render_dir
    target.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(settings)
    glossary = Glossary.load(settings.data_dir / "glossary.json")
    db = Database(settings.db_path)
    context: list[str] = []
    failures = 0
    for image in sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        width, height = _page_size(image)
        page = Page(id=image.stem, width=width, height=height)
        result = pipeline.run(
            image,
            page,
            out_dir=target,
            context=context,
            glossary=glossary.entries,
        )
        db.save(result.page)
        if result.rendered_path is None:
            failures += 1
            sys.stderr.write(f"FAIL {image.name}: {result.page.error}\n")
        else:
            sys.stdout.write(f"OK   {image.name} -> {result.rendered_path.name}\n")
            context = [*context, *result.context][-MAX_CARRIED_CONTEXT:]
    db.close()
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    settings = Settings()
    if args.command == "batch":
        return run_batch(args.directory, settings, args.out)
    import uvicorn

    from mangatl.web.app import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0
