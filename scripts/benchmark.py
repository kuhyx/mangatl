#!/usr/bin/env python3
r"""A/B two translation backends on your own pages and score the difference.

BLEU is the wrong tool here. The OpenMantra work found the system human
raters preferred scored *lower* BLEU than the one they didn't, because good
manga translation legitimately reorders and rewrites. And at inference time
you have no reference translation anyway.

So this harness scores without a reference:

  * round-trip fidelity  -- back-translate the output, compare with chrF
  * length sanity        -- catch collapsed and runaway lines
  * glossary compliance  -- did the mandatory terms actually survive
  * latency              -- seconds per page, because you have one GPU

Then it prints a side-by-side and writes a CSV you can sort. The numbers
narrow the field; you still read twenty pages yourself before deciding.

    ./scripts/benchmark.py --lines samples.txt --a local-llm --b libretranslate
    ./scripts/benchmark.py --lines samples.txt --a local-llm --model-a qwen3-14b \\
                           --b local-llm --model-b gemma-3-12b-it

`samples.txt` is one Japanese line per page, blank line between pages.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from mangatl.adapters.translate_libre import LibreTranslateTranslator
from mangatl.adapters.translate_llm import HttpChatBackend, LlmTranslator
from mangatl.config import Settings
from mangatl.quality.glossary import Glossary
from mangatl.quality.prompt import parse_lines_response
from mangatl.quality.scoring import chrf, length_sanity


def load_pages(path: Path) -> list[list[str]]:
    """Read the sample file into pages of lines."""
    pages: list[list[str]] = []
    current: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            current.append(line)
        elif current:
            pages.append(current)
            current = []
    if current:
        pages.append(current)
    return pages


def make_translator(name: str, settings: Settings):  # noqa: ANN201
    """Build a translator by short name."""
    if name == "libretranslate":
        return LibreTranslateTranslator(settings)
    if name == "local-llm":
        return LlmTranslator(settings)
    msg = f"unknown backend: {name}"
    raise SystemExit(msg)


def back_translate(text: str, settings: Settings) -> str:
    """Round-trip an English line back to Japanese with the local LLM."""
    if not text.strip():
        return ""
    backend = HttpChatBackend(settings)
    prompt = (
        "Translate this English manga line back into natural Japanese.\n"
        f"English: {text}\n"
        'Reply with {"lines": ["..."]} containing exactly 1 string.'
    )
    try:
        return parse_lines_response(backend.chat("You are a translator.", prompt), 1)[0]
    except (RuntimeError, TypeError, ValueError):
        return ""


def evaluate(
    name: str,
    settings: Settings,
    pages: list[list[str]],
    glossary: Glossary,
) -> list[dict[str, object]]:
    """Translate every page and score each line."""
    translator = make_translator(name, settings)
    rows: list[dict[str, object]] = []
    context: list[str] = []
    for page_index, lines in enumerate(pages):
        start = time.perf_counter()
        try:
            out = translator.translate_page(lines, context=context, glossary=glossary.entries)
        except (RuntimeError, TypeError, ValueError) as exc:
            sys.stderr.write(f"{name}: page {page_index} failed: {exc}\n")
            continue
        elapsed = time.perf_counter() - start
        for source, target in zip(lines, out, strict=True):
            back = back_translate(target, settings)
            rows.append(
                {
                    "backend": name,
                    "page": page_index,
                    "source": source,
                    "target": target,
                    "round_trip": round(chrf(back, source), 4),
                    "length_sanity": round(length_sanity(source, target), 4),
                    "glossary_ok": not glossary.violations(source, target),
                    "seconds_per_page": round(elapsed, 2),
                },
            )
        context = [*context, *out][-24:]
    return rows


def summarise(rows: list[dict[str, object]], name: str) -> dict[str, float]:
    """Aggregate one backend's rows."""
    mine = [r for r in rows if r["backend"] == name]
    if not mine:
        return {"round_trip": 0.0, "length_sanity": 0.0, "glossary": 0.0, "seconds": 0.0}
    pages = {r["page"]: r["seconds_per_page"] for r in mine}
    return {
        "round_trip": sum(float(r["round_trip"]) for r in mine) / len(mine),
        "length_sanity": sum(float(r["length_sanity"]) for r in mine) / len(mine),
        "glossary": sum(1.0 for r in mine if r["glossary_ok"]) / len(mine),
        "seconds": sum(float(v) for v in pages.values()) / max(len(pages), 1),
    }


def main() -> int:
    """Run the A/B comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--a", default="local-llm")
    parser.add_argument("--b", default="libretranslate")
    parser.add_argument("--model-a", default=None)
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--glossary", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=Path("benchmark.csv"))
    args = parser.parse_args()

    pages = load_pages(args.lines)
    if not pages:
        sys.stderr.write("no pages found in the sample file\n")
        return 1
    glossary = Glossary.load(args.glossary) if args.glossary else Glossary()

    rows: list[dict[str, object]] = []
    for label, model in ((args.a, args.model_a), (args.b, args.model_b)):
        settings = Settings()
        if model:
            settings.llm_model = model
        sys.stdout.write(f"running {label}{f' ({model})' if model else ''}...\n")
        rows.extend(evaluate(label, settings, pages, glossary))

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'backend':<18}{'round-trip':>12}{'len-sanity':>12}{'glossary':>10}{'s/page':>9}"
    sys.stdout.write(f"\n{header}\n{'-' * len(header)}\n")
    for label in (args.a, args.b):
        s = summarise(rows, label)
        sys.stdout.write(
            f"{label:<18}{s['round_trip']:>12.3f}{s['length_sanity']:>12.3f}"
            f"{s['glossary']:>10.1%}{s['seconds']:>9.1f}\n",
        )
    sys.stdout.write(
        f"\nPer-line detail: {args.csv}\n"
        "These numbers narrow the field. Read twenty pages yourself before deciding.\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
