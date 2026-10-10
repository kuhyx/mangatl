# mangatl

Fully local, FOSS Japanese → English manga translation. Drop a page in a
browser window, get a typeset English page back. Nothing leaves the machine.

Built for an Arch Linux box with an NVIDIA RTX 3090. Translation quality is
the primary design goal; throughput is secondary.

## Why not just use LibreTranslate

You can, and it is wired in as a fallback. But Argos/LibreTranslate is a
general-web NMT model, and generic ja→en models fall over on the register
manga actually uses: dropped subjects, dropped pronouns, heavy context
dependence, honorifics, and clipped spoken forms. Published numbers on
subtitle-style colloquial Japanese put generic models around 8 BLEU against
~18 for an in-domain model — and that gap is *understated*, because BLEU
itself misranks manga translation (the OpenMantra work found the system
humans preferred scored lower BLEU than the one they didn't).

What actually moves quality, in descending order of effect:

1. **Whole-page translation with reading order.** Japanese resolves pronouns
   and speaker gender from surrounding lines. Translating bubble-by-bubble
   throws that away before the model ever sees it.
2. **A local instruction-tuned LLM** rather than a dedicated NMT model.
3. **A series glossary injected into every prompt**, plus a deterministic
   post-check that re-asks for any line that ignored a term.
4. **A second-pass self-critique** over the draft.
5. **Round-trip QE** (back-translate, score with chrF) for final QC.

All five are implemented. The first four are on by default.

## Quickstart

```bash
./scripts/install-arch.sh          # unattended: drivers, models, fonts, services
./run.sh                           # start everything, wait, open the UI
```

`run.sh stop` shuts it down again. That matters: `llama-server` pins ~10 GB of
VRAM, so the services are installed but **not** enabled at login — you turn
them on for a translating session and get the card back afterwards.

```bash
./run.sh          # start both services and open http://127.0.0.1:8781
./run.sh stop     # stop both, release the VRAM
./run.sh status   # what is running, and how much VRAM it is holding
./run.sh logs     # follow the LLM log
```

Headless, a whole chapter with context carried page to page:

```bash
mangatl batch ~/scans/ch-01 --out ~/out/ch-01
```

### Already running ollama?

`llm_base_url` is just an OpenAI-compatible endpoint, so an existing ollama
install works with no code change and skips the llama.cpp build entirely:

```bash
export MANGATL_LLM_BASE_URL=http://127.0.0.1:11434/v1
export MANGATL_LLM_MODEL=qwen3:30b-a3b
mangatl serve
```

Verified end to end on an RTX 3090 with `qwen3:30b-a3b`: a 5-balloon page
detected at ~0.95 confidence, OCR'd, translated in right-to-left reading
order, erased and typeset in about 55 s including cold model load. A two-page
`batch` with the critique pass on its default takes ~81 s.

The page-to-page context claim is observable in that run: page two's
`駅に着いたよ` became "I got here early!" (the station comes from page one),
and the subjectless `さっき電話したけど出なかった` became "I called *her* …
but *she* didn't answer" — the referent survives the page boundary, which is
the whole argument for whole-page translation with carried context.

## Architecture

```
upload → detect (YOLO seg) → reading order (RTL X/Y-cut) → OCR (manga-ocr)
       → translate (whole page, local LLM, glossary + context + critique)
       → inpaint (AOT-GAN) → typeset (auto-fit into inscribed rect) → PNG
```

Every stage is a `Protocol` in `mangatl/ports.py`. The pipeline depends on
the protocols, never on a concrete adapter, so backends swap without touching
orchestration and the whole suite runs with zero ML dependencies installed.

| Stage | Default | Licence | Notes |
|---|---|---|---|
| Detect | `kitsumed/yolov8m_seg-speech-bubble` | GPL-3.0 | returns masks, not just boxes |
| OCR | `kha-white/manga-ocr-base` | Apache-2.0 | ~444 MB, vertical + furigana, multi-line in one pass |
| Translate | Qwen3-14B-Instruct via `llama-server` | Apache-2.0 | whole page + glossary + critique |
| Inpaint | AOT-GAN (`mayocream/aot-inpainting`) | MIT | real generative inpainting; falls back to a polygon fill with no weights |
| Typeset | Pillow + Comic Neue | OFL-1.1 | Wild Words and Blambot faces are **not** redistributable |

## Development

```bash
make check      # ruff format --check, ruff check, mypy --strict, pytest
```

Quality gates, all enforced in CI:

- `ruff` with `select = ["ALL"]` and five documented ignores, four of which
  are ruff's own documented rule conflicts.
- `mypy --strict` plus eight extra error codes strict does not imply.
- `pytest --cov-branch --cov-fail-under=100`. **There is no
  `pragma: no cover` escape hatch** — `exclude_lines` holds only
  `if TYPE_CHECKING:` and `if __name__ == "__main__":`, neither of which is
  executable code by definition.
- A separate `integration` job installs CPU torch and runs the AOT-GAN
  generator against the *real* library, because a fake that is more capable
  than the real object hides bugs — which is exactly how a Tensor with no
  `__round__` shipped past a green suite. The 100% gate is applied to the
  union of both runs, so neither half can quietly drop coverage:

  ```bash
  pip install -e '.[ml,dev]'
  pytest tests/integration --override-ini="testpaths=tests/integration"
  ```

## Licence

AGPL-3.0-or-later, matching LibreTranslate and the GPL-licensed detection
weights.

## Scope note

This is a translation *tool* that runs on your machine and hosts nothing.
Translating a work you do not hold rights to is infringement regardless of
what produced the translation. Deliberately absent: any raw-source
downloader, any bundled non-redistributable font, any hosted library.
