# Critique: what will actually go wrong

Written against mangatl, not for it. If you only read one document before
committing a weekend, read this one.

## The three quality gates prove less than they look like they prove

`ruff --select ALL`, `mypy --strict` and 100% branch coverage all pass. Here
is exactly what that does and does not buy you.

**What it buys:** no unreachable code, no untyped seams, no import-time
dependency on a GPU, every error branch exercised at least once, every
protocol implementation structurally verified, and a suite that runs in
seconds on a laptop with zero ML installed.

**What it does not buy:** any evidence that the translations are good. Not
one test asserts that a Japanese line produces a correct English line,
because doing so would require pinning model output, and pinned model output
tests rot the moment you change a model or a temperature. Coverage measures
whether code ran, not whether it was right.

Concretely: `test_pipeline_happy_path` passes with a translator that returns
`"EN:" + source`. That is the correct thing for a unit test to do, and it is
also why the benchmark harness exists as a separate, human-in-the-loop tool.
**The 100% number is a floor on plumbing correctness, not a claim about
output quality. Do not let it feel like the latter.**

Two more honest caveats on the coverage figure:

1. The adapters are covered by injecting fakes into `sys.modules`. That
   proves the adapter's own logic — clamping, padding, error paths — but it
   proves nothing about whether the real `ultralytics` API actually looks
   like the fake. A YOLO API change ships a green suite and a broken app.
   The CI job deliberately does not install the `[ml]` extra, so this gap is
   permanent by design; an integration job with real weights would be the
   fix, and it does not exist yet.

   **This is no longer hypothetical — it happened on the very first real
   page.** The fake returned plain `float` coordinates; real ultralytics
   returns 0-dim `torch.Tensor`, which does not implement `__round__`.
   `round(v)` in `_to_region` raised `TypeError` against the live library
   while all 198 tests stayed green. Fixed by coercing with `float()` first,
   and the fake now uses a `_TensorLike` double that defines `__float__` but
   *not* `__round__`, so the suite can see this class of defect. Every future
   fake should model the real type's *protocol gaps*, not just its values —
   a fake that is more capable than the real object is a fake that hides bugs.
2. One branch in `reading_order._split` was *deleted* rather than tested,
   because the gap invariant makes it unreachable. That is the right call,
   but it means the code now has no guard if that invariant is ever broken
   by a future edit. The invariant is documented in the docstring. Docstrings
   do not execute.

## Failure modes, roughly in order of how often you will hit them

**Typesetting will be the thing you hate.** This is the same wall
manga-image-translator's maintainer hit and openly documents as unsolved.
The binary-search auto-fit into a largest-inscribed-rectangle is a decent
algorithm and it will still produce text that overflows tall narrow balloons,
sits wrong in teardrop-shaped ones, and looks amateur next to hand
lettering. Expect to touch up in GIMP. The stroke outline is there
specifically so overflow stays legible rather than becoming unreadable.

**OCR errors propagate silently and are amplified by the LLM.** manga-ocr is
good — roughly 7–10% CER on Manga109-s — but a garbled source line does not
produce a garbled English line. It produces a *fluent, confident, wrong*
English line, because the LLM will happily interpolate. This is strictly
worse than a visibly broken output, because it survives a casual read-through.
The round-trip QE pass exists to catch exactly this and it is **off by
default** because it triples latency. Turn it on for anything you intend to
show another human.

**Reading order will be wrong on complex layouts.** The X/Y-cut handles grid
panels well. It will fail on overlapping panels, splash pages, diagonal panel
borders, and text deliberately placed across a gutter — which is to say, on
most action sequences. Wrong reading order silently corrupts the context that
the whole quality argument rests on. There is no detector for this; you have
to look.

**The `min_gap=12` default is a magic number I picked, not one I measured.**
Tightly packed pages will under-split; sparse pages will over-split. It is
exposed as a parameter for a reason.

**The deterministic fill only works on flat balloons.** Text over artwork,
screentone, or a gradient will get a visible flat patch. That is the honest
cost of not shipping inpainting weights by default. It is a one-line config
change to upgrade, and the licence check on whichever LaMa checkpoint you pull
is genuinely your problem, not a formality.

It is much less bad than it was. Two defects here were found by looking at a
real render rather than at the test suite:

1. The detector threw the segmentation **masks** away and kept only boxes, so
   erasure could only paint rectangles — destroying the balloon outline and any
   artwork sharing the bounding box. `TextRegion.polygon` existed and was
   documented as "used for inpainting and text fitting"; nothing ever populated
   it. The masks are now carried through and the fill follows the balloon
   outline. This is the fix that mattered; no generative model would have
   helped, because a perfect inpainter fed a bounding box still erases the
   wrong pixels.
2. The fill colour was sampled from a **single centre pixel**. When that pixel
   landed on a glyph stroke the whole balloon flooded with ink — a solid black
   bubble on the page. It now uses the modal colour of the region, since text
   is a minority of a balloon's area.

Note also that the weights branch has **never run**: it does
`torch.load(...)` then calls `model.inpaint(...)`, but `mayocream/aot-inpainting`
ships a raw `model.safetensors` state dict with no such method. Installing the
documented weights today would fail. Wiring a real AOT-GAN generator is
outstanding work, and given how well the polygon fill now performs on flat
balloons, it is worth doing only for screentone and gradient pages.

**Whole-page translation makes a page all-or-nothing.** If the model returns
the wrong array length, the entire page fails rather than one line. That is
deliberate — a silently misaligned page is worse than a failed one — but it
means one flaky response costs you a whole page of GPU time.

**Long chapters drift anyway.** Context is capped at 24 lines. A character
introduced on page 3 will be forgotten by page 40 unless they are in the
glossary. The glossary is the durable mechanism; context is not.

## Maintenance cost, stated plainly

- **Model churn is the real tax.** Every component here is on a repo that
  ships breaking changes: ultralytics majors, HuggingFace API shifts,
  llama.cpp renames flags roughly every quarter. The lazy-import design
  contains the blast radius to one adapter file each time, which is the main
  thing the architecture actually buys you.

  Both halves of that prediction have since come true, on the first real run:
  ultralytics returned Tensors where the fake returned floats, and the systemd
  unit's bare `--flash-attn` became `--flash-attn on|off|auto`, so the service
  consumed the *next* flag as its value and crash-looped. Neither is caught by
  any test, because neither the real library nor the real unit is exercised in
  CI. **Read the journal after an upgrade; a green suite says nothing here.**
- **Arch is rolling, and package names move too.** The installer pinned the
  package `nvidia`, which no longer exists — Arch ships `nvidia-open` /
  `nvidia-open-dkms`. With `set -e` and `--noconfirm` that aborted the entire
  install on any current box. It now probes for whichever module package is
  installed, else the first that still resolves. A driver update that outpaces
  your CUDA toolkit will still break llama.cpp until you rebuild.
- **`select = ["ALL"]` will fail on ruff upgrades.** New rules land enabled.
  This is a feature — it finds real things — but it is also unattended-CI
  breakage on someone else's release schedule. Pin ruff, upgrade
  deliberately.
- **AGPL-3.0 is load-bearing and viral.** The detection weights are GPL-3.0
  and LibreTranslate is AGPL-3.0, so the project cannot be relicensed to
  anything more permissive without dropping both. If you ever want to
  offer this as a hosted service, AGPL means you owe source to your users.

## Security surface

Small but not zero:

- `create_page` writes attacker-controlled bytes to disk before validating
  them as an image. Size is capped first, and the file is unlinked on
  failure, but a Pillow decoder CVE would be reachable. Pillow is the single
  most security-relevant dependency here. Keep it current.
- The app binds `127.0.0.1` by default and the systemd unit hardcodes that.
  **Do not change it to `0.0.0.0` without putting auth in front.** There is
  no authentication of any kind — it is a single-user tool and assumes a
  trusted network boundary of "this machine".
- `_image_size` and the pipeline decode untrusted images. Decompression-bomb
  protection is Pillow's default `MAX_IMAGE_PIXELS`, which is not disabled
  here, but a 30 MB legitimate-looking PNG can still consume real memory.
- The LLM endpoint is unauthenticated on localhost. Anything else on the box
  can use your GPU.
- Prompt injection is real and under-defended: OCR'd text goes into an LLM
  prompt. Text drawn inside a manga panel saying "ignore previous
  instructions" will be passed straight through. The consequence here is
  garbage output rather than data exfiltration — there are no tools attached
  to the model — but it is worth knowing before you wire anything else in.

## Abandonment risk

Honestly assessed, this is a personal tool. The realistic failure mode is not
a bug; it is that you use it for two chapters and then stop, because
BallonsTranslator already does 80% of it and does not require you to
maintain anything. `scripts/try-existing-first.sh` exists to let you reach
that conclusion in twenty minutes instead of after a weekend.

The things that genuinely justify building rather than adopting are narrow
and specific: a browser UI reachable from another machine, chapter-level
context carried automatically, an enforced-and-verified glossary, state that
survives closing the window, and handing pages to someone else for QC. If
none of those matter to you, do not build this.
