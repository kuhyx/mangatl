# Translation quality: the design argument

You asked for quality "first and foremost" and pointed at LibreTranslate.
Those two things are in tension, and this document is the case for how
mangatl resolves it.

## The short version

LibreTranslate is wired in, labelled a fallback, and not the default. Argos
(its engine) is a general-web NMT model, and it translates one line at a
time. Both of those are disqualifying for manga specifically — not because
the model is bad, but because manga dialogue is the worst possible case for
that shape of system.

## Why generic NMT collapses on manga specifically

Japanese manga dialogue is short, spoken, and radically context-dependent:

- **Subjects and pronouns are dropped.** A line is frequently two words with
  no stated actor. Who is speaking and about whom is recoverable only from
  the surrounding panels.
- **Politeness level carries meaning.** The difference between plain and
  polite form is often the entire characterisation of a line, and it has no
  direct English surface form.
- **Honorifics encode relationships** that English marks in word choice
  instead, if at all.
- **Register swings hard**, sometimes within one balloon.

A sentence-level NMT model sees none of this. It sees a two-word fragment
with no antecedent and guesses.

The published evidence lines up with the intuition. Generic ja→en models
score roughly 8 BLEU on colloquial subtitle-style Japanese against ~18 for an
in-domain model — a gap of more than 2x on exactly the register manga uses.
Argos itself is documented as materially slower and lower quality than peer
CPU translation engines.

**And the real gap is wider than that number suggests**, because BLEU
*understates* the problem here. In the OpenMantra manga-translation
experiments, the sentence-level system scored the *highest* BLEU (14.11) while
the context-aware system scored lower (12.65) — yet human raters preferred the
context-aware one (2.98 vs 2.76 out of 5). The authors concluded outright that
BLEU is not suitable for evaluating manga translation. Good manga translation
reorders and rewrites; BLEU punishes exactly that.

So: do not select a backend on BLEU, and do not select LibreTranslate at all
unless you are offline and desperate.

## The levers, ranked by measured effect

Implemented in `mangatl/quality/`. All five exist; the first four are on by
default.

### 1. Whole-page translation in reading order — the largest single win

Every source line on the page goes into one request, ordered right-to-left
top-to-bottom, with the previous pages' translated lines as context.

This is not a batching optimisation. It is the mechanism by which dropped
subjects get resolved at all. A model that can see the four surrounding
balloons can infer the speaker; a model handed one balloon cannot.

Everything upstream exists to serve this: the reading-order pass runs
*before* OCR precisely so that the lines handed to the model are already in
narrative order. Wrong reading order silently corrupts the entire premise.

Implementation: `quality/prompt.py::build_translation_prompt`, ordering in
`domain/reading_order.py`.

### 2. A local instruction-tuned LLM rather than dedicated NMT

This is where the field has moved. Every serious 2025–2026 tool
(BallonsTranslator, comic-translate, koharu) now defaults to an LLM fed page
context. comic-translate's README puts it bluntly: for distant language pairs
like Japanese↔English, a frontier LLM beats Google, Papago and DeepL "by
far", while the others still often devolve into gibberish.

Default: **Qwen3-14B-Instruct at Q5_K_M**, ~10 GB on your 3090, Apache-2.0,
served by llama.cpp on an OpenAI-compatible endpoint. That leaves comfortable
headroom for manga-ocr (~0.4 GB), the detector (~0.5 GB) and inpainting to
stay resident alongside it.

Deliberately not chosen:
- **Gemma 3** — strong, but a custom licence with use restrictions, not OSI.
- **SakuraLLM / Sakura-GalTransl** — genuinely ACGN-tuned, but targets
  Japanese→**Chinese**, not English, and is CC-BY-NC-SA.
- **NLLB-200 / SeamlessM4T** — CC-BY-NC-4.0, explicitly not for production.
- **Cloud frontier models** — best quality available, and they violate the
  "everything locally" requirement. The config can point at one if you ever
  change your mind; nothing else has to change.

### 3. Glossary injection, then deterministic verification

A local model will call the same character "Big Brother", "Onii-chan" and
"Nii-san" on three consecutive pages. Two mechanisms:

- **Injection**: only the terms that actually appear on this page go into
  the prompt (`Glossary.relevant_to`). Sending the whole glossary every time
  dilutes attention and wastes context.
- **Verification**: after translation, `Glossary.violations` checks each line
  for terms whose source appeared but whose required target did not, and
  re-asks for just those lines. This is a deterministic string check, not a
  hope — the model does not get to decide whether it complied.

This is the difference between mangatl and tools that merely "support" a
glossary.

### 4. Second-pass self-critique

The draft goes back to the model against a rubric: mistranslation, wrong
pronoun or speaker gender, wrong politeness register, glossary violation,
unnatural English, implausible length. Correct lines are returned unchanged.

Roughly doubles latency. Clearly worth it, so it is on by default
(`MANGATL_CRITIQUE_PASS=true`).

### 5. Round-trip quality estimation — off by default

Back-translate each English line to Japanese with the same model and score
the round trip with **chrF** (character n-gram F-score), combined with a
length-sanity term. Implemented from scratch in `quality/scoring.py` — pure
Python, no dependency.

chrF rather than BLEU because it is far more robust on short,
morphologically dense Japanese lines, and because at inference time there is
no reference translation to score against at all. This is reference-free QE,
which is the only kind available when you are actually translating.

Triples latency. Off by default; turn it on for final QC
(`MANGATL_ROUND_TRIP_QE=true`), and read the critique doc on why fluent-wrong
output is the failure mode it catches.

## What is deliberately not implemented

- **Speaker attribution.** Manga109Dialog provides 132,692 speaker-to-text
  pairs and would let you tell the model who is talking, which would help
  gendered pronouns considerably. Not built; it needs a second detector and
  a face/body model.
- **Panel-level reading order.** Currently ordering runs over text regions
  directly. Running it over detected *panels* first, then over balloons
  within each panel, is the approach the Manga109 panel-order work uses and
  reaches ~92% correct page order. This is the highest-value next
  improvement.
- **SFX translation.** Left untranslated by default (`keep_sfx=True`), which
  is what most readers prefer and what the honest limits of the pipeline
  support.

## How to actually decide, on your own pages

Do not trust this document. Run:

```bash
./scripts/benchmark.py --lines samples.txt --a local-llm --b libretranslate
```

It scores round-trip fidelity, length sanity, glossary compliance and
seconds-per-page for both, and writes per-line CSV. Then read twenty pages
yourself. The numbers narrow the field; they do not decide it — that is the
whole lesson of the BLEU result above.
