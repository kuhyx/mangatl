"""Prompt construction for LLM-based manga translation.

This module is where translation quality is actually won or lost. The model
matters less than what you hand it: whole-page lines in reading order, prior
page context, the series glossary, an explicit register/honorific policy, and
a rigid output contract that can be parsed deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mangatl.domain.models import GlossaryEntry

MAX_CONTEXT_LINES = 24


class HonorificPolicy(StrEnum):
    """How Japanese honorifics should survive into English."""

    KEEP = "keep"
    """Leave -san/-kun/-chan/-sama attached, the fan-translation default."""

    DROP = "drop"
    """Remove honorifics entirely, the licensed-publisher default."""

    ADAPT = "adapt"
    """Render the social relationship in natural English instead."""


@dataclass(frozen=True, slots=True)
class StyleGuide:
    """Series-level choices the translator must apply consistently.

    Attributes:
        honorifics: How to handle -san and friends.
        register: Target voice, e.g. "casual contemporary teen dialogue".
        keep_sfx: Whether sound effects should be translated or left alone.
        notes: Any extra series-specific instruction.
    """

    honorifics: HonorificPolicy = HonorificPolicy.KEEP
    register: str = "natural contemporary English dialogue"
    keep_sfx: bool = True
    notes: str = ""


SYSTEM_PROMPT = """\
You are a professional Japanese-to-English manga translator.

You will receive the complete set of text lines from a single manga page, in
reading order (right to left, top to bottom), plus dialogue from preceding
pages for context.

Rules:
1. Translate every line. Never merge, split, drop, or reorder lines.
2. Use the surrounding lines to resolve Japanese pronoun drop, subject
   omission, speaker gender, and politeness level. This is why you get the
   whole page.
3. Output natural spoken English a real letterer would set, not literal
   glosses. Match the emotional register of the original.
4. Apply the supplied glossary exactly. Those renderings are mandatory.
5. Preserve line-internal emphasis and interjections. Keep exclamation and
   question marks where the Japanese carries them.
6. If a line is unreadable or empty, output an empty string for it.

Output contract: reply with a JSON object of the form
{"lines": ["...", "..."]} and nothing else. No prose, no markdown fence.
The array length must exactly equal the number of input lines.
"""

CRITIQUE_PROMPT = """\
You are a scanlation quality checker reviewing a draft translation.

For each line, check: mistranslation, wrong pronoun or speaker gender, wrong
politeness register, glossary violation, unnatural English, and lines that are
implausibly longer or shorter than their source.

Rewrite any line that needs it. Leave correct lines exactly as they are.

Output contract: reply with a JSON object of the form
{"lines": ["...", "..."]} and nothing else. The array length must exactly
equal the number of draft lines.
"""


def _format_glossary(glossary: Sequence[GlossaryEntry]) -> str:
    """Render glossary entries as a compact instruction block."""
    if not glossary:
        return "(none)"
    rows = []
    for entry in glossary:
        suffix = f"  # {entry.note}" if entry.note else ""
        rows.append(f"{entry.source} -> {entry.target}{suffix}")
    return "\n".join(rows)


def _format_style(style: StyleGuide) -> str:
    """Render the style guide as an instruction block."""
    honorifics = {
        HonorificPolicy.KEEP: "Keep Japanese honorifics attached to names.",
        HonorificPolicy.DROP: "Remove Japanese honorifics entirely.",
        HonorificPolicy.ADAPT: "Convey the social relationship in English instead of honorifics.",
    }[style.honorifics]
    sfx = (
        "Leave stylised sound effects untranslated."
        if style.keep_sfx
        else "Translate sound effects into English onomatopoeia."
    )
    parts = [honorifics, sfx, f"Target register: {style.register}."]
    if style.notes:
        parts.append(f"Series notes: {style.notes}")
    return "\n".join(parts)


def build_translation_prompt(
    lines: Sequence[str],
    *,
    context: Sequence[str] = (),
    glossary: Sequence[GlossaryEntry] = (),
    style: StyleGuide | None = None,
) -> str:
    """Build the user-side prompt for a whole-page translation request.

    Args:
        lines: Source lines in reading order.
        context: Dialogue from preceding pages, oldest first. Truncated to
            the most recent :data:`MAX_CONTEXT_LINES`.
        glossary: Terms that must be honoured.
        style: Series style guide; defaults are used when omitted.

    Returns:
        The prompt body.

    Raises:
        ValueError: If ``lines`` is empty.
    """
    if not lines:
        msg = "cannot build a translation prompt with zero lines"
        raise ValueError(msg)
    guide = style if style is not None else StyleGuide()
    recent = list(context)[-MAX_CONTEXT_LINES:]
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines))
    return (
        f"## Style guide\n{_format_style(guide)}\n\n"
        f"## Glossary (mandatory)\n{_format_glossary(glossary)}\n\n"
        f"## Context from preceding pages\n{chr(10).join(recent) if recent else '(none)'}\n\n"
        f"## Page lines to translate ({len(lines)} lines, reading order)\n{numbered}\n\n"
        f'Reply with {{"lines": [...]}} containing exactly {len(lines)} strings.'
    )


def build_critique_prompt(
    sources: Sequence[str],
    drafts: Sequence[str],
    *,
    glossary: Sequence[GlossaryEntry] = (),
) -> str:
    """Build the second-pass review prompt.

    Args:
        sources: Source lines in reading order.
        drafts: First-pass translations, same length and order.
        glossary: Terms that must be honoured.

    Returns:
        The prompt body.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if not sources or len(sources) != len(drafts):
        msg = f"sources/drafts length mismatch: {len(sources)} vs {len(drafts)}"
        raise ValueError(msg)
    pairs = "\n".join(
        f"{i}. JA: {src}\n   EN: {draft}"
        for i, (src, draft) in enumerate(zip(sources, drafts, strict=True))
    )
    return (
        f"## Glossary (mandatory)\n{_format_glossary(glossary)}\n\n"
        f"## Draft to review ({len(drafts)} lines)\n{pairs}\n\n"
        f'Reply with {{"lines": [...]}} containing exactly {len(drafts)} strings.'
    )


def parse_lines_response(raw: str, expected: int) -> list[str]:
    """Parse the model's JSON reply into exactly ``expected`` lines.

    Local models leak markdown fences and stray prose constantly, so this
    recovers the first JSON object in the payload rather than trusting the
    whole string to parse.

    Args:
        raw: Raw model output.
        expected: Number of lines the caller requires.

    Returns:
        Exactly ``expected`` strings, padded with empty strings if the model
        returned too few, truncated if it returned too many.

    Raises:
        ValueError: If no JSON object with a ``lines`` array can be found.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        msg = "model reply contained no JSON object"
        raise ValueError(msg)
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        msg = f"model reply was not valid JSON: {exc.msg}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        msg = "model reply lacked a 'lines' array"
        raise TypeError(msg)
    lines = [str(item) for item in payload["lines"]]
    if len(lines) < expected:
        lines.extend([""] * (expected - len(lines)))
    return lines[:expected]
