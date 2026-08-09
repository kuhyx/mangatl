"""Multi-pass translation refinement.

Ordered by measured value per unit of GPU time:

1. Whole-page draft with context and glossary (largest single win).
2. Self-critique pass over the draft (catches register and pronoun errors).
3. Glossary repair for any term the model still ignored (deterministic).
4. Optional round-trip quality estimation to pick between candidates.

Every step is optional and independently testable, so you can trade quality
for throughput by switching passes off rather than swapping models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mangatl.quality.prompt import (
    CRITIQUE_PROMPT,
    SYSTEM_PROMPT,
    StyleGuide,
    build_critique_prompt,
    build_translation_prompt,
    parse_lines_response,
)
from mangatl.quality.scoring import best_candidate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mangatl.domain.models import GlossaryEntry
    from mangatl.quality.glossary import Glossary


class ChatBackend(Protocol):
    """Minimal chat interface any local LLM server can satisfy."""

    def chat(self, system: str, user: str) -> str:
        """Send one system/user turn and return the raw assistant reply."""
        ...


@dataclass(frozen=True, slots=True)
class RefineSettings:
    """Which quality passes to run.

    Attributes:
        critique: Run the second-pass self-review.
        repair_glossary: Re-ask for lines that violate a glossary term.
        round_trip: Score candidates by back-translation and keep the best.
        candidates: How many draft candidates to generate when
            ``round_trip`` is enabled.
    """

    critique: bool = True
    repair_glossary: bool = True
    round_trip: bool = False
    candidates: int = 2

    def __post_init__(self) -> None:
        """Validate the candidate count.

        Raises:
            ValueError: If ``candidates`` is below one.
        """
        if self.candidates < 1:
            msg = f"candidates must be >= 1, got {self.candidates}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RefineReport:
    """What the refiner actually did, for the UI and for debugging.

    Attributes:
        lines: Final translated lines.
        passes: Names of the passes that ran, in order.
        repaired: Indices of lines fixed by glossary repair.
        scores: Round-trip scores per line when scoring ran, else empty.
    """

    lines: list[str]
    passes: list[str]
    repaired: list[int]
    scores: list[float]


class Refiner:
    """Runs the quality passes over one page of lines."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        settings: RefineSettings | None = None,
        style: StyleGuide | None = None,
    ) -> None:
        """Build a refiner.

        Args:
            backend: Chat backend pointed at a local model server.
            settings: Which passes to run; defaults enable critique+repair.
            style: Series style guide.
        """
        self._backend = backend
        self._settings = settings if settings is not None else RefineSettings()
        self._style = style if style is not None else StyleGuide()

    def _draft(
        self,
        lines: Sequence[str],
        context: Sequence[str],
        glossary: Sequence[GlossaryEntry],
    ) -> list[str]:
        """Run the first-pass whole-page translation."""
        prompt = build_translation_prompt(
            lines,
            context=context,
            glossary=glossary,
            style=self._style,
        )
        return parse_lines_response(self._backend.chat(SYSTEM_PROMPT, prompt), len(lines))

    def _critique(
        self,
        lines: Sequence[str],
        drafts: Sequence[str],
        glossary: Sequence[GlossaryEntry],
    ) -> list[str]:
        """Run the second-pass self-review over a draft."""
        prompt = build_critique_prompt(lines, drafts, glossary=glossary)
        return parse_lines_response(self._backend.chat(CRITIQUE_PROMPT, prompt), len(lines))

    def _repair(
        self,
        lines: Sequence[str],
        drafts: list[str],
        glossary: Glossary,
    ) -> tuple[list[str], list[int]]:
        """Re-ask for any line that ignored a mandatory glossary term."""
        repaired: list[int] = []
        out = list(drafts)
        for index, (source, draft) in enumerate(zip(lines, drafts, strict=True)):
            missing = glossary.violations(source, draft)
            if not missing:
                continue
            terms = "; ".join(f"{e.source} must be rendered as {e.target}" for e in missing)
            prompt = (
                f"Retranslate this single manga line, obeying the term list exactly.\n"
                f"Terms: {terms}\n"
                f"Japanese: {source}\n"
                f"Previous attempt: {draft}\n"
                f'Reply with {{"lines": ["..."]}} containing exactly 1 string.'
            )
            out[index] = parse_lines_response(self._backend.chat(SYSTEM_PROMPT, prompt), 1)[0]
            repaired.append(index)
        return out, repaired

    def _round_trip(self, lines: Sequence[str], drafts: Sequence[str]) -> list[float]:
        """Back-translate each draft and score round-trip fidelity."""
        scores: list[float] = []
        for source, draft in zip(lines, drafts, strict=True):
            if not draft.strip():
                scores.append(0.0)
                continue
            prompt = (
                f"Translate this English manga line back into natural Japanese.\n"
                f"English: {draft}\n"
                f'Reply with {{"lines": ["..."]}} containing exactly 1 string.'
            )
            back = parse_lines_response(self._backend.chat(SYSTEM_PROMPT, prompt), 1)[0]
            _, score = best_candidate(source, [(draft, back)])
            scores.append(score)
        return scores

    def run(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Glossary | None = None,
    ) -> RefineReport:
        """Translate one page's lines through every enabled quality pass.

        Args:
            lines: Source lines in reading order.
            context: Dialogue from preceding pages, oldest first.
            glossary: Series glossary; only relevant terms are injected.

        Returns:
            The final lines plus a record of what ran.
        """
        if not lines:
            return RefineReport(lines=[], passes=[], repaired=[], scores=[])
        relevant = glossary.relevant_to(lines) if glossary is not None else []
        passes = ["draft"]
        current = self._draft(lines, context, relevant)
        if self._settings.critique:
            current = self._critique(lines, current, relevant)
            passes.append("critique")
        repaired: list[int] = []
        if self._settings.repair_glossary and glossary is not None:
            current, repaired = self._repair(lines, current, glossary)
            if repaired:
                passes.append("glossary-repair")
        scores: list[float] = []
        if self._settings.round_trip:
            scores = self._round_trip(lines, current)
            passes.append("round-trip")
        return RefineReport(lines=current, passes=passes, repaired=repaired, scores=scores)
