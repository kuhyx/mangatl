"""Pipeline orchestration.

Depends only on the protocols in :mod:`mangatl.ports`, so it can be exercised
end to end with fakes and swapped to different models without edits.

Order matters: detect, order, OCR, translate as one page, erase, typeset. The
ordering pass runs before OCR so that the lines handed to the translator are
already in reading order, which is what makes whole-page context work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from mangatl.domain.models import Page, Stage
from mangatl.domain.reading_order import order_regions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mangatl.domain.models import GlossaryEntry
    from mangatl.ports import Detector, Inpainter, Ocr, Translator, Typesetter


@dataclass(slots=True)
class PipelineResult:
    """Everything one page run produced.

    Attributes:
        page: The page with regions, OCR text and translations attached.
        cleaned_path: Where the text-erased artwork was written.
        rendered_path: Where the finished typeset page was written.
        context: Translated lines, for feeding the next page's prompt.
    """

    page: Page
    cleaned_path: Path | None = None
    rendered_path: Path | None = None
    context: list[str] = field(default_factory=list)


class Pipeline:
    """Runs one page from upload to finished render."""

    def __init__(
        self,
        *,
        detector: Detector,
        ocr: Ocr,
        translator: Translator,
        inpainter: Inpainter,
        typesetter: Typesetter,
        min_confidence: float = 0.25,
    ) -> None:
        """Wire the pipeline.

        Args:
            detector: Balloon/text detector.
            ocr: Source-language OCR.
            translator: Whole-page translator.
            inpainter: Text eraser.
            typesetter: Target-text renderer.
            min_confidence: Drop detections below this confidence.
        """
        self._detector = detector
        self._ocr = ocr
        self._translator = translator
        self._inpainter = inpainter
        self._typesetter = typesetter
        self._min_confidence = min_confidence

    def _detect(self, image_path: Path, page: Page) -> None:
        """Detect and reading-order the page's text regions."""
        found = [
            r
            for r in self._detector.detect(image_path, page)
            if r.confidence >= self._min_confidence
        ]
        page.regions = order_regions(found)
        page.stage = Stage.DETECTED

    def _recognise(self, image_path: Path, page: Page) -> None:
        """Run OCR over every detected region."""
        page.regions = [
            replace(region, source_text=self._ocr.read(image_path, region.box))
            for region in page.regions
        ]
        page.stage = Stage.RECOGNISED

    def _translate(
        self,
        page: Page,
        context: Sequence[str],
        glossary: Sequence[GlossaryEntry],
    ) -> list[str]:
        """Translate every non-empty region as one page-level request."""
        targets = page.translatable
        if not targets:
            page.stage = Stage.TRANSLATED
            return []
        lines = [r.source_text for r in targets]
        translated = self._translator.translate_page(lines, context=context, glossary=glossary)
        if len(translated) != len(lines):
            msg = f"translator returned {len(translated)} lines for {len(lines)} inputs"
            raise ValueError(msg)
        mapping = {region.id: text for region, text in zip(targets, translated, strict=True)}
        page.regions = [
            region.with_translation(mapping[region.id]) if region.id in mapping else region
            for region in page.regions
        ]
        page.stage = Stage.TRANSLATED
        return translated

    def run(
        self,
        image_path: Path,
        page: Page,
        *,
        out_dir: Path,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> PipelineResult:
        """Process one page end to end.

        Args:
            image_path: Uploaded page image.
            page: Page record to populate.
            out_dir: Directory for the cleaned and rendered outputs.
            context: Translated lines from preceding pages, oldest first.
            glossary: Terms that must be honoured.

        Returns:
            The populated page plus output paths. On failure the page is
            marked :attr:`Stage.FAILED` and the error text is attached
            rather than an exception escaping into the request handler.
        """
        try:
            self._detect(image_path, page)
            self._recognise(image_path, page)
            translated = self._translate(page, context, glossary)
            cleaned = out_dir / f"{page.id}.clean.png"
            rendered = out_dir / f"{page.id}.render.png"
            erasable = [r for r in page.regions if r.is_translatable]
            self._inpainter.erase(image_path, erasable, cleaned)
            self._typesetter.render(cleaned, page.regions, rendered)
            page.stage = Stage.RENDERED
        except (OSError, ValueError, RuntimeError) as exc:
            page.fail(f"{type(exc).__name__}: {exc}")
            return PipelineResult(page=page)
        return PipelineResult(
            page=page,
            cleaned_path=cleaned,
            rendered_path=rendered,
            context=translated,
        )
