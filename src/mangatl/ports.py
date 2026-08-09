"""Protocol definitions for every swappable pipeline component.

The pipeline depends only on these protocols, never on a concrete adapter.
That is what lets the test suite run to 100% coverage without a single heavy
ML dependency installed, and what lets you swap a translation backend without
touching orchestration code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mangatl.domain.models import Box, GlossaryEntry, Page, TextRegion


@runtime_checkable
class Detector(Protocol):
    """Finds text regions on a page image."""

    def detect(self, image_path: Path, page: Page) -> list[TextRegion]:
        """Detect text regions.

        Args:
            image_path: Path to the page image on disk.
            page: Page metadata, used for dimensions and clamping.

        Returns:
            Detected regions in arbitrary order.
        """
        ...


@runtime_checkable
class Ocr(Protocol):
    """Reads source-language text out of a cropped region."""

    def read(self, image_path: Path, box: Box) -> str:
        """Recognise text.

        Args:
            image_path: Path to the page image on disk.
            box: Crop rectangle for the region.

        Returns:
            The recognised string, possibly empty.
        """
        ...


@runtime_checkable
class Translator(Protocol):
    """Turns a page's worth of source lines into target lines."""

    name: str

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str],
        glossary: Sequence[GlossaryEntry],
    ) -> list[str]:
        """Translate every line of one page together.

        Whole-page translation is not an optimisation: it is the single
        largest quality lever, because pronouns, honorifics and speaker
        gender in Japanese are resolved from surrounding lines.

        Args:
            lines: Source lines in reading order.
            context: Preceding lines from earlier pages, oldest first.
            glossary: Terms that must be rendered consistently.

        Returns:
            One target line per source line, same order and length.
        """
        ...


@runtime_checkable
class Inpainter(Protocol):
    """Erases source text from artwork."""

    def erase(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Remove text and write the cleaned image.

        Takes whole regions rather than bare boxes so an implementation can use
        ``region.polygon`` — the balloon outline from the segmentation model —
        and avoid erasing artwork that merely shares the bounding box.

        Args:
            image_path: Source page image.
            regions: Regions to erase.
            out_path: Destination path.

        Returns:
            The written path.
        """
        ...


@runtime_checkable
class Typesetter(Protocol):
    """Draws target text back onto the cleaned artwork."""

    def render(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Typeset every region and write the finished page.

        Args:
            image_path: Cleaned page image.
            regions: Regions carrying ``target_text``.
            out_path: Destination path.

        Returns:
            The written path.
        """
        ...
