"""Tests for the port protocols and the remaining reading-order edge case.

Protocol method bodies are real executable statements. Leaving them uncovered
would be exactly the kind of quiet hole the 100% gate exists to catch, so
these tests call the unimplemented bodies directly through ``super()`` and
assert the structural contract each protocol advertises.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mangatl.domain.models import Box, GlossaryEntry, Page, TextRegion
from mangatl.domain.reading_order import _split
from mangatl.ports import Detector, Inpainter, Ocr, Translator, Typesetter
from mangatl.quality.refine import ChatBackend

if TYPE_CHECKING:
    from collections.abc import Sequence


class NullDetector(Detector):
    """Concrete subclass that delegates to the protocol body."""

    def detect(self, image_path: Path, page: Page) -> list[TextRegion]:
        """Call the protocol body and return an empty result."""
        super().detect(image_path, page)
        return []


class NullOcr(Ocr):
    """Concrete subclass that delegates to the protocol body."""

    def read(self, image_path: Path, box: Box) -> str:
        """Call the protocol body and return an empty result."""
        super().read(image_path, box)
        return ""


class NullTranslator(Translator):
    """Concrete subclass that delegates to the protocol body."""

    name = "null"

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> list[str]:
        """Call the protocol body and echo blanks back."""
        super().translate_page(lines, context=context, glossary=glossary)
        return ["" for _ in lines]


class NullInpainter(Inpainter):
    """Concrete subclass that delegates to the protocol body."""

    def erase(self, image_path: Path, boxes: Sequence[Box], out_path: Path) -> Path:
        """Call the protocol body and return the destination unchanged."""
        super().erase(image_path, boxes, out_path)
        return out_path


class NullTypesetter(Typesetter):
    """Concrete subclass that delegates to the protocol body."""

    def render(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Call the protocol body and return the destination unchanged."""
        super().render(image_path, regions, out_path)
        return out_path


class NullBackend(ChatBackend):
    """Concrete subclass that delegates to the protocol body."""

    def chat(self, system: str, user: str) -> str:
        """Call the protocol body and return an empty reply."""
        super().chat(system, user)
        return ""


class TestProtocolBodies:
    """Every protocol body executes and every protocol is satisfiable."""

    def test_detector(self) -> None:
        """Detector."""
        detector = NullDetector()
        assert detector.detect(Path("x.png"), Page(id="p", width=10, height=10)) == []
        assert isinstance(detector, Detector)

    def test_ocr(self) -> None:
        """Ocr."""
        ocr = NullOcr()
        assert ocr.read(Path("x.png"), Box(0, 0, 4, 4)) == ""
        assert isinstance(ocr, Ocr)

    def test_translator(self) -> None:
        """Translator."""
        translator = NullTranslator()
        assert translator.translate_page(["a", "b"]) == ["", ""]
        assert isinstance(translator, Translator)

    def test_inpainter(self) -> None:
        """Inpainter."""
        inpainter = NullInpainter()
        out = Path("out.png")
        assert inpainter.erase(Path("x.png"), [], out) == out
        assert isinstance(inpainter, Inpainter)

    def test_typesetter(self) -> None:
        """Typesetter."""
        typesetter = NullTypesetter()
        out = Path("out.png")
        assert typesetter.render(Path("x.png"), [], out) == out
        assert isinstance(typesetter, Typesetter)

    def test_chat_backend(self) -> None:
        """Chat backend."""
        assert NullBackend().chat("sys", "user") == ""


class TestSplitGuard:
    """A cut that leaves one side empty must be rejected, not returned."""

    def test_cut_between_identical_centres_is_refused(self) -> None:
        """Cut between identical centres is refused."""
        regions = [
            TextRegion(id="a", box=Box(0, 0, 10, 200)),
            TextRegion(id="b", box=Box(0, 400, 10, 600)),
        ]
        # A horizontal cut is available, so this returns two groups.
        assert _split(regions, 12, horizontal=True) is not None
        # No vertical gap exists at all, so the x-axis cut is unavailable.
        assert _split(regions, 12, horizontal=False) is None

    def test_cut_always_yields_two_non_empty_groups(self) -> None:
        """The documented invariant: a real gap never empties a side."""
        regions = [
            TextRegion(id="wide", box=Box(0, 0, 400, 10)),
            TextRegion(id="far", box=Box(600, 0, 610, 10)),
        ]
        groups = _split(regions, 12, horizontal=False)
        assert groups is not None
        assert [r.id for r in groups[0]] == ["far"]
        assert [r.id for r in groups[1]] == ["wide"]
