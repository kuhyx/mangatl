"""Core domain types for a manga page and its translation state.

Everything here is pure data with no I/O and no ML dependency, so the whole
module is trivially testable and reusable by every adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Self

MIN_POLYGON_POINTS = 3


class RegionKind(StrEnum):
    """What a detected text region actually is on the page."""

    BUBBLE = "bubble"
    """Dialogue inside a speech balloon."""

    FREE = "free"
    """Narration or dialogue floating on the artwork with no balloon."""

    SFX = "sfx"
    """Sound effect / onomatopoeia, usually stylised lettering."""


class Stage(StrEnum):
    """Pipeline stage a page has reached."""

    UPLOADED = "uploaded"
    DETECTED = "detected"
    RECOGNISED = "recognised"
    TRANSLATED = "translated"
    RENDERED = "rendered"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge, inclusive.
        y1: Top edge, inclusive.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        """Reject degenerate or inverted boxes at construction time.

        Raises:
            ValueError: If the box has non-positive width or height.
        """
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            msg = f"degenerate box: {self.x1},{self.y1},{self.x2},{self.y2}"
            raise ValueError(msg)

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        """Geometric centre as ``(x, y)``."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def pad(self, amount: int, *, bounds: Box | None = None) -> Box:
        """Grow the box by ``amount`` on every side.

        Args:
            amount: Pixels to add per side. May be negative to shrink.
            bounds: Optional clamp region, usually the full page box.

        Returns:
            A new padded box, clamped to ``bounds`` when supplied.
        """
        x1 = self.x1 - amount
        y1 = self.y1 - amount
        x2 = self.x2 + amount
        y2 = self.y2 + amount
        if bounds is not None:
            x1 = max(x1, bounds.x1)
            y1 = max(y1, bounds.y1)
            x2 = min(x2, bounds.x2)
            y2 = min(y2, bounds.y2)
        return Box(x1, y1, x2, y2)

    def intersects(self, other: Box) -> bool:
        """Report whether two boxes overlap by at least one pixel."""
        return not (
            self.x2 <= other.x1 or other.x2 <= self.x1 or self.y2 <= other.y1 or other.y2 <= self.y1
        )

    @classmethod
    def hull(cls, points: list[tuple[int, int]]) -> Self:
        """Build the tightest box containing every supplied point.

        Args:
            points: At least one ``(x, y)`` pair.

        Returns:
            The bounding box, expanded by one pixel on the exclusive edges so
            that a single point still yields a valid non-degenerate box.

        Raises:
            ValueError: If ``points`` is empty.
        """
        if not points:
            msg = "cannot build a hull from zero points"
            raise ValueError(msg)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs) + 1, max(ys) + 1)


@dataclass(frozen=True, slots=True)
class TextRegion:
    """One detected piece of text and everything derived from it.

    Attributes:
        id: Stable identifier, unique within a page.
        box: Bounding box of the source text.
        kind: Whether the region is a balloon, free text, or an SFX.
        polygon: Optional tighter outline used for inpainting and text fitting.
        confidence: Detector confidence in ``[0, 1]``.
        source_text: OCR output in the source language.
        target_text: Final translated text.
        vertical: Whether the source text ran vertically.
        order: Reading-order index assigned by the reading-order pass.
        notes: Free-form translator/QC annotations.
    """

    id: str
    box: Box
    kind: RegionKind = RegionKind.BUBBLE
    polygon: tuple[tuple[int, int], ...] = ()
    confidence: float = 1.0
    source_text: str = ""
    target_text: str = ""
    vertical: bool = True
    order: int = -1
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate confidence and polygon shape.

        Raises:
            ValueError: If confidence is outside ``[0, 1]`` or the polygon has
                fewer than three points while being non-empty.
        """
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence out of range: {self.confidence}"
            raise ValueError(msg)
        if self.polygon and len(self.polygon) < MIN_POLYGON_POINTS:
            msg = f"polygon needs >= {MIN_POLYGON_POINTS} points, got {len(self.polygon)}"
            raise ValueError(msg)

    @property
    def is_translatable(self) -> bool:
        """Whether this region carries text worth sending to a translator."""
        return bool(self.source_text.strip())

    def with_translation(self, text: str) -> TextRegion:
        """Return a copy carrying ``text`` as the translated output."""
        return replace(self, target_text=text)

    def with_order(self, order: int) -> TextRegion:
        """Return a copy carrying ``order`` as its reading-order index."""
        return replace(self, order=order)


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """A term the translator must render consistently across a series.

    Attributes:
        source: Term as it appears in the source language.
        target: Required rendering in the target language.
        note: Optional guidance, e.g. "female, uses polite speech".
    """

    source: str
    target: str
    note: str = ""

    def __post_init__(self) -> None:
        """Reject blank terms.

        Raises:
            ValueError: If either side of the mapping is blank.
        """
        if not self.source.strip() or not self.target.strip():
            msg = "glossary entries need a non-blank source and target"
            raise ValueError(msg)


@dataclass(slots=True)
class Page:
    """A single manga page moving through the pipeline.

    Attributes:
        id: Stable page identifier.
        width: Image width in pixels.
        height: Image height in pixels.
        regions: Detected text regions, in reading order once ordered.
        stage: Furthest pipeline stage reached.
        error: Failure reason when ``stage`` is :attr:`Stage.FAILED`.
    """

    id: str
    width: int
    height: int
    regions: list[TextRegion] = field(default_factory=list)
    stage: Stage = Stage.UPLOADED
    error: str = ""

    def __post_init__(self) -> None:
        """Reject non-positive page dimensions.

        Raises:
            ValueError: If width or height is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            msg = f"page dimensions must be positive, got {self.width}x{self.height}"
            raise ValueError(msg)

    @property
    def box(self) -> Box:
        """The full page as a box, useful for clamping padded regions."""
        return Box(0, 0, self.width, self.height)

    @property
    def translatable(self) -> list[TextRegion]:
        """Regions that carry OCR text, in current list order."""
        return [r for r in self.regions if r.is_translatable]

    def fail(self, reason: str) -> None:
        """Mark the page failed with a human-readable ``reason``."""
        self.stage = Stage.FAILED
        self.error = reason
