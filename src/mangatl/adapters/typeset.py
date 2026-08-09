"""Typesetting: fit English text into the space the Japanese vacated.

The algorithm is the one every working scanlation tool converges on:

1. Compute the usable rectangle inside the balloon polygon.
2. Binary-search the font size, greedily wrapping at each candidate size.
3. Keep the largest size whose wrapped block still fits.
4. Centre the block and draw with a contrasting stroke so it stays legible
   over artwork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mangatl.domain.geometry import text_area

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mangatl.domain.models import Box, TextRegion

MIN_FONT_PX = 8
MAX_FONT_PX = 72


@dataclass(frozen=True, slots=True)
class Layout:
    """A wrapped text block chosen for one region.

    Attributes:
        lines: Wrapped lines of text.
        size: Font size in pixels.
        line_height: Vertical advance per line in pixels.
    """

    lines: list[str]
    size: int
    line_height: int

    @property
    def height(self) -> int:
        """Total block height in pixels."""
        return self.line_height * len(self.lines)


def wrap_text(text: str, max_width: int, measure: Any) -> list[str]:
    """Greedily wrap ``text`` to ``max_width`` pixels.

    Args:
        text: Text to wrap.
        max_width: Maximum line width in pixels.
        measure: Callable returning the pixel width of a string.

    Returns:
        Wrapped lines. Words wider than ``max_width`` get their own line
        rather than being dropped or hyphenated mid-glyph.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class PillowTypesetter:
    """Renders target text with Pillow using an open-licensed comic font."""

    def __init__(self, font_path: Path, *, stroke: int = 2) -> None:
        """Build the typesetter.

        Args:
            font_path: TTF/OTF to letter with. Ship only OFL or Apache
                fonts: Comic Neue, Bangers and Noto Sans JP are all safe.
                Wild Words and the Blambot faces are not redistributable.
            stroke: Outline width in pixels, which keeps text readable when
                it overflows a balloon onto artwork.
        """
        self._font_path = font_path
        self._stroke = stroke

    def _font(self, size: int) -> Any:
        """Load the font at ``size`` pixels."""
        from PIL import ImageFont

        return ImageFont.truetype(str(self._font_path), size)

    def _fit(self, text: str, area: Box) -> Layout | None:
        """Binary-search the largest font size whose wrapped block fits.

        Args:
            text: Text to lay out.
            area: Usable rectangle.

        Returns:
            The chosen layout, or ``None`` when even the minimum size fails
            or the text is blank.
        """
        from PIL import Image, ImageDraw

        if not text.strip():
            return None
        scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        low, high = MIN_FONT_PX, MAX_FONT_PX
        best: Layout | None = None
        while low <= high:
            size = (low + high) // 2
            font = self._font(size)

            def measure(value: str, font: Any = font) -> float:
                return scratch.textlength(value, font=font)

            lines = wrap_text(text, area.width, measure)
            line_height = int(size * 1.18)
            layout = Layout(lines=lines, size=size, line_height=line_height)
            if lines and layout.height <= area.height:
                best = layout
                low = size + 1
            else:
                high = size - 1
        return best

    def render(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Draw every region's target text and write the finished page.

        Args:
            image_path: Cleaned page image.
            regions: Regions carrying ``target_text``.
            out_path: Destination path.

        Returns:
            The written path.
        """
        from PIL import Image, ImageDraw

        with Image.open(image_path) as img:
            canvas = img.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for region in regions:
                area = text_area(region.box, region.polygon)
                layout = self._fit(region.target_text, area)
                if layout is None:
                    continue
                font = self._font(layout.size)
                top = area.y1 + max(0, (area.height - layout.height) // 2)
                for index, line in enumerate(layout.lines):
                    draw.text(
                        (area.centre[0], top + index * layout.line_height),
                        line,
                        font=font,
                        fill=(0, 0, 0),
                        anchor="ma",
                        stroke_width=self._stroke,
                        stroke_fill=(255, 255, 255),
                    )
            canvas.save(out_path)
        return out_path
