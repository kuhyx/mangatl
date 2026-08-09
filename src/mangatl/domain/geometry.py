"""Geometry used by the typesetter to fit English text into balloons.

The hard problem in scanlation typesetting is that a balloon is an irregular
blob, not a rectangle. The approach here is the pragmatic one every working
tool converges on: rasterise the balloon polygon into a boolean mask, find the
largest axis-aligned rectangle wholly inside it, and lay text out in that.
"""

from __future__ import annotations

from collections.abc import Sequence

from mangatl.domain.models import Box

Polygon = Sequence[tuple[int, int]]
MIN_POLYGON_POINTS = 3


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Test point containment with the even-odd ray-casting rule.

    Args:
        x: Point x coordinate.
        y: Point y coordinate.
        polygon: Closed polygon as ``(x, y)`` vertices, implicitly closed.

    Returns:
        ``True`` when the point lies inside the polygon.
    """
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        straddles = (yi > y) != (yj > y)
        if straddles:
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def polygon_mask(polygon: Polygon, box: Box) -> list[list[bool]]:
    """Rasterise ``polygon`` into a boolean mask covering ``box``.

    Args:
        polygon: Balloon outline in page coordinates.
        box: Region to rasterise, normally the polygon's bounding box.

    Returns:
        Row-major mask where ``mask[row][col]`` is inside-ness of the pixel at
        ``(box.x1 + col, box.y1 + row)``.
    """
    return [
        [
            point_in_polygon(box.x1 + col + 0.5, box.y1 + row + 0.5, polygon)
            for col in range(box.width)
        ]
        for row in range(box.height)
    ]


def _largest_rect_in_histogram(heights: list[int]) -> tuple[int, int, int]:
    """Largest rectangle in a histogram.

    Args:
        heights: Bar heights, left to right.

    Returns:
        ``(area, left_index, right_index_exclusive)`` of the best rectangle.
    """
    stack: list[int] = []
    best = (0, 0, 0)
    for i, h in enumerate([*heights, 0]):
        start = i
        while stack and heights[stack[-1]] >= h:
            top = stack.pop()
            area = heights[top] * (i - top)
            if area > best[0]:
                best = (area, top, i)
            start = top
        if h > 0:
            stack.append(min(i, start))
    return best


def largest_inscribed_rect(mask: list[list[bool]], origin: Box) -> Box | None:
    """Find the largest all-true axis-aligned rectangle inside ``mask``.

    Args:
        mask: Row-major boolean mask from :func:`polygon_mask`.
        origin: Box the mask was rasterised over, used to map back to page
            coordinates.

    Returns:
        The rectangle in page coordinates, or ``None`` when the mask is empty.
    """
    if not mask or not mask[0]:
        return None
    width = len(mask[0])
    heights = [0] * width
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for row_index, row in enumerate(mask):
        for col in range(width):
            heights[col] = heights[col] + 1 if row[col] else 0
        area, left, right = _largest_rect_in_histogram(heights)
        if area > best_area:
            height = area // (right - left)
            best_area = area
            best = (left, row_index + 1 - height, right, row_index + 1)
    if best is None:
        return None
    left, top, right, bottom = best
    return Box(origin.x1 + left, origin.y1 + top, origin.x1 + right, origin.y1 + bottom)


def text_area(region_box: Box, polygon: Polygon, *, inset: int = 4) -> Box:
    """Compute the rectangle text should be laid out in for one region.

    Args:
        region_box: Bounding box of the detected region.
        polygon: Balloon outline; when empty the box itself is used.
        inset: Pixels of breathing room to keep away from the balloon edge.

    Returns:
        The usable text rectangle, never smaller than a 1x1 box.
    """
    if len(polygon) < MIN_POLYGON_POINTS:
        base = region_box
    else:
        hull = Box.hull(list(polygon))
        rect = largest_inscribed_rect(polygon_mask(polygon, hull), hull)
        base = rect if rect is not None else region_box
    if base.width <= 2 * inset or base.height <= 2 * inset:
        return base
    return base.pad(-inset)
