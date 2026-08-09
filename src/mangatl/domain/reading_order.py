"""Reading-order estimation for right-to-left manga pages.

Implements the recursive X/Y-cut used by the Manga109 panel-order work: split
the region set by the widest horizontal gap first (top-to-bottom bands), then
by the widest vertical gap within each band (right-to-left columns for manga),
recursing until a group can no longer be split.

Correct ordering matters far more than it looks: the translation prompt feeds
lines to the model in reading order, and out-of-order dialogue is the single
biggest source of wrong pronouns and wrong speaker attribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mangatl.domain.models import TextRegion

MIN_SPLITTABLE = 2


def _gaps(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find empty intervals between the union of 1-D ``spans``.

    Args:
        spans: ``(start, end)`` pairs, ends exclusive.

    Returns:
        The maximal gaps between merged spans, as ``(start, end)`` pairs.
    """
    ordered = sorted(spans)
    gaps: list[tuple[int, int]] = []
    reach = ordered[0][1]
    for start, end in ordered[1:]:
        if start > reach:
            gaps.append((reach, start))
        reach = max(reach, end)
    return gaps


def _widest_gap(spans: list[tuple[int, int]], min_gap: int) -> int | None:
    """Return the midpoint of the widest gap wide enough to cut on.

    Args:
        spans: ``(start, end)`` pairs, ends exclusive.
        min_gap: Smallest gap in pixels that counts as a real separator.

    Returns:
        The cut coordinate, or ``None`` when no gap qualifies.
    """
    gaps = _gaps(spans)
    if not gaps:
        return None
    start, end = max(gaps, key=lambda g: g[1] - g[0])
    if end - start < min_gap:
        return None
    return (start + end) // 2


def _split(
    regions: list[TextRegion],
    min_gap: int,
    *,
    horizontal: bool,
) -> list[list[TextRegion]] | None:
    """Split ``regions`` into groups along one axis, if a clean gap exists.

    Args:
        regions: Regions to split.
        min_gap: Smallest qualifying gap in pixels.
        horizontal: Cut on the y axis when true (bands), x axis when false.

    Returns:
        Two groups ordered for manga reading (top band first; right column
        first), or ``None`` when the axis offers no clean cut.

    Note:
        Both groups are guaranteed non-empty. The cut always lands strictly
        inside a gap between merged spans, so every span starting after the
        gap has its centre past the cut and every span ending at the gap has
        its centre before it. An emptiness guard here would be dead code.
    """
    if horizontal:
        spans = [(r.box.y1, r.box.y2) for r in regions]
    else:
        spans = [(r.box.x1, r.box.x2) for r in regions]
    cut = _widest_gap(spans, min_gap)
    if cut is None:
        return None
    if horizontal:
        first = [r for r in regions if r.box.centre[1] < cut]
        second = [r for r in regions if r.box.centre[1] >= cut]
    else:
        # Manga reads right to left, so the higher-x group comes first.
        first = [r for r in regions if r.box.centre[0] >= cut]
        second = [r for r in regions if r.box.centre[0] < cut]
    return [first, second]


def _fallback(regions: list[TextRegion]) -> list[TextRegion]:
    """Order an unsplittable cluster right-to-left, then top-to-bottom."""
    return sorted(regions, key=lambda r: (-r.box.centre[0], r.box.centre[1]))


def _recurse(regions: list[TextRegion], min_gap: int, *, horizontal: bool) -> list[TextRegion]:
    """Recursively X/Y-cut ``regions`` into reading order.

    Args:
        regions: Regions in the current group.
        min_gap: Smallest qualifying gap in pixels.
        horizontal: Which axis to attempt first at this depth.

    Returns:
        The regions in reading order.
    """
    if len(regions) < MIN_SPLITTABLE:
        return list(regions)
    groups = _split(regions, min_gap, horizontal=horizontal)
    if groups is None:
        groups = _split(regions, min_gap, horizontal=not horizontal)
        if groups is None:
            return _fallback(regions)
        horizontal = not horizontal
    out: list[TextRegion] = []
    for group in groups:
        out.extend(_recurse(group, min_gap, horizontal=not horizontal))
    return out


def order_regions(regions: Sequence[TextRegion], *, min_gap: int = 12) -> list[TextRegion]:
    """Assign right-to-left manga reading order to ``regions``.

    Args:
        regions: Detected text regions in arbitrary order.
        min_gap: Smallest pixel gap treated as a real column/row separator.
            Lower values split more aggressively on tightly packed pages.

    Returns:
        A new list, sorted into reading order, each region carrying its
        ``order`` index.
    """
    if not regions:
        return []
    ordered = _recurse(list(regions), min_gap, horizontal=True)
    return [r.with_order(i) for i, r in enumerate(ordered)]
