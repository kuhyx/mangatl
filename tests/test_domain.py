"""Tests for domain models, geometry and reading order."""

from __future__ import annotations

import pytest

from mangatl.domain.geometry import (
    largest_inscribed_rect,
    point_in_polygon,
    polygon_mask,
    text_area,
)
from mangatl.domain.models import (
    Box,
    GlossaryEntry,
    Page,
    RegionKind,
    Stage,
    TextRegion,
)
from mangatl.domain.reading_order import order_regions


class TestBox:
    """Bounding box behaviour."""

    def test_dimensions_and_centre(self) -> None:
        """Dimensions and centre."""
        box = Box(10, 20, 40, 60)
        assert box.width == 30
        assert box.height == 40
        assert box.area == 1200
        assert box.centre == (25.0, 40.0)

    @pytest.mark.parametrize(
        "coords",
        [(10, 10, 10, 20), (10, 10, 20, 10), (10, 10, 5, 20)],
    )
    def test_degenerate_rejected(self, coords: tuple[int, int, int, int]) -> None:
        """Degenerate rejected."""
        with pytest.raises(ValueError, match="degenerate box"):
            Box(*coords)

    def test_pad_grows_and_shrinks(self) -> None:
        """Pad grows and shrinks."""
        box = Box(10, 10, 20, 20)
        assert box.pad(5) == Box(5, 5, 25, 25)
        assert box.pad(-2) == Box(12, 12, 18, 18)

    def test_pad_clamps_to_bounds(self) -> None:
        """Pad clamps to bounds."""
        box = Box(2, 2, 8, 8)
        assert box.pad(10, bounds=Box(0, 0, 12, 12)) == Box(0, 0, 12, 12)

    def test_intersects(self) -> None:
        """Intersects."""
        a = Box(0, 0, 10, 10)
        assert a.intersects(Box(5, 5, 15, 15))
        assert not a.intersects(Box(10, 0, 20, 10))
        assert not a.intersects(Box(0, 10, 10, 20))
        assert not Box(10, 0, 20, 10).intersects(a)
        assert not Box(0, 10, 10, 20).intersects(a)

    def test_hull(self) -> None:
        """Hull."""
        assert Box.hull([(3, 4), (9, 1), (5, 7)]) == Box(3, 1, 10, 8)
        assert Box.hull([(2, 2)]) == Box(2, 2, 3, 3)

    def test_hull_rejects_empty(self) -> None:
        """Hull rejects empty."""
        with pytest.raises(ValueError, match="zero points"):
            Box.hull([])


class TestTextRegion:
    """Region validation and derived state."""

    def test_defaults(self) -> None:
        """Defaults."""
        region = TextRegion(id="a", box=Box(0, 0, 10, 10))
        assert region.kind is RegionKind.BUBBLE
        assert region.vertical
        assert not region.is_translatable

    def test_translatable_requires_non_blank(self) -> None:
        """Translatable requires non blank."""
        assert not TextRegion(id="a", box=Box(0, 0, 5, 5), source_text="   ").is_translatable
        assert TextRegion(id="a", box=Box(0, 0, 5, 5), source_text="あ").is_translatable

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_confidence_bounds(self, value: float) -> None:
        """Confidence bounds."""
        with pytest.raises(ValueError, match="confidence out of range"):
            TextRegion(id="a", box=Box(0, 0, 5, 5), confidence=value)

    def test_short_polygon_rejected(self) -> None:
        """Short polygon rejected."""
        with pytest.raises(ValueError, match="polygon needs"):
            TextRegion(id="a", box=Box(0, 0, 5, 5), polygon=((0, 0), (1, 1)))

    def test_valid_polygon_accepted(self) -> None:
        """Valid polygon accepted."""
        region = TextRegion(id="a", box=Box(0, 0, 5, 5), polygon=((0, 0), (4, 0), (2, 4)))
        assert len(region.polygon) == 3

    def test_with_helpers_return_copies(self) -> None:
        """With helpers return copies."""
        region = TextRegion(id="a", box=Box(0, 0, 5, 5))
        assert region.with_translation("hi").target_text == "hi"
        assert region.with_order(7).order == 7
        assert region.target_text == ""
        assert region.order == -1


class TestGlossaryEntry:
    """Entry validation."""

    def test_valid(self) -> None:
        """Valid."""
        assert GlossaryEntry(source="兄", target="Nii-san").note == ""

    @pytest.mark.parametrize(("src", "tgt"), [("", "x"), ("x", ""), ("  ", "x")])
    def test_blank_rejected(self, src: str, tgt: str) -> None:
        """Blank rejected."""
        with pytest.raises(ValueError, match="non-blank"):
            GlossaryEntry(source=src, target=tgt)


class TestPage:
    """Page container behaviour."""

    def test_box_and_translatable(self) -> None:
        """Box and translatable."""
        page = Page(
            id="p",
            width=100,
            height=200,
            regions=[
                TextRegion(id="a", box=Box(0, 0, 5, 5), source_text="あ"),
                TextRegion(id="b", box=Box(5, 5, 10, 10)),
            ],
        )
        assert page.box == Box(0, 0, 100, 200)
        assert [r.id for r in page.translatable] == ["a"]

    @pytest.mark.parametrize(("w", "h"), [(0, 10), (10, 0), (-1, 5)])
    def test_bad_dimensions(self, w: int, h: int) -> None:
        """Bad dimensions."""
        with pytest.raises(ValueError, match="must be positive"):
            Page(id="p", width=w, height=h)

    def test_fail(self) -> None:
        """Fail."""
        page = Page(id="p", width=10, height=10)
        page.fail("boom")
        assert page.stage is Stage.FAILED
        assert page.error == "boom"


class TestGeometry:
    """Polygon rasterisation and inscribed-rectangle search."""

    SQUARE = ((0, 0), (10, 0), (10, 10), (0, 10))

    def test_point_in_polygon(self) -> None:
        """Point in polygon."""
        assert point_in_polygon(5, 5, self.SQUARE)
        assert not point_in_polygon(15, 5, self.SQUARE)
        assert not point_in_polygon(5, 15, self.SQUARE)
        assert not point_in_polygon(-1, 5, self.SQUARE)

    def test_point_in_concave_polygon(self) -> None:
        """Point in concave polygon."""
        arrow = ((0, 0), (10, 0), (10, 10), (5, 5), (0, 10))
        assert point_in_polygon(2, 2, arrow)
        assert not point_in_polygon(5, 9, arrow)

    def test_polygon_mask_shape(self) -> None:
        """Polygon mask shape."""
        mask = polygon_mask(self.SQUARE, Box(0, 0, 10, 10))
        assert len(mask) == 10
        assert len(mask[0]) == 10
        assert all(all(row) for row in mask)

    def test_largest_inscribed_rect_full(self) -> None:
        """Largest inscribed rect full."""
        mask = [[True] * 6 for _ in range(4)]
        rect = largest_inscribed_rect(mask, Box(0, 0, 6, 4))
        assert rect == Box(0, 0, 6, 4)

    def test_largest_inscribed_rect_notched(self) -> None:
        """Largest inscribed rect notched."""
        mask = [[True] * 6 for _ in range(4)]
        mask[0][0] = False
        rect = largest_inscribed_rect(mask, Box(0, 0, 6, 4))
        assert rect is not None
        assert rect.area == 20

    def test_largest_inscribed_rect_empty_mask(self) -> None:
        """Largest inscribed rect empty mask."""
        assert largest_inscribed_rect([], Box(0, 0, 4, 4)) is None

    def test_largest_inscribed_rect_all_false(self) -> None:
        """Largest inscribed rect all false."""
        mask = [[False] * 4 for _ in range(3)]
        assert largest_inscribed_rect(mask, Box(0, 0, 4, 3)) is None

    def test_text_area_without_polygon(self) -> None:
        """Text area without polygon."""
        assert text_area(Box(0, 0, 100, 100), ()) == Box(4, 4, 96, 96)

    def test_text_area_with_polygon(self) -> None:
        """Text area with polygon."""
        area = text_area(Box(0, 0, 40, 40), ((0, 0), (40, 0), (40, 40), (0, 40)))
        assert area.width < 40
        assert area.height < 40

    def test_text_area_tiny_box_not_inset(self) -> None:
        """Text area tiny box not inset."""
        tiny = Box(0, 0, 5, 5)
        assert text_area(tiny, ()) == tiny

    def test_text_area_falls_back_when_polygon_degenerate(self) -> None:
        """Text area falls back when polygon degenerate."""
        sliver = ((0, 0), (1, 0), (0, 1))
        area = text_area(Box(0, 0, 30, 30), sliver)
        assert area.width >= 1


class TestReadingOrder:
    """Right-to-left manga ordering."""

    def test_empty(self) -> None:
        """Empty."""
        assert order_regions([]) == []

    def test_single(self) -> None:
        """Single."""
        region = TextRegion(id="a", box=Box(0, 0, 10, 10))
        assert [r.order for r in order_regions([region])] == [0]

    def test_two_columns_right_first(self) -> None:
        """Two columns right first."""
        left = TextRegion(id="left", box=Box(0, 0, 40, 40))
        right = TextRegion(id="right", box=Box(200, 0, 240, 40))
        ordered = order_regions([left, right])
        assert [r.id for r in ordered] == ["right", "left"]

    def test_bands_top_first(self) -> None:
        """Bands top first."""
        top = TextRegion(id="top", box=Box(0, 0, 40, 40))
        bottom = TextRegion(id="bottom", box=Box(0, 200, 40, 240))
        assert [r.id for r in order_regions([top, bottom])] == ["top", "bottom"]

    def test_grid(self) -> None:
        """Grid."""
        regions = [
            TextRegion(id="tl", box=Box(0, 0, 40, 40)),
            TextRegion(id="tr", box=Box(200, 0, 240, 40)),
            TextRegion(id="bl", box=Box(0, 200, 40, 240)),
            TextRegion(id="br", box=Box(200, 200, 240, 240)),
        ]
        assert [r.id for r in order_regions(regions)] == ["tr", "tl", "br", "bl"]

    def test_overlapping_cluster_uses_fallback(self) -> None:
        """Overlapping cluster uses fallback."""
        regions = [
            TextRegion(id="a", box=Box(0, 0, 100, 100)),
            TextRegion(id="b", box=Box(50, 50, 150, 150)),
            TextRegion(id="c", box=Box(20, 20, 120, 120)),
        ]
        ordered = order_regions(regions)
        assert [r.id for r in ordered] == ["b", "c", "a"]
        assert [r.order for r in ordered] == [0, 1, 2]

    def test_gap_below_threshold_is_not_a_cut(self) -> None:
        """Gap below threshold is not a cut."""
        regions = [
            TextRegion(id="a", box=Box(0, 0, 40, 40)),
            TextRegion(id="b", box=Box(42, 0, 80, 40)),
        ]
        ordered = order_regions(regions, min_gap=50)
        assert [r.id for r in ordered] == ["b", "a"]

    def test_vertical_only_split(self) -> None:
        """Vertical only split."""
        regions = [
            TextRegion(id="a", box=Box(0, 0, 40, 300)),
            TextRegion(id="b", box=Box(200, 0, 240, 300)),
        ]
        assert [r.id for r in order_regions(regions)] == ["b", "a"]
