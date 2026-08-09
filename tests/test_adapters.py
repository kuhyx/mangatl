"""Tests for the concrete adapters.

None of the heavy ML dependencies are installed. Every adapter imports them
lazily inside the call site, so each test injects a fake module into
``sys.modules`` and exercises the real adapter code against it.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image, ImageDraw

from conftest import (
    FakeHttpClient,
    FakeInpainter,
    FakeOcr,
    FakeResponse,
    FakeTranslator,
    FakeTypesetter,
    ScriptedBackend,
    lines_reply,
)
from mangatl.adapters import require
from mangatl.adapters.translate_libre import LibreTranslateTranslator
from mangatl.adapters.translate_llm import HttpChatBackend, LlmTranslator
from mangatl.adapters.typeset import PillowTypesetter, wrap_text
from mangatl.adapters.vision import (
    INPAINT_MAX_SIDE,
    INPAINT_MIN_SIDE,
    INPAINT_PAD_MULTIPLE,
    AotInpainter,
    MangaOcr,
    YoloBubbleDetector,
    _fit_for_inpaint,
)
from mangatl.domain.models import Box, GlossaryEntry, Page, RegionKind, Stage, TextRegion
from mangatl.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

    from mangatl.config import Settings


def _region(box: Box, polygon: tuple[tuple[int, int], ...] = ()) -> TextRegion:
    """Build a minimal region for the erasure tests.

    Args:
        box: Bounding box.
        polygon: Optional balloon outline.

    Returns:
        A translatable region.
    """
    return TextRegion(id="r000", box=box, kind=RegionKind.BUBBLE, polygon=polygon)


def _raise_missing(real: Callable[..., ModuleType]) -> Callable[..., ModuleType]:
    """Wrap ``import_module`` so ``ultralytics`` looks uninstalled.

    Args:
        real: The genuine ``importlib.import_module``.

    Returns:
        A replacement that fails for ultralytics and defers otherwise.
    """

    def _import(name: str, package: str | None = None) -> ModuleType:
        if name == "ultralytics":
            msg = "No module named 'ultralytics'"
            raise ModuleNotFoundError(msg)
        return real(name, package)

    return _import


class _Det:
    """One fake YOLO detection."""

    def __init__(self, xyxy: list[float], conf: float) -> None:
        """Build the fixture."""
        self.xyxy = [xyxy]
        self.conf = [conf]


class _TensorLike:
    """A stand-in for the 0-dim ``torch.Tensor`` real ultralytics returns.

    Deliberately defines ``__float__`` but **not** ``__round__``, which is
    exactly how a real Tensor behaves. The original fake handed back plain
    floats, so ``round(v)`` passed in the suite and raised
    ``TypeError: type Tensor doesn't define __round__ method`` against the real
    library on the first GPU run.
    """

    def __init__(self, value: float) -> None:
        """Build the fixture."""
        self._value = value

    def __float__(self) -> float:
        """Convert to float, the only numeric protocol a Tensor offers here."""
        return self._value


class _FakeMasks:
    """The ``masks`` attribute of a segmentation result."""

    def __init__(self, xy: list[list[tuple[float, float]]]) -> None:
        """Build the fixture."""
        self.xy = xy


class _Result:
    """One fake YOLO result."""

    def __init__(self, boxes: list[_Det]) -> None:
        """Build the fixture."""
        self.boxes = boxes
        self.masks: _FakeMasks | None = None


class _FakeYolo:
    """Fake Ultralytics model."""

    def __init__(self, results: list[_Result]) -> None:
        """Build the fixture."""
        self.results = results
        self.loaded_from: str | None = None

    def predict(self, source: str, *, verbose: bool) -> list[_Result]:
        """Fake predict used by the tests."""
        del source, verbose
        return self.results


class TestYoloBubbleDetector:
    """Detection, clamping and filtering."""

    def test_detect_maps_and_clamps(self, settings: Settings, page_image: Path) -> None:
        """Detect maps and clamps."""
        model = _FakeYolo([_Result([_Det([10.4, 20.6, 120.0, 140.0], 0.91)])])
        detector = YoloBubbleDetector(settings, model=model)
        page = Page(id="p", width=400, height=600)
        regions = detector.detect(page_image, page)
        assert len(regions) == 1
        assert regions[0].id == "r000"
        assert regions[0].box == Box(10, 21, 120, 140)
        assert regions[0].confidence == pytest.approx(0.91)

    def test_detect_drops_tiny_regions(self, settings: Settings, page_image: Path) -> None:
        """Detect drops tiny regions."""
        model = _FakeYolo([_Result([_Det([10, 10, 13, 13], 0.9), _Det([0, 0, 100, 100], 0.5)])])
        detector = YoloBubbleDetector(settings, model=model)
        regions = detector.detect(page_image, Page(id="p", width=400, height=600))
        assert [r.id for r in regions] == ["r001"]

    def test_detect_clamps_out_of_bounds(self, settings: Settings, page_image: Path) -> None:
        """Detect clamps out of bounds."""
        model = _FakeYolo([_Result([_Det([-50, -50, 9999, 9999], 1.5)])])
        detector = YoloBubbleDetector(settings, model=model)
        regions = detector.detect(page_image, Page(id="p", width=400, height=600))
        assert regions[0].box == Box(0, 0, 400, 600)
        assert regions[0].confidence == pytest.approx(1.0)

    def test_detect_clamps_inverted_box(self, settings: Settings, page_image: Path) -> None:
        """Detect clamps inverted box."""
        model = _FakeYolo([_Result([_Det([300, 300, 10, 10], 0.4)])])
        detector = YoloBubbleDetector(settings, model=model)
        assert detector.detect(page_image, Page(id="p", width=400, height=600)) == []

    def test_lazy_model_load(
        self,
        settings: Settings,
        page_image: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lazy model load."""
        loaded: list[str] = []

        def _yolo(path: str) -> _FakeYolo:
            loaded.append(path)
            return _FakeYolo([_Result([])])

        module = types.ModuleType("ultralytics")
        module.YOLO = _yolo  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", module)
        detector = YoloBubbleDetector(settings)
        assert detector.detect(page_image, Page(id="p", width=400, height=600)) == []
        assert loaded == [str(settings.detector_weights)]

    def test_accepts_tensor_like_coordinates(self, settings: Settings, page_image: Path) -> None:
        """Coordinates that only support ``float()`` must still convert.

        Regression: real ultralytics returns 0-dim torch Tensors, which have no
        ``__round__``. Detection crashed on the first real page while the suite
        stayed green, because the fake used plain floats.
        """
        det = _Det(
            [_TensorLike(10.4), _TensorLike(20.6), _TensorLike(90.2), _TensorLike(80.8)], 0.9
        )  # type: ignore[list-item]
        model = _FakeYolo([_Result([det])])
        detector = YoloBubbleDetector(settings, model=model)
        regions = detector.detect(page_image, Page(id="p", width=400, height=600))
        assert [(r.box.x1, r.box.y1, r.box.x2, r.box.y2) for r in regions] == [(10, 21, 90, 81)]

    def test_segmentation_masks_become_polygons(self, settings: Settings, page_image: Path) -> None:
        """A segmentation result keeps its outline, not just its box.

        Regression: the masks were discarded, so erasure could only paint
        rectangles — wiping balloon borders and neighbouring artwork.
        """
        result = _Result([_Det([10, 10, 100, 100], 0.9)])
        result.masks = _FakeMasks([[(20.4, 20.6), (90, 20), (90, 90), (20, 90)]])
        detector = YoloBubbleDetector(settings, model=_FakeYolo([result]))
        regions = detector.detect(page_image, Page(id="p", width=400, height=600))
        assert regions[0].polygon == ((20, 21), (90, 20), (90, 90), (20, 90))

    def test_missing_masks_degrade_to_boxes(self, settings: Settings, page_image: Path) -> None:
        """A detection-only checkpoint still works, with an empty polygon."""
        model = _FakeYolo([_Result([_Det([10, 10, 100, 100], 0.9)])])
        detector = YoloBubbleDetector(settings, model=model)
        regions = detector.detect(page_image, Page(id="p", width=400, height=600))
        assert regions[0].polygon == ()

    def test_degenerate_mask_is_discarded(self, settings: Settings, page_image: Path) -> None:
        """An outline with too few points falls back to the box."""
        result = _Result([_Det([10, 10, 100, 100], 0.9)])
        result.masks = _FakeMasks([[(20, 20), (90, 90)]])
        detector = YoloBubbleDetector(settings, model=_FakeYolo([result]))
        regions = detector.detect(page_image, Page(id="p", width=400, height=600))
        assert regions[0].polygon == ()


class TestMangaOcr:
    """OCR cropping and lazy engine loading."""

    def test_read_crops_and_strips(self, settings: Settings, page_image: Path) -> None:
        """Read crops and strips."""
        seen: list[tuple[int, int]] = []

        def engine(image: Image.Image) -> str:
            """Fake engine used by the tests."""
            seen.append(image.size)
            return "  こんにちは  "

        ocr = MangaOcr(settings, engine=engine)
        assert ocr.read(page_image, Box(40, 40, 160, 140)) == "こんにちは"
        assert seen == [(128, 108)]

    def test_read_clamps_padding_at_edges(self, settings: Settings, page_image: Path) -> None:
        """Read clamps padding at edges."""
        ocr = MangaOcr(settings, engine=lambda _img: "あ")
        assert ocr.read(page_image, Box(0, 0, 20, 20)) == "あ"

    def test_lazy_engine_load(
        self,
        settings: Settings,
        page_image: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lazy engine load."""
        built: list[str] = []

        class Engine:
            def __init__(self, pretrained_model_name_or_path: str) -> None:
                """Build the fixture."""
                built.append(pretrained_model_name_or_path)

            def __call__(self, image: Image.Image) -> str:
                del image
                return "テスト"

        module = types.ModuleType("manga_ocr")
        module.MangaOcr = Engine  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "manga_ocr", module)
        ocr = MangaOcr(settings)
        assert ocr.read(page_image, Box(10, 10, 60, 60)) == "テスト"
        assert built == [settings.ocr_model]


class TestAotInpainter:
    """Text erasure with and without weights present."""

    def test_flat_fill_when_no_weights(
        self,
        settings: Settings,
        page_image: Path,
        tmp_path: Path,
    ) -> None:
        """Flat fill when no weights."""
        out = tmp_path / "clean.png"
        AotInpainter(settings).erase(page_image, [_region(Box(40, 40, 160, 140))], out)
        with Image.open(out) as img:
            assert img.getpixel((100, 100)) == (20, 20, 20)

    def test_polygon_fill_spares_the_corners(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """A masked region fills its outline, not its bounding box.

        Regression: the detector discarded the segmentation masks, so erasure
        painted the whole rectangle — wiping the balloon border and any
        artwork that merely shared the box. Here the corner pixel must survive
        while the centre is erased.
        """
        # Grey artwork everywhere; only the diamond-shaped "balloon" is white.
        # A box-shaped fill would sample white and paint the whole rectangle,
        # destroying the grey at the corners.
        source = tmp_path / "page.png"
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        for x in range(20, 80):
            for y in range(20, 80):
                # A wide diamond: white dominates its own bounding box, as a
                # real balloon interior does, with grey artwork at the corners.
                if abs(x - 50) * 3 + abs(y - 50) * 3 <= 100:
                    img.putpixel((x, y), (255, 255, 255))
        img.save(source)

        diamond = ((50, 21), (79, 50), (50, 79), (21, 50))
        region = _region(Box(20, 20, 80, 80), polygon=diamond)
        out = tmp_path / "clean.png"
        AotInpainter(settings).erase(source, [region], out)

        with Image.open(out) as cleaned:
            assert cleaned.getpixel((50, 50)) == (255, 255, 255)
            # Outside the diamond but inside the bounding box: still artwork.
            assert cleaned.getpixel((23, 23)) == (128, 128, 128)

    def test_fill_ignores_text_and_uses_the_background(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """The fill is the balloon's background, not whatever a probe landed on.

        Regression: the colour was sampled from the single centre pixel. When
        that pixel fell on a glyph stroke the whole balloon was flooded with
        ink — a solid black bubble in the rendered page.
        """
        source = tmp_path / "page.png"
        img = Image.new("RGB", (60, 60), (255, 255, 255))
        # A black glyph sitting exactly under the centre probe.
        for x in range(25, 36):
            for y in range(25, 36):
                img.putpixel((x, y), (0, 0, 0))
        img.save(source)

        out = tmp_path / "clean.png"
        AotInpainter(settings).erase(source, [_region(Box(10, 10, 50, 50))], out)
        with Image.open(out) as cleaned:
            assert cleaned.getpixel((30, 30)) == (255, 255, 255)

    def test_smallest_possible_region(
        self,
        settings: Settings,
        page_image: Path,
        tmp_path: Path,
    ) -> None:
        """A 1x1 region is the smallest ``Box`` allows and must still work."""
        out = tmp_path / "clean.png"
        assert AotInpainter(settings).erase(page_image, [_region(Box(5, 5, 6, 6))], out) == out

    def test_flat_fill_with_no_boxes(
        self,
        settings: Settings,
        page_image: Path,
        tmp_path: Path,
    ) -> None:
        """Flat fill with no boxes."""
        out = tmp_path / "clean.png"
        assert AotInpainter(settings).erase(page_image, [], out) == out

    def test_mask_covers_glyphs_not_the_whole_balloon(
        self,
        settings: Settings,
    ) -> None:
        """The generative mask selects ink, not the region.

        Regression: masking the whole balloon asked the network to hallucinate
        an entire balloon rather than erase lettering, which measured worse
        than a flat fill. Only dark pixels inside the region may be masked.
        """
        page = Image.new("RGB", (60, 60), (255, 255, 255))
        for x in range(25, 36):
            for y in range(25, 36):
                page.putpixel((x, y), (0, 0, 0))
        region = _region(Box(10, 10, 50, 50))
        mask = AotInpainter(settings)._mask_image(page, [region], page.size)
        assert mask.getpixel((30, 30)) == 255
        # Inside the region but not ink: must be left for the original page.
        assert mask.getpixel((14, 14)) == 0

    def test_mask_spares_the_balloon_outline(
        self,
        settings: Settings,
    ) -> None:
        """The outline is ink too, and must not be erased with the lettering.

        Regression: masking every dark pixel inside the balloon polygon took
        the balloon's own border with it, which measured worse than a flat
        fill. The polygon is inset before the ink is selected.
        """
        page = Image.new("RGB", (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        outline = ((200, 60), (340, 200), (200, 340), (60, 200))
        draw.polygon(outline, outline=(0, 0, 0), width=6)
        draw.rectangle((170, 170, 230, 230), fill=(0, 0, 0))

        region = _region(Box(60, 60, 341, 341), polygon=outline)
        mask = AotInpainter(settings)._mask_image(page, [region], page.size)
        # Lettering is masked...
        assert mask.getpixel((200, 200)) == 255
        # ...but the balloon's own border is not.
        assert mask.getpixel((200, 62)) == 0

    def test_mask_falls_back_to_the_box_without_a_polygon(
        self,
        settings: Settings,
    ) -> None:
        """A region with no outline still masks the ink inside its box."""
        page = Image.new("RGB", (40, 40), (255, 255, 255))
        page.putpixel((20, 20), (0, 0, 0))
        mask = AotInpainter(settings)._mask_image(page, [_region(Box(5, 5, 35, 35))], page.size)
        assert mask.getpixel((20, 20)) == 255


class TestFitForInpaint:
    """Sizing an image for the generator."""

    def test_shrinks_an_oversized_page(self) -> None:
        """A page longer than the working size is scaled down."""
        fitted = _fit_for_inpaint(Image.new("RGB", (2048, 1024)))
        assert max(fitted.size) == INPAINT_MAX_SIDE
        assert fitted.size == (1024, 512)

    def test_rounds_down_to_a_multiple_of_eight(self) -> None:
        """The three stride-2 stages need both sides divisible by 8."""
        fitted = _fit_for_inpaint(Image.new("RGB", (1001, 667)))
        assert fitted.size[0] % INPAINT_PAD_MULTIPLE == 0
        assert fitted.size[1] % INPAINT_PAD_MULTIPLE == 0

    def test_leaves_an_already_valid_size_alone(self) -> None:
        """No resample when the size already fits: resizing costs sharpness."""
        image = Image.new("RGB", (640, 320))
        assert _fit_for_inpaint(image) is image

    def test_scales_a_tiny_crop_up(self) -> None:
        """Below the minimum the widest dilation exceeds the feature map.

        Regression: a small page raised ``Padding size should be less than the
        corresponding input dimension`` from torch instead of being erased.
        """
        fitted = _fit_for_inpaint(Image.new("RGB", (32, 32)))
        assert min(fitted.size) >= INPAINT_MIN_SIDE


class TestWrapText:
    """Greedy pixel-width wrapping."""

    def test_empty(self) -> None:
        """Empty."""
        assert wrap_text("   ", 100, len) == []

    def test_single_line_fits(self) -> None:
        """Single line fits."""
        assert wrap_text("a b", 100, lambda s: len(s) * 1.0) == ["a b"]

    def test_wraps_on_overflow(self) -> None:
        """Wraps on overflow."""
        assert wrap_text("aa bb cc", 5, lambda s: len(s) * 1.0) == ["aa bb", "cc"]

    def test_oversized_word_gets_own_line(self) -> None:
        """Oversized word gets own line."""
        assert wrap_text("aa enormous bb", 4, lambda s: len(s) * 1.0) == ["aa", "enormous", "bb"]


class TestPillowTypesetter:
    """Font fitting and rendering."""

    def test_renders_text(self, page_image: Path, font_file: Path, tmp_path: Path) -> None:
        """Renders text."""
        out = tmp_path / "render.png"
        regions = [TextRegion(id="a", box=Box(40, 40, 300, 300), target_text="Hello there")]
        assert PillowTypesetter(font_file).render(page_image, regions, out) == out
        assert out.exists()

    def test_skips_blank_text(self, page_image: Path, font_file: Path, tmp_path: Path) -> None:
        """Skips blank text."""
        out = tmp_path / "render.png"
        regions = [TextRegion(id="a", box=Box(40, 40, 300, 300), target_text="   ")]
        PillowTypesetter(font_file).render(page_image, regions, out)
        with Image.open(page_image) as before, Image.open(out) as after:
            assert before.convert("RGB").tobytes() == after.convert("RGB").tobytes()

    def test_skips_region_too_small_for_min_font(
        self,
        page_image: Path,
        font_file: Path,
        tmp_path: Path,
    ) -> None:
        """Skips region too small for min font."""
        out = tmp_path / "render.png"
        regions = [TextRegion(id="a", box=Box(0, 0, 9, 9), target_text="Extremely long sentence")]
        PillowTypesetter(font_file).render(page_image, regions, out)
        assert out.exists()

    def test_uses_polygon_when_present(
        self,
        page_image: Path,
        font_file: Path,
        tmp_path: Path,
    ) -> None:
        """Uses polygon when present."""
        out = tmp_path / "render.png"
        regions = [
            TextRegion(
                id="a",
                box=Box(40, 40, 300, 300),
                polygon=((40, 40), (300, 40), (300, 300), (40, 300)),
                target_text="Hi",
            ),
        ]
        assert PillowTypesetter(font_file).render(page_image, regions, out) == out


class TestHttpChatBackend:
    """OpenAI-compatible chat plumbing."""

    def test_chat_returns_content(self, settings: Settings) -> None:
        """Chat returns content."""
        client = FakeHttpClient([FakeResponse({"choices": [{"message": {"content": "hi"}}]})])
        backend = HttpChatBackend(settings, client=client)
        assert backend.chat("sys", "user") == "hi"
        url, body = client.requests[0]
        assert url.endswith("/chat/completions")
        assert body["model"] == settings.llm_model
        assert body["messages"][0]["role"] == "system"

    def test_chat_raises_without_choices(self, settings: Settings) -> None:
        """Chat raises without choices."""
        client = FakeHttpClient([FakeResponse({"choices": []})])
        with pytest.raises(RuntimeError, match="no choices"):
            HttpChatBackend(settings, client=client).chat("s", "u")

    def test_chat_propagates_http_error(self, settings: Settings) -> None:
        """Chat propagates http error."""
        client = FakeHttpClient([FakeResponse({}, status=500)])
        with pytest.raises(RuntimeError, match="HTTP 500"):
            HttpChatBackend(settings, client=client).chat("s", "u")

    def test_lazy_client_creation(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lazy client creation."""
        made: list[float] = []

        def client_factory(*, timeout: float) -> FakeHttpClient:
            """Fake client factory used by the tests."""
            made.append(timeout)
            return FakeHttpClient([FakeResponse({"choices": [{"message": {"content": "ok"}}]})])

        module = types.ModuleType("httpx")
        module.Client = client_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "httpx", module)
        assert HttpChatBackend(settings).chat("s", "u") == "ok"
        assert made == [settings.llm_timeout_s]


class TestLlmTranslator:
    """The recommended translation backend."""

    def test_translate_page(self, settings: Settings) -> None:
        """Translate page."""
        backend = ScriptedBackend([lines_reply(["Hello"]), lines_reply(["Hello there"])])
        translator = LlmTranslator(settings, backend=backend)
        assert translator.translate_page(["こんにちは"]) == ["Hello there"]
        assert translator.name == "local-llm"

    def test_glossary_flows_into_prompt(self, settings: Settings) -> None:
        """Glossary flows into prompt."""
        backend = ScriptedBackend([lines_reply(["Nii-san"]), lines_reply(["Nii-san"])])
        translator = LlmTranslator(settings, backend=backend)
        translator.translate_page(["兄"], glossary=[GlossaryEntry(source="兄", target="Nii-san")])
        assert "Nii-san" in backend.prompts[0][1]

    def test_critique_disabled_by_settings(self, settings: Settings) -> None:
        """Critique disabled by settings."""
        settings.critique_pass = False
        backend = ScriptedBackend([lines_reply(["Hello"])])
        assert LlmTranslator(settings, backend=backend).translate_page(["あ"]) == ["Hello"]
        assert len(backend.prompts) == 1


class TestLibreTranslateTranslator:
    """The offline fallback."""

    def test_translate_lines(self, settings: Settings) -> None:
        """Translate lines."""
        client = FakeHttpClient(
            [FakeResponse({"translatedText": "Hello"}), FakeResponse({"translatedText": "Bye"})],
        )
        translator = LibreTranslateTranslator(settings, client=client)
        assert translator.translate_page(["こんにちは", "またね"]) == ["Hello", "Bye"]
        assert translator.name == "libretranslate"
        assert client.requests[0][1]["source"] == "ja"

    def test_blank_lines_skipped(self, settings: Settings) -> None:
        """Blank lines skipped."""
        client = FakeHttpClient([FakeResponse({"translatedText": "Hello"})])
        translator = LibreTranslateTranslator(settings, client=client)
        assert translator.translate_page(["  ", "こんにちは"]) == ["", "Hello"]

    def test_glossary_applied_post_hoc(self, settings: Settings) -> None:
        """Glossary applied post hoc."""
        client = FakeHttpClient([FakeResponse({"translatedText": "Hello 兄 there"})])
        translator = LibreTranslateTranslator(settings, client=client)
        out = translator.translate_page(
            ["兄"],
            glossary=[
                GlossaryEntry(source="兄", target="Nii-san"),
                GlossaryEntry(source="お兄ちゃん", target="Onii-chan"),
            ],
        )
        assert out == ["Hello Nii-san there"]

    def test_missing_translated_text_key(self, settings: Settings) -> None:
        """Missing translated text key."""
        client = FakeHttpClient([FakeResponse({})])
        assert LibreTranslateTranslator(settings, client=client).translate_page(["あ"]) == [""]

    def test_lazy_client_creation(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lazy client creation."""

        def client_factory(*, timeout: float) -> FakeHttpClient:
            """Fake client factory used by the tests."""
            del timeout
            return FakeHttpClient([FakeResponse({"translatedText": "Hi"})])

        module = types.ModuleType("httpx")
        module.Client = client_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "httpx", module)
        assert LibreTranslateTranslator(settings).translate_page(["あ"]) == ["Hi"]

    def test_context_is_ignored(self, settings: Settings) -> None:
        """Context is ignored."""
        client = FakeHttpClient([FakeResponse({"translatedText": "Hi"})])
        translator = LibreTranslateTranslator(settings, client=client)
        assert translator.translate_page(["あ"], context=["ignored"]) == ["Hi"]


def test_fake_response_json_passthrough() -> None:
    """The test double itself must behave, or the tests above prove nothing."""
    response: Any = FakeResponse({"a": 1})
    response.raise_for_status()
    assert response.json() == {"a": 1}


class TestRequire:
    """The optional-dependency guard.

    Regression cover for a real defect: ``ModuleNotFoundError`` subclasses
    ``ImportError``, which is *not* among the exceptions
    :meth:`mangatl.pipeline.Pipeline.run` catches. Without this guard a machine
    missing the ``[ml]`` extra returned an unhandled 500 instead of a failed
    page. The rest of the suite cannot catch that, because every other test
    injects a fake into ``sys.modules`` so the import always succeeds.
    """

    def test_returns_the_module_when_present(self) -> None:
        """A module that is installed comes back unchanged."""
        assert require("json") is sys.modules["json"]

    def test_missing_module_becomes_runtime_error(self) -> None:
        """A missing module raises RuntimeError, not ImportError."""
        with pytest.raises(RuntimeError, match=r"not installed.*'ml' extra") as caught:
            require("mangatl_definitely_not_installed")
        assert isinstance(caught.value.__cause__, ImportError)

    def test_error_names_the_extra(self) -> None:
        """The message tells the user which extra to install."""
        with pytest.raises(RuntimeError, match=r"pip install -e '\.\[vision\]'"):
            require("mangatl_definitely_not_installed", extra="vision")

    def test_missing_dependency_fails_the_page_not_the_request(
        self,
        settings: Settings,
        page_image: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End to end: no ultralytics installed means a FAILED page, not a 500."""
        monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
        monkeypatch.setattr(importlib, "import_module", _raise_missing(importlib.import_module))
        pipeline = Pipeline(
            detector=YoloBubbleDetector(settings),
            ocr=FakeOcr(),
            translator=FakeTranslator(),
            inpainter=FakeInpainter(),
            typesetter=FakeTypesetter(),
        )
        page = Page(id="p", width=400, height=600)
        result = pipeline.run(page_image, page, out_dir=tmp_path)
        assert result.page.stage is Stage.FAILED
        assert "ultralytics" in (result.page.error or "")
        assert result.rendered_path is None
