"""Vision adapters: balloon detection, Japanese OCR, and text erasure.

Every heavy dependency (``torch``, ``ultralytics``, ``manga_ocr``) is imported
inside the method that needs it, never at module scope. That keeps the import
graph light, keeps startup fast, and lets the test suite exercise every branch
by injecting fakes into ``sys.modules``.

Those imports go through :func:`mangatl.adapters.require`, which converts a
missing dependency into a ``RuntimeError`` the pipeline can catch, so a box
without the ``[ml]`` extra gets a failed page rather than an unhandled 500.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import ImageChops, ImageFilter

from mangatl.adapters import require
from mangatl.domain.models import MIN_POLYGON_POINTS, Box, RegionKind, TextRegion

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mangatl.config import Settings
    from mangatl.domain.models import Page

MIN_REGION_SIDE = 6
# The AOT-GAN checkpoint's working resolution: trained at up to 1024px on the
# long side, and its stride-2 stages need both dimensions divisible by 8.
INPAINT_MAX_SIDE = 1024
INPAINT_PAD_MULTIPLE = 8
# The encoder downsamples 4x and the widest AOT branch dilates by 16, so a side
# shorter than this makes the reflection padding larger than the feature map and
# torch raises. Small crops are scaled up rather than rejected.
INPAINT_MIN_SIDE = 176
# Grow the hole slightly so the network is not asked to reconstruct against a
# glyph's anti-aliased edge. Must be odd; 5 is about two pixels of slack.
MASK_DILATE = 5
# A mask pixel counts as a hole above this; the mask is effectively binary.
MASK_THRESHOLD = 0.5
# Pixels darker than this inside a text region are treated as ink. Manga
# lettering is near-black on a light balloon, so the split is unambiguous.
INK_LUMINANCE = 128
# Contract a balloon outline by this before masking its ink, so the outline
# itself is never erased along with the lettering.
POLYGON_INSET = 0.93


def _fit_for_inpaint(image: Any) -> Any:
    """Scale an image to the generator's working resolution.

    The network was trained at up to 1024px on the long side and its three
    stride-2 stages require both dimensions to be multiples of 8. Oversized
    pages are scaled down rather than tiled: a seam across a balloon looks far
    worse than slightly softer texture.

    Args:
        image: An RGB ``PIL.Image``.

    Returns:
        A copy sized for the network.
    """
    from PIL import Image

    width, height = image.size
    longest = max(width, height)
    if longest > INPAINT_MAX_SIDE:
        ratio = INPAINT_MAX_SIDE / longest
        width, height = round(width * ratio), round(height * ratio)
    if min(width, height) < INPAINT_MIN_SIDE:
        ratio = INPAINT_MIN_SIDE / min(width, height)
        width, height = round(width * ratio), round(height * ratio)
    width = max(INPAINT_MIN_SIDE, width - width % INPAINT_PAD_MULTIPLE)
    height = max(INPAINT_MIN_SIDE, height - height % INPAINT_PAD_MULTIPLE)
    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _dominant_colour(crop: Any) -> tuple[int, int, int]:
    """Return the most common colour in an image crop.

    Args:
        crop: An RGB ``PIL.Image``.

    Returns:
        The modal RGB triple.

    ``getcolors`` is given a ``maxcolors`` equal to the pixel count, so it can
    never exceed its limit and never returns ``None``; a zero-area crop is
    impossible here because :class:`~mangatl.domain.models.Box` guarantees a
    positive width and height. There is therefore no empty case to guard.
    """
    colours = crop.getcolors(crop.width * crop.height)
    _count, colour = max(colours, key=lambda item: item[0])
    return (int(colour[0]), int(colour[1]), int(colour[2]))


def _shrink(
    polygon: Sequence[tuple[int, int]], centre: tuple[int, int], factor: float
) -> list[tuple[int, int]]:
    """Scale a polygon towards its centre.

    Args:
        polygon: Outline points.
        centre: Point to contract towards.
        factor: Scale factor; below 1.0 shrinks.

    Returns:
        The contracted outline.
    """
    cx, cy = centre
    return [(round(cx + (x - cx) * factor), round(cy + (y - cy) * factor)) for x, y in polygon]


class YoloBubbleDetector:
    """Balloon and text-block detection via a YOLO segmentation model.

    Works with any Ultralytics-compatible checkpoint. The recommended weights
    are ``kitsumed/yolov8m_seg-speech-bubble`` (GPL-3.0), which returns masks
    rather than boxes, so the typesetter can fit text to the real balloon
    outline instead of a rectangle.
    """

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        """Build the detector.

        Args:
            settings: Runtime configuration.
            model: Optional pre-loaded model, mainly for tests.
        """
        self._settings = settings
        self._model = model

    def _ensure_model(self) -> Any:
        """Load the YOLO checkpoint on first use."""
        if self._model is None:
            yolo = require("ultralytics").YOLO

            self._model = yolo(str(self._settings.detector_weights))
        return self._model

    def _to_region(
        self,
        index: int,
        xyxy: Sequence[float],
        conf: float,
        page: Page,
        polygon: Sequence[Sequence[float]] = (),
    ) -> TextRegion | None:
        """Convert one raw detection into a domain region, or drop it."""
        # float() first: real ultralytics hands back 0-dim torch Tensors, which
        # have no __round__. Tensor, numpy scalar and plain float all convert.
        x1, y1, x2, y2 = (round(float(v)) for v in xyxy[:4])
        x1 = max(0, min(x1, page.width - 1))
        y1 = max(0, min(y1, page.height - 1))
        x2 = max(x1 + 1, min(x2, page.width))
        y2 = max(y1 + 1, min(y2, page.height))
        box = Box(x1, y1, x2, y2)
        if box.width < MIN_REGION_SIDE or box.height < MIN_REGION_SIDE:
            return None
        points = tuple(
            (
                max(0, min(round(float(p[0])), page.width)),
                max(0, min(round(float(p[1])), page.height)),
            )
            for p in polygon
        )
        return TextRegion(
            id=f"r{index:03d}",
            box=box,
            kind=RegionKind.BUBBLE,
            confidence=max(0.0, min(1.0, float(conf))),
            polygon=points if len(points) >= MIN_POLYGON_POINTS else (),
        )

    def detect(self, image_path: Path, page: Page) -> list[TextRegion]:
        """Detect text regions on ``image_path``.

        Args:
            image_path: Page image.
            page: Page metadata used to clamp coordinates.

        Returns:
            Detected regions, tiny detections dropped.

        The checkpoint is a *segmentation* model, so each detection carries a
        mask as well as a box. Keeping the mask matters: erasing a bounding box
        wipes the balloon outline and any artwork sharing its corners, whereas
        filling the polygon touches only the balloon interior.
        """
        model = self._ensure_model()
        results = model.predict(str(image_path), verbose=False)
        regions: list[TextRegion] = []
        index = 0
        for result in results:
            outlines = self._mask_polygons(result)
            for offset, det in enumerate(result.boxes):
                region = self._to_region(
                    index,
                    list(det.xyxy[0]),
                    float(det.conf[0]),
                    page,
                    outlines[offset] if offset < len(outlines) else (),
                )
                index += 1
                if region is not None:
                    regions.append(region)
        return regions

    @staticmethod
    def _mask_polygons(result: Any) -> list[Sequence[Sequence[float]]]:
        """Return one outline per detection, or an empty list if unavailable.

        Older checkpoints and pure-detection models expose no ``masks`` at all,
        so this degrades to boxes rather than failing.
        """
        masks = getattr(result, "masks", None)
        if masks is None:
            return []
        return list(getattr(masks, "xy", []) or [])


class MangaOcr:
    """Japanese manga OCR via ``kha-white/manga-ocr``.

    Purpose-built for the job: it handles vertical text, furigana, text over
    artwork and decorative fonts, and reads multi-line regions in a single
    forward pass. It is ~444 MB and Apache-2.0, so it is both small and
    genuinely redistributable.
    """

    def __init__(self, settings: Settings, engine: Any | None = None) -> None:
        """Build the OCR adapter.

        Args:
            settings: Runtime configuration.
            engine: Optional pre-loaded engine, mainly for tests.
        """
        self._settings = settings
        self._engine = engine

    def _ensure_engine(self) -> Any:
        """Load the OCR model on first use."""
        if self._engine is None:
            engine = require("manga_ocr").MangaOcr

            self._engine = engine(pretrained_model_name_or_path=self._settings.ocr_model)
        return self._engine

    def read(self, image_path: Path, box: Box) -> str:
        """Recognise the text inside ``box``.

        Args:
            image_path: Page image.
            box: Crop rectangle. Padded slightly, because manga-ocr is more
                accurate with a little margin around the glyphs.

        Returns:
            The recognised string, stripped.
        """
        from PIL import Image

        engine = self._ensure_engine()
        with Image.open(image_path) as img:
            padded = box.pad(4, bounds=Box(0, 0, img.width, img.height))
            crop = img.convert("RGB").crop((padded.x1, padded.y1, padded.x2, padded.y2))
        return str(engine(crop)).strip()


class AotInpainter:
    """Text erasure via AOT-GAN, with a deterministic fill fallback.

    AOT-GAN is chosen over LaMa as the default because the widely-mirrored
    ``big-lama`` weights carry a non-commercial licence in several
    distributions, while the AOT conversion is MIT. If you want maximum
    quality and accept the licence, point ``inpaint_weights`` at an
    anime-finetuned LaMa checkpoint instead.
    """

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        """Build the inpainter.

        Args:
            settings: Runtime configuration.
            model: Optional pre-loaded model. When absent and the weights
                file does not exist, the flat-fill fallback is used.
        """
        self._settings = settings
        self._model = model

    def _flat_fill(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Paint each region with its own interior colour.

        Crude but genuinely useful: inside a white balloon it is visually
        perfect, and it never hallucinates artwork the way a generative model
        can. This is the path taken when no weights are installed.

        Where the detector supplied a mask, the balloon *polygon* is filled and
        the rectangle is not, which is what keeps the outline intact and stops
        artwork sharing the bounding box from being painted over.

        The fill colour is the *most common* colour in the region, not the
        colour of any single sampled pixel: a lone sample lands on a glyph
        stroke often enough, and when it does the balloon is flooded with ink.
        Text is a minority of a balloon's pixels by area, so the mode is the
        background.
        """
        from PIL import Image, ImageDraw

        with Image.open(image_path) as img:
            canvas = img.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for region in regions:
                box = region.box
                fill = _dominant_colour(canvas.crop((box.x1, box.y1, box.x2, box.y2)))
                if len(region.polygon) >= MIN_POLYGON_POINTS:
                    centre = ((box.x1 + box.x2) // 2, (box.y1 + box.y2) // 2)
                    # Shrink slightly so the balloon's own outline survives.
                    draw.polygon(_shrink(region.polygon, centre, 0.97), fill=fill)
                else:
                    draw.rectangle((box.x1, box.y1, box.x2 - 1, box.y2 - 1), fill=fill)
            canvas.save(out_path)
        return out_path

    def erase(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Remove source text and write the cleaned page.

        Args:
            image_path: Page image.
            regions: Regions to erase.
            out_path: Destination path.

        Returns:
            The written path.
        """
        if self._model is None and not self._settings.inpaint_weights.exists():
            return self._flat_fill(image_path, regions, out_path)
        if self._model is None:
            from mangatl.adapters.aot import load_aot_inpainter

            self._model = load_aot_inpainter(self._settings.inpaint_weights, self._settings.device)
        return self._generative_fill(image_path, regions, out_path)

    def _mask_image(self, page: Any, regions: Sequence[TextRegion], size: tuple[int, int]) -> Any:
        """Build the hole mask: white where text must be removed.

        Only the **glyphs** are masked, not the balloon. The detector's region
        covers the whole balloon interior, and handing that to the generator
        asks it to hallucinate an entire balloon rather than erase some
        lettering -- measurably worse than a flat fill, because the network
        rebuilds surrounding texture it was never asked to touch.

        So the region is used only to *restrict where we look*, and within it
        the ink itself is selected by luminance. Manga lettering is near-black
        on a light balloon, which makes a simple threshold reliable here.
        """
        from PIL import Image, ImageDraw

        region_mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(region_mask)
        for region in regions:
            box = region.box
            if len(region.polygon) >= MIN_POLYGON_POINTS:
                # Shrink first: the balloon's own outline is ink too, and
                # erasing it would cost the page its balloons.
                centre = ((box.x1 + box.x2) // 2, (box.y1 + box.y2) // 2)
                draw.polygon(_shrink(region.polygon, centre, POLYGON_INSET), fill=255)
            else:
                draw.rectangle((box.x1, box.y1, box.x2 - 1, box.y2 - 1), fill=255)
        # Ink inside those regions: dark pixels are lettering, everything
        # lighter is balloon or artwork and must survive untouched.
        ink = page.convert("L").point(lambda v: 255 if v < INK_LUMINANCE else 0)
        glyphs = ImageChops.multiply(region_mask, ink)
        # A few pixels of slack, so the network is not asked to reconstruct
        # right up against a glyph's anti-aliased edge.
        return glyphs.filter(ImageFilter.MaxFilter(MASK_DILATE))

    def _generative_fill(
        self, image_path: Path, regions: Sequence[TextRegion], out_path: Path
    ) -> Path:
        """Erase with the AOT-GAN generator and composite the result.

        Only the masked pixels are taken from the network; everything else is
        the untouched original, so the artwork can never be subtly redrawn.
        """
        torch = require("torch")
        numpy = require("numpy")
        from PIL import Image

        with Image.open(image_path) as handle:
            original = handle.convert("RGB")
        mask = self._mask_image(original, regions, original.size)

        # The checkpoint works at a bounded resolution, on multiples of 8.
        work = _fit_for_inpaint(original)
        work_mask = mask.resize(work.size, Image.Resampling.BILINEAR)

        img_t = torch.from_numpy(numpy.array(work, dtype="float32")).permute(2, 0, 1)[None]
        img_t = img_t / 127.5 - 1.0
        mask_t = torch.from_numpy(numpy.array(work_mask, dtype="float32"))[None, None] / 255.0
        mask_t = (mask_t >= MASK_THRESHOLD).float()

        device = self._settings.device
        img_t, mask_t = img_t.to(device), mask_t.to(device)
        model: Any = self._model
        with torch.no_grad():
            generated = model(img_t * (1 - mask_t), mask_t)

        array = ((generated[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255)
        filled = Image.fromarray(array.astype("uint8")).resize(
            original.size, Image.Resampling.BILINEAR
        )
        Image.composite(filled, original, mask).save(out_path)
        return out_path
