"""Integration tests for the AOT-GAN generator, against the real ``torch``.

These are the only tests in the suite that need the ``[ml]`` extra. They exist
because the rest of the suite proves nothing about the real libraries: the unit
tests inject fakes into ``sys.modules``, and a fake that is more capable than
the real object hides exactly the defects that have bitten this project twice
(``Tensor`` having no ``__round__``; a bare ``--flash-attn`` flag).

Run them with::

    pip install -e '.[ml,dev]'
    pytest tests/integration --cov=mangatl.adapters.aot

They are excluded from the default ``pytest`` run by ``testpaths``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="needs the [ml] extra")

from mangatl.adapters.aot import (  # noqa: E402
    BASE_CHANNELS,
    DILATION_RATES,
    NUM_BLOCKS,
    build_modules,
    load_aot_inpainter,
)
from mangatl.adapters.vision import AotInpainter  # noqa: E402
from mangatl.config import Settings  # noqa: E402
from mangatl.domain.models import Box, RegionKind, TextRegion  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def generator() -> object:
    """An untrained generator at the checkpoint's shape."""
    return build_modules()["AOTGenerator"]()


class TestArchitecture:
    """Shape and structure of the network."""

    def test_round_trips_its_input_shape(self, generator: object) -> None:
        """A page in gives a same-sized RGB page out."""
        image = torch.randn(1, 3, 256, 256)
        mask = torch.zeros(1, 1, 256, 256)
        with torch.no_grad():
            out = generator(image, mask)  # type: ignore[operator]
        assert out.shape == (1, 3, 256, 256)

    def test_output_is_clipped_to_the_image_range(self, generator: object) -> None:
        """Inference clips to [-1, 1] so the result is a valid image."""
        image = torch.randn(1, 3, 128, 128) * 10
        mask = torch.ones(1, 1, 128, 128)
        with torch.no_grad():
            out = generator(image, mask)  # type: ignore[operator]
        assert float(out.min()) >= -1.0
        assert float(out.max()) <= 1.0

    def test_has_the_documented_block_count(self, generator: object) -> None:
        """Ten AOT blocks, four dilation rates each."""
        assert len(generator.body_conv) == NUM_BLOCKS  # type: ignore[attr-defined]
        assert generator.body_conv[0].rates == DILATION_RATES  # type: ignore[attr-defined]

    def test_encoder_widths(self, generator: object) -> None:
        """The encoder ends at 4x the base width."""
        image = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            encoded = generator.head(image)  # type: ignore[attr-defined]
        assert encoded.shape[1] == BASE_CHANNELS * 4


class TestScaledWeightStandardisation:
    """The reparameterisation every conv in the checkpoint depends on."""

    def test_weight_is_centred_and_rescaled(self) -> None:
        """Each output channel is zero-mean with a fan-in-scaled norm.

        Getting this wrong still loads the checkpoint and still produces an
        image -- just the wrong one -- so it is worth asserting directly.
        """
        conv = _conv_class()(4, 8, 3)
        with torch.no_grad():
            conv.gain.fill_(1.0)
            weight = conv.standardised_weight()
        per_channel = weight.flatten(1)
        assert torch.allclose(per_channel.mean(dim=1), torch.zeros(8), atol=1e-6)
        # fan_in = in_channels * kh * kw. torch.var_mean is unbiased by
        # default, so the standardised weight's *unbiased* std is what lands on
        # 1/sqrt(fan_in) -- matching upstream, which uses the same default.
        expected = 1.0 / (4 * 3 * 3) ** 0.5
        assert torch.allclose(
            per_channel.std(dim=1, unbiased=True),
            torch.full((8,), expected),
            rtol=1e-3,
        )

    def test_gain_scales_the_weight(self) -> None:
        """Doubling the gain doubles the effective weight."""
        conv = _conv_class()(4, 2, 3)
        with torch.no_grad():
            conv.gain.fill_(1.0)
            base = conv.standardised_weight().clone()
            conv.gain.fill_(2.0)
            doubled = conv.standardised_weight()
        assert torch.allclose(doubled, base * 2, atol=1e-5)


class TestCheckpointLoading:
    """Loading a real state dict."""

    def test_rejects_a_mismatched_checkpoint(self, tmp_path: Path) -> None:
        """A state dict from another architecture is an error, not a silent load."""
        bad = tmp_path / "bad.pt"
        torch.save({"not_a_layer": torch.zeros(1)}, bad)
        with pytest.raises(RuntimeError, match="does not match the AOT-GAN architecture"):
            load_aot_inpainter(bad, "cpu")

    def test_loads_a_matching_checkpoint(self, tmp_path: Path) -> None:
        """A state dict from this architecture round-trips."""
        model = build_modules()["AOTGenerator"]()
        path = tmp_path / "good.pt"
        torch.save(model.state_dict(), path)
        loaded = load_aot_inpainter(path, "cpu")
        assert not loaded.training


def _conv_class() -> object:
    """Return the ``ScaledWSConv2d`` class from a freshly built module set."""
    generator = build_modules()["AOTGenerator"]()
    return type(generator.head[0].conv)


class TestGenerativeFill:
    """The adapter's erasure path, driven by the real network.

    This is the end-to-end shape check the unit suite cannot do: real tensors,
    real convolutions, real composite.
    """

    def test_only_masked_pixels_change(self, tmp_path: Path) -> None:
        """Everything outside the mask is the byte-identical original."""
        weights = tmp_path / "aot.pt"
        torch.save(build_modules()["AOTGenerator"]().state_dict(), weights)

        source = tmp_path / "page.png"
        page = Image.new("RGB", (128, 128), (200, 200, 200))
        for x in range(56, 72):
            for y in range(56, 72):
                page.putpixel((x, y), (0, 0, 0))
        page.save(source)

        settings = Settings(inpaint_weights=weights, device="cpu", data_dir=tmp_path)
        region = TextRegion(id="r", box=Box(40, 40, 88, 88), kind=RegionKind.BUBBLE)
        out = tmp_path / "clean.png"
        AotInpainter(settings).erase(source, [region], out)

        with Image.open(out) as cleaned:
            assert cleaned.size == page.size
            # Far outside the mask: untouched.
            assert cleaned.getpixel((5, 5)) == (200, 200, 200)
            # The ink is gone: the network replaced it with something else.
            assert cleaned.getpixel((64, 64)) != (0, 0, 0)

    def test_reuses_an_already_loaded_model(self, tmp_path: Path) -> None:
        """The checkpoint is read once, not on every page."""
        weights = tmp_path / "aot.pt"
        model = build_modules()["AOTGenerator"]()
        torch.save(model.state_dict(), weights)

        source = tmp_path / "page.png"
        Image.new("RGB", (64, 64), (180, 180, 180)).save(source)
        settings = Settings(inpaint_weights=weights, device="cpu", data_dir=tmp_path)
        region = TextRegion(id="r", box=Box(10, 10, 50, 50), kind=RegionKind.BUBBLE)

        # Passing the model in means the lazy-load branch must be skipped.
        inpainter = AotInpainter(settings, model=model)
        assert inpainter.erase(source, [region], tmp_path / "a.png").exists()
        assert inpainter.erase(source, [region], tmp_path / "b.png").exists()
