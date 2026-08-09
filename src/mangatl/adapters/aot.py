"""AOT-GAN inpainting generator.

A port of the generator from ``zyddnys/manga-image-translator`` (GPL-3.0,
compatible with this project's AGPL-3.0-or-later), matching the checkpoint
published as ``mayocream/aot-inpainting`` (MIT). That checkpoint ships a raw
``model.safetensors`` state dict and no code, so the network has to exist here
for the weights to be loadable at all.

The design is AOT-GAN (Zeng et al., "Aggregated Contextual Transformations for
High-Resolution Image Inpainting"): an encoder, ten aggregated-contextual-
transformation blocks that fuse four dilation rates, and a decoder. Every
convolution is a *gated* convolution built on scaled weight standardisation, so
each layer carries a ``gain`` alongside its weight.

``torch`` is imported lazily by the caller, never at module scope, so importing
:mod:`mangatl.adapters` still costs nothing on a machine without the ``[ml]``
extra. Everything here is constructed only from inside
:func:`load_aot_inpainter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mangatl.adapters import require

if TYPE_CHECKING:
    from pathlib import Path

# The published checkpoint's fixed hyper-parameters. They are not configurable:
# these are the shapes the weights were trained with.
BASE_CHANNELS = 32
NUM_BLOCKS = 10
DILATION_RATES = (2, 4, 8, 16)
PAD_MULTIPLE = 8
DEFAULT_MAX_SIDE = 1024
WS_EPS = 1e-4
GATE_SCALE = 1.8
LAYER_NORM_SCALE = 5.0
# relu(x) * this keeps the variance stable in a normaliser-free net.
RELU_NF_GAIN = 1.7139588594436646


def build_modules() -> dict[str, Any]:
    """Define the network against a torch imported at call time.

    Returns:
        A mapping of class name to class object.

    Building the classes inside a function is what keeps ``torch`` out of the
    import graph: :class:`torch.nn.Module` cannot be subclassed at module scope
    without importing torch there.
    """
    torch = require("torch")
    nn = torch.nn
    functional = torch.nn.functional

    class ScaledWSConv2d(nn.Conv2d):  # type: ignore[misc, name-defined]
        """Conv2d with scaled weight standardisation.

        The weight is centred and rescaled per output channel before use, then
        multiplied by a learned ``gain``. This is the NFNet formulation and it
        is why every conv in the checkpoint has a ``gain`` tensor.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Add the per-output-channel gain."""
            super().__init__(*args, **kwargs)
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))

        def standardised_weight(self) -> Any:
            """Return the centred, rescaled, gained weight."""
            var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
            fan_in = torch.prod(torch.tensor(self.weight.shape[1:], device=self.weight.device))
            scale = torch.rsqrt(torch.clamp(var * fan_in, min=WS_EPS)) * self.gain
            return self.weight * scale - (mean * scale)

        def forward(self, x: Any) -> Any:
            """Convolve with the standardised weight."""
            return functional.conv2d(
                x,
                self.standardised_weight(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    class ScaledWSTransposeConv2d(nn.ConvTranspose2d):  # type: ignore[misc, name-defined]
        """ConvTranspose2d with the same standardisation.

        Note the gain is sized by ``in_channels``: a transposed convolution's
        weight is ``(in, out, kh, kw)``, so the leading dim is the input.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Add the per-input-channel gain."""
            super().__init__(*args, **kwargs)
            self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))

        def standardised_weight(self) -> Any:
            """Return the centred, rescaled, gained weight."""
            var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
            fan_in = torch.prod(torch.tensor(self.weight.shape[1:], device=self.weight.device))
            scale = torch.rsqrt(torch.clamp(var * fan_in, min=WS_EPS)) * self.gain
            return self.weight * scale - (mean * scale)

        def forward(self, x: Any, output_size: Any = None) -> Any:
            """Deconvolve with the standardised weight."""
            del output_size
            return functional.conv_transpose2d(
                x,
                self.standardised_weight(),
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )

    class GatedWSConvPadded(nn.Module):  # type: ignore[misc, name-defined]
        """Reflection-padded gated convolution."""

        def __init__(
            self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, dilation: int = 1
        ) -> None:
            """Build the signal and gate branches."""
            super().__init__()
            padding = ((kernel - 1) * dilation) // 2
            self.pad = nn.ReflectionPad2d(padding)
            self.conv = ScaledWSConv2d(in_ch, out_ch, kernel, stride=stride, dilation=dilation)
            self.conv_gate = ScaledWSConv2d(in_ch, out_ch, kernel, stride=stride, dilation=dilation)

        def forward(self, x: Any) -> Any:
            """Multiply the signal by its sigmoid gate."""
            padded = self.pad(x)
            return self.conv(padded) * torch.sigmoid(self.conv_gate(padded)) * GATE_SCALE

    class GatedWSTransposeConvPadded(nn.Module):  # type: ignore[misc, name-defined]
        """Gated transposed convolution used by the decoder."""

        def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> None:
            """Build the signal and gate branches."""
            super().__init__()
            padding = (kernel - 1) // 2
            self.conv = ScaledWSTransposeConv2d(
                in_ch, out_ch, kernel, stride=stride, padding=padding
            )
            self.conv_gate = ScaledWSTransposeConv2d(
                in_ch, out_ch, kernel, stride=stride, padding=padding
            )

        def forward(self, x: Any) -> Any:
            """Multiply the signal by its sigmoid gate."""
            return self.conv(x) * torch.sigmoid(self.conv_gate(x)) * GATE_SCALE

    class ReluNf(nn.Module):  # type: ignore[misc, name-defined]
        """ReLU rescaled to preserve variance.

        The encoder and decoder are normaliser-free: there is no batch norm to
        restore the variance a ReLU removes, so the activation itself carries
        the compensating gain. Substituting a plain ``nn.ReLU`` here loads the
        same weights and silently produces a near-constant image, because the
        signal is attenuated once per activation through eight of them.
        """

        def forward(self, x: Any) -> Any:
            """Apply the scaled rectifier."""
            return functional.relu(x) * RELU_NF_GAIN

    def spatial_norm(feat: Any) -> Any:
        """Normalise per-sample over spatial dims, then rescale.

        This is the ``my_layer_norm`` of the original: standardise, map to
        roughly ``[-1, 1]``, then amplify before the sigmoid so the gate
        saturates.
        """
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True) + 1e-9
        return LAYER_NORM_SCALE * (2 * (feat - mean) / std - 1)

    class AOTBlock(nn.Module):  # type: ignore[misc, name-defined]
        """Aggregated contextual transformation block.

        Four parallel dilated branches see the region at different scales; their
        concatenation is fused, and a separately-gated mask decides per pixel
        how much of the transformation to keep.
        """

        def __init__(self, dim: int, rates: tuple[int, ...]) -> None:
            """Build one block."""
            super().__init__()
            self.rates = rates
            # These are plain convolutions, not scaled-WS ones: the checkpoint
            # carries no `gain` for the branch, fuse or gate layers.
            for index, rate in enumerate(rates):
                branch = nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // len(rates), 3, dilation=rate),
                    nn.ReLU(inplace=True),
                )
                self.add_module(f"block{index:02d}", branch)
            self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, dilation=1))
            self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, dilation=1))

        def forward(self, x: Any) -> Any:
            """Aggregate the dilated branches and blend by the learned mask."""
            branches = [getattr(self, f"block{index:02d}")(x) for index in range(len(self.rates))]
            out = self.fuse(torch.cat(branches, dim=1))
            mask = torch.sigmoid(spatial_norm(self.gate(x)))
            return x * (1 - mask) + out * mask

    class AOTGenerator(nn.Module):  # type: ignore[misc, name-defined]
        """The full generator: encode, transform, decode."""

        def __init__(self, base: int = BASE_CHANNELS) -> None:
            """Build the generator at the checkpoint's channel widths."""
            super().__init__()
            self.head = nn.Sequential(
                GatedWSConvPadded(4, base, 3, stride=1),
                ReluNf(),
                GatedWSConvPadded(base, base * 2, 4, stride=2),
                ReluNf(),
                GatedWSConvPadded(base * 2, base * 4, 4, stride=2),
            )
            self.body_conv = nn.Sequential(
                *[AOTBlock(base * 4, DILATION_RATES) for _ in range(NUM_BLOCKS)]
            )
            self.tail = nn.Sequential(
                GatedWSConvPadded(base * 4, base * 4, 3, 1),
                ReluNf(),
                GatedWSConvPadded(base * 4, base * 4, 3, 1),
                ReluNf(),
                GatedWSTransposeConvPadded(base * 4, base * 2, 4, 2),
                ReluNf(),
                GatedWSTransposeConvPadded(base * 2, base, 4, 2),
                ReluNf(),
                GatedWSConvPadded(base, 3, 3, 1),
            )

        def forward(self, img: Any, mask: Any) -> Any:
            """Inpaint ``img`` where ``mask`` is 1.

            Args:
                img: Image in ``[-1, 1]``, shape ``(N, 3, H, W)``.
                mask: Hole mask in ``{0, 1}``, shape ``(N, 1, H, W)``.

            Returns:
                The generated image, clipped to ``[-1, 1]``.
            """
            x = self.head(torch.cat([mask, img], dim=1))
            x = self.body_conv(x)
            return torch.clip(self.tail(x), -1.0, 1.0)

    return {"AOTGenerator": AOTGenerator, "ScaledWSConv2d": ScaledWSConv2d}


def load_aot_inpainter(weights: Path, device: str = "cpu") -> Any:
    """Build the generator and load a checkpoint into it.

    Args:
        weights: Path to a ``.safetensors`` or ``.ckpt``/``.pt`` state dict.
        device: Torch device string.

    Returns:
        The generator in eval mode on ``device``.

    Raises:
        RuntimeError: If the checkpoint does not match the architecture.
    """
    torch = require("torch")
    model = build_modules()["AOTGenerator"]()

    if weights.suffix == ".safetensors":
        safetensors = require("safetensors.torch", extra="ml")
        state = safetensors.load_file(str(weights))
    else:
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = state.get("model", state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        msg = (
            f"checkpoint does not match the AOT-GAN architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected tensors "
            f"(first missing: {list(missing)[:3]}, first unexpected: {list(unexpected)[:3]})"
        )
        raise RuntimeError(msg)

    model.eval()
    model.to(device)
    return model
