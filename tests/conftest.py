"""Shared fixtures and in-memory fakes for every pipeline port."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from mangatl.config import Settings
from mangatl.domain.models import Box, GlossaryEntry, Page, TextRegion

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted entirely inside a temporary directory."""
    cfg = Settings(
        data_dir=tmp_path / "data",
        font_dir=tmp_path / "fonts",
        detector_weights=tmp_path / "models" / "bubble.pt",
        inpaint_weights=tmp_path / "models" / "aot.safetensors",
        device="cpu",
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def page_image(tmp_path: Path) -> Path:
    """A small white page image with two dark blobs standing in for text."""
    img = Image.new("RGB", (400, 600), "white")
    for box in ((40, 40, 160, 140), (240, 300, 360, 420)):
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                img.putpixel((x, y), (20, 20, 20))
    path = tmp_path / "page.png"
    img.save(path)
    return path


@pytest.fixture
def font_file(tmp_path: Path) -> Path:
    """A real TrueType file, borrowed from Pillow's bundled test font."""
    search = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
    candidates = [f for root in search if root.is_dir() for f in sorted(root.rglob("*.ttf"))]
    if not candidates:
        pytest.skip("no TrueType font available in this environment")
    target = tmp_path / "fonts" / "ComicNeue-Bold.ttf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(candidates[0].read_bytes())
    return target


class FakeDetector:
    """Returns a fixed set of regions, optionally with a low-confidence one."""

    def __init__(self, regions: list[TextRegion] | None = None) -> None:
        """Build the fixture."""
        self.regions = regions if regions is not None else _default_regions()
        self.calls = 0

    def detect(self, image_path: Path, page: Page) -> list[TextRegion]:
        """Fake detect used by the tests."""
        del image_path, page
        self.calls += 1
        return list(self.regions)


class FakeOcr:
    """Maps each region box to a canned Japanese string."""

    def __init__(self, texts: list[str] | None = None) -> None:
        """Build the fixture."""
        self.texts = texts if texts is not None else ["こんにちは", "またね"]
        self.index = 0

    def read(self, image_path: Path, box: Box) -> str:
        """Fake read used by the tests."""
        del image_path, box
        value = self.texts[self.index % len(self.texts)]
        self.index += 1
        return value


class FakeTranslator:
    """Echoes an English marker per line, recording what it was given."""

    name = "fake"

    def __init__(self, *, drop_one: bool = False) -> None:
        """Build the fixture."""
        self.drop_one = drop_one
        self.seen_context: list[str] = []
        self.seen_glossary: list[GlossaryEntry] = []

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> list[str]:
        """Fake translate page used by the tests."""
        self.seen_context = list(context)
        self.seen_glossary = list(glossary)
        out = [f"EN:{line}" for line in lines]
        return out[:-1] if self.drop_one else out


class FakeInpainter:
    """Copies the source image through, recording the regions it was asked for."""

    def __init__(self) -> None:
        """Build the fixture."""
        self.regions: list[TextRegion] = []

    @property
    def boxes(self) -> list[Box]:
        """The boxes of the recorded regions."""
        return [region.box for region in self.regions]

    def erase(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Fake erase used by the tests."""
        self.regions = list(regions)
        out_path.write_bytes(image_path.read_bytes())
        return out_path


class FakeTypesetter:
    """Copies the cleaned image through, recording the regions."""

    def __init__(self) -> None:
        """Build the fixture."""
        self.regions: list[TextRegion] = []

    def render(self, image_path: Path, regions: Sequence[TextRegion], out_path: Path) -> Path:
        """Fake render used by the tests."""
        self.regions = list(regions)
        out_path.write_bytes(image_path.read_bytes())
        return out_path


class ScriptedBackend:
    """Chat backend replaying canned replies in order."""

    def __init__(self, replies: list[str]) -> None:
        """Build the fixture."""
        self.replies = replies
        self.prompts: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        """Fake chat used by the tests."""
        self.prompts.append((system, user))
        if not self.replies:
            return json.dumps({"lines": [""]})
        return self.replies.pop(0)


class FakeResponse:
    """Minimal stand-in for an ``httpx.Response``."""

    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        """Build the fixture."""
        self.payload = payload
        self.status = status

    def raise_for_status(self) -> None:
        """Fake raise for status used by the tests."""
        if self.status >= 400:
            msg = f"HTTP {self.status}"
            raise RuntimeError(msg)

    def json(self) -> dict[str, Any]:
        """Fake json used by the tests."""
        return self.payload


class FakeHttpClient:
    """Minimal stand-in for an ``httpx.Client`` that replays responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        """Build the fixture."""
        self.responses = responses
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        """Fake post used by the tests."""
        self.requests.append((url, json))
        return self.responses.pop(0)


def _default_regions() -> list[TextRegion]:
    """Two well-separated regions, right one first in manga reading order."""
    return [
        TextRegion(id="r000", box=Box(40, 40, 160, 140), confidence=0.9),
        TextRegion(id="r001", box=Box(240, 300, 360, 420), confidence=0.8),
    ]


def lines_reply(lines: list[str]) -> str:
    """Build a well-formed model reply for ``lines``."""
    return json.dumps({"lines": lines}, ensure_ascii=False)
