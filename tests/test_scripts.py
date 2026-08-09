"""Tests for the shipped helper scripts under ``scripts/``.

These are held to the same 100% branch bar as the package. ``huggingface_hub``
is not installed in CI, so every test injects a fake into ``sys.modules`` in the
same style :mod:`tests.test_adapters` uses for the ML adapters.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import benchmark
import fetch_models
import pytest

from mangatl.adapters.translate_libre import LibreTranslateTranslator
from mangatl.adapters.translate_llm import LlmTranslator
from mangatl.quality.glossary import Glossary

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mangatl.config import Settings
    from mangatl.domain.models import GlossaryEntry


class _StubTranslator:
    """Returns pre-scripted page translations."""

    def __init__(self, pages: list[list[str]]) -> None:
        """Build the fixture."""
        self._pages = list(pages)

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> list[str]:
        """Pop the next scripted page."""
        del lines, context, glossary
        return self._pages.pop(0)


class _ExplodingTranslator:
    """Fails every page, to exercise the skip-and-continue path."""

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> list[str]:
        """Always fail."""
        del lines, context, glossary
        msg = "backend exploded"
        raise RuntimeError(msg)


class FakeHub:
    """A stand-in for ``huggingface_hub`` that records what was requested."""

    def __init__(self, cache: Path) -> None:
        """Build the fixture."""
        self.calls: list[tuple[str, ...]] = []
        self._cache = cache

    def hf_hub_download(self, repo: str, filename: str) -> str:
        """Pretend to fetch one file, returning a real path on disk."""
        self.calls.append(("download", repo, filename))
        blob = self._cache / filename
        blob.write_bytes(b"weights")
        return str(blob)

    def snapshot_download(self, repo: str) -> str:
        """Pretend to fetch a whole repo."""
        self.calls.append(("snapshot", repo))
        return str(self._cache)

    def as_module(self) -> types.ModuleType:
        """Expose the fake with a module's shape, for ``sys.modules``."""
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = self.hf_hub_download  # type: ignore[attr-defined]
        module.snapshot_download = self.snapshot_download  # type: ignore[attr-defined]
        return module


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHub:
    """Install a fake ``huggingface_hub`` and return it.

    Args:
        monkeypatch: Pytest patcher.
        tmp_path: Temporary directory backing the fake downloads.

    Returns:
        The fake, with ``calls`` recording what was requested.
    """
    cache = tmp_path / "cached"
    cache.mkdir()
    fake = FakeHub(cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake.as_module())
    return fake


class TestFetchOcr:
    """The manga-ocr cache warm-up."""

    def test_snapshots_the_repo(self, hub: FakeHub) -> None:
        """It asks the Hub for the whole manga-ocr repo."""
        fetch_models.fetch_ocr()
        assert hub.calls == [("snapshot", "kha-white/manga-ocr-base")]


class TestFetchDetector:
    """The bubble-segmentation weights."""

    def test_downloads_and_places_the_file(self, hub: FakeHub, tmp_path: Path) -> None:
        """The checkpoint lands at ``bubble-seg.pt``."""
        models = tmp_path / "models"
        models.mkdir()
        fetch_models.fetch_detector(models)
        assert (models / "bubble-seg.pt").read_bytes() == b"weights"
        assert hub.calls == [("download", "kitsumed/yolov8m_seg-speech-bubble", "model.pt")]

    def test_is_idempotent(self, hub: FakeHub, tmp_path: Path) -> None:
        """An existing checkpoint is left alone and nothing is downloaded."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "bubble-seg.pt").write_bytes(b"already here")
        fetch_models.fetch_detector(models)
        assert (models / "bubble-seg.pt").read_bytes() == b"already here"
        assert hub.calls == []


class TestFetchInpainter:
    """The AOT-GAN inpainting weights."""

    def test_downloads_and_places_the_file(self, hub: FakeHub, tmp_path: Path) -> None:
        """The checkpoint lands under the name the config expects."""
        models = tmp_path / "models"
        models.mkdir()
        fetch_models.fetch_inpainter(models)
        assert (models / "aot-inpainting.safetensors").read_bytes() == b"weights"
        assert hub.calls == [("download", "mayocream/aot-inpainting", "model.safetensors")]

    def test_is_idempotent(self, hub: FakeHub, tmp_path: Path) -> None:
        """Existing weights are not re-downloaded."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "aot-inpainting.safetensors").write_bytes(b"already here")
        fetch_models.fetch_inpainter(models)
        assert hub.calls == []


class TestFetchGguf:
    """The GGUF translation checkpoint."""

    def test_downloads_and_places_the_file(self, hub: FakeHub, tmp_path: Path) -> None:
        """The GGUF lands under its own name."""
        models = tmp_path / "models"
        models.mkdir()
        fetch_models.fetch_gguf(models, "Qwen/Qwen3-14B-GGUF", "Q5.gguf")
        assert (models / "Q5.gguf").read_bytes() == b"weights"
        assert hub.calls == [("download", "Qwen/Qwen3-14B-GGUF", "Q5.gguf")]

    def test_is_idempotent(self, hub: FakeHub, tmp_path: Path) -> None:
        """An existing GGUF is not re-downloaded."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "Q5.gguf").write_bytes(b"already here")
        fetch_models.fetch_gguf(models, "Qwen/Qwen3-14B-GGUF", "Q5.gguf")
        assert hub.calls == []


class TestMain:
    """Argument dispatch."""

    def test_ocr_subcommand(self, hub: FakeHub) -> None:
        """``ocr`` warms the cache."""
        assert fetch_models.main(["ocr"]) == 0
        assert hub.calls == [("snapshot", "kha-white/manga-ocr-base")]

    def test_detector_subcommand(self, hub: FakeHub, tmp_path: Path) -> None:
        """``detector`` writes the checkpoint."""
        assert fetch_models.main(["detector", str(tmp_path)]) == 0
        assert (tmp_path / "bubble-seg.pt").exists()
        assert hub.calls == [("download", "kitsumed/yolov8m_seg-speech-bubble", "model.pt")]

    def test_gguf_subcommand(self, hub: FakeHub, tmp_path: Path) -> None:
        """``gguf`` writes the named file."""
        assert fetch_models.main(["gguf", str(tmp_path), "repo/id", "m.gguf"]) == 0
        assert (tmp_path / "m.gguf").exists()
        assert hub.calls == [("download", "repo/id", "m.gguf")]

    def test_inpainter_subcommand(self, hub: FakeHub, tmp_path: Path) -> None:
        """``inpainter`` writes the AOT weights."""
        assert fetch_models.main(["inpainter", str(tmp_path)]) == 0
        assert (tmp_path / "aot-inpainting.safetensors").exists()
        assert hub.calls == [("download", "mayocream/aot-inpainting", "model.safetensors")]

    def test_requires_a_subcommand(self) -> None:
        """No subcommand is a usage error, not a traceback."""
        with pytest.raises(SystemExit):
            fetch_models.main([])


class TestLoadPages:
    """Sample-file parsing for the benchmark harness."""

    def test_blank_lines_separate_pages(self, tmp_path: Path) -> None:
        """A blank line ends a page; trailing content still counts."""
        path = tmp_path / "samples.txt"
        path.write_text("あ\nい\n\nう\n", encoding="utf-8")
        assert benchmark.load_pages(path) == [["あ", "い"], ["う"]]

    def test_repeated_blanks_do_not_make_empty_pages(self, tmp_path: Path) -> None:
        """Consecutive blank lines collapse rather than emitting empty pages."""
        path = tmp_path / "samples.txt"
        path.write_text("\n\nあ\n\n\n\nい\n\n", encoding="utf-8")
        assert benchmark.load_pages(path) == [["あ"], ["い"]]

    def test_empty_file_has_no_pages(self, tmp_path: Path) -> None:
        """An empty file yields nothing."""
        path = tmp_path / "samples.txt"
        path.write_text("", encoding="utf-8")
        assert benchmark.load_pages(path) == []


class TestMakeTranslator:
    """Backend selection by short name."""

    def test_libretranslate(self, settings: Settings) -> None:
        """The libretranslate name builds the Libre adapter."""
        assert isinstance(
            benchmark.make_translator("libretranslate", settings), LibreTranslateTranslator
        )

    def test_local_llm(self, settings: Settings) -> None:
        """The local-llm name builds the LLM adapter."""
        assert isinstance(benchmark.make_translator("local-llm", settings), LlmTranslator)

    def test_unknown_name_exits(self, settings: Settings) -> None:
        """An unknown backend is a clean exit, not a traceback."""
        with pytest.raises(SystemExit, match="unknown backend: nope"):
            benchmark.make_translator("nope", settings)


class TestBackTranslate:
    """Round-trip scoring helper."""

    def test_blank_input_short_circuits(self, settings: Settings) -> None:
        """Whitespace never reaches the network."""
        assert benchmark.back_translate("   ", settings) == ""

    def test_returns_the_parsed_line(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A well-formed reply yields the Japanese line."""
        monkeypatch.setattr(
            benchmark.HttpChatBackend, "chat", lambda *_: '{"lines": ["こんにちは"]}'
        )
        assert benchmark.back_translate("Hello", settings) == "こんにちは"

    def test_bad_reply_scores_as_empty(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed reply degrades to an empty string, not a crash."""

        def _boom(*_: object) -> str:
            msg = "server down"
            raise RuntimeError(msg)

        monkeypatch.setattr(benchmark.HttpChatBackend, "chat", _boom)
        assert benchmark.back_translate("Hello", settings) == ""


class TestEvaluate:
    """The per-page scoring loop."""

    def test_scores_every_line(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each source line produces one scored row."""
        monkeypatch.setattr(
            benchmark, "make_translator", lambda *_: _StubTranslator([["A", "B"], ["C"]])
        )
        monkeypatch.setattr(benchmark, "back_translate", lambda *_: "あ")
        rows = benchmark.evaluate("stub", settings, [["あ", "い"], ["う"]], Glossary())
        assert [r["target"] for r in rows] == ["A", "B", "C"]
        assert {r["backend"] for r in rows} == {"stub"}

    def test_failed_page_is_skipped_not_fatal(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One bad page does not abort the whole benchmark."""
        monkeypatch.setattr(benchmark, "make_translator", lambda *_: _ExplodingTranslator())
        monkeypatch.setattr(benchmark, "back_translate", lambda *_: "あ")
        assert benchmark.evaluate("stub", settings, [["あ"]], Glossary()) == []


class TestSummarise:
    """Aggregation across rows."""

    def test_no_rows_gives_zeroes(self) -> None:
        """A backend that produced nothing scores zero rather than dividing by zero."""
        assert benchmark.summarise([], "absent") == {
            "round_trip": 0.0,
            "length_sanity": 0.0,
            "glossary": 0.0,
            "seconds": 0.0,
        }

    def test_averages_only_its_own_backend(self) -> None:
        """Rows belonging to another backend are excluded."""
        rows: list[dict[str, object]] = [
            {
                "backend": "a",
                "page": 0,
                "round_trip": 1.0,
                "length_sanity": 0.5,
                "glossary_ok": True,
                "seconds_per_page": 2.0,
            },
            {
                "backend": "b",
                "page": 0,
                "round_trip": 0.0,
                "length_sanity": 0.0,
                "glossary_ok": False,
                "seconds_per_page": 9.0,
            },
        ]
        assert benchmark.summarise(rows, "a") == {
            "round_trip": 1.0,
            "length_sanity": 0.5,
            "glossary": 1.0,
            "seconds": 2.0,
        }


class TestBenchmarkMain:
    """End-to-end argument handling for the harness."""

    def test_empty_sample_file_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No pages is exit status 1 with a message."""
        sample = tmp_path / "s.txt"
        sample.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--lines", str(sample)])
        assert benchmark.main() == 1

    def test_writes_csv_and_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A full run writes one CSV row per line and prints both backends."""
        sample = tmp_path / "s.txt"
        sample.write_text("あ\n", encoding="utf-8")
        out = tmp_path / "b.csv"
        monkeypatch.setattr(
            benchmark, "make_translator", lambda *_: _StubTranslator([["A"], ["A"]])
        )
        monkeypatch.setattr(benchmark, "back_translate", lambda *_: "あ")
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark.py", "--lines", str(sample), "--csv", str(out), "--model-a", "m1"],
        )
        assert benchmark.main() == 0
        assert "backend,page,source,target" in out.read_text(encoding="utf-8")

    def test_glossary_file_is_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A --glossary path is read rather than ignored."""
        sample = tmp_path / "s.txt"
        sample.write_text("あ\n", encoding="utf-8")
        gloss = tmp_path / "g.json"
        gloss.write_text('[{"source": "あ", "target": "A"}]', encoding="utf-8")
        seen: list[Glossary] = []

        def _capture(_n: str, _s: Settings, _p: object, glossary: Glossary) -> list[Any]:
            seen.append(glossary)
            return [
                {
                    "backend": _n,
                    "page": 0,
                    "source": "あ",
                    "target": "A",
                    "round_trip": 1.0,
                    "length_sanity": 1.0,
                    "glossary_ok": True,
                    "seconds_per_page": 0.1,
                }
            ]

        monkeypatch.setattr(benchmark, "evaluate", _capture)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark.py",
                "--lines",
                str(sample),
                "--glossary",
                str(gloss),
                "--csv",
                str(tmp_path / "o.csv"),
            ],
        )
        assert benchmark.main() == 0
        assert [len(g) for g in seen] == [1, 1]
