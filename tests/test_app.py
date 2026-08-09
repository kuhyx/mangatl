"""Tests for the pipeline, persistence, CLI and HTTP surface."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import (
    FakeDetector,
    FakeInpainter,
    FakeOcr,
    FakeTranslator,
    FakeTypesetter,
)
from mangatl import __version__
from mangatl.cli import _build_parser, _page_size, main, run_batch
from mangatl.config import Settings
from mangatl.domain.models import Box, GlossaryEntry, Page, RegionKind, Stage, TextRegion
from mangatl.pipeline import Pipeline
from mangatl.storage.db import Database
from mangatl.web.app import create_app
from mangatl.web.assembly import build_pipeline, build_translator, pick_font

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mangatl.ports import Detector, Inpainter, Ocr, Translator, Typesetter


def make_pipeline(
    *,
    detector: Detector | None = None,
    ocr: Ocr | None = None,
    translator: Translator | None = None,
    inpainter: Inpainter | None = None,
    typesetter: Typesetter | None = None,
) -> tuple[Pipeline, dict[str, object]]:
    """Build a pipeline of fakes, returning the parts for assertions."""
    parts: dict[str, object] = {
        "detector": detector if detector is not None else FakeDetector(),
        "ocr": ocr if ocr is not None else FakeOcr(),
        "translator": translator if translator is not None else FakeTranslator(),
        "inpainter": inpainter if inpainter is not None else FakeInpainter(),
        "typesetter": typesetter if typesetter is not None else FakeTypesetter(),
    }
    return Pipeline(**parts), parts  # type: ignore[arg-type]


class TestVersion:
    """Package metadata."""

    def test_version_exported(self) -> None:
        """Version exported."""
        assert __version__ == "0.1.0"


class TestSettings:
    """Configuration derived paths."""

    def test_derived_paths(self, tmp_path: Path) -> None:
        """Derived paths."""
        cfg = Settings(data_dir=tmp_path / "d")
        assert cfg.upload_dir == tmp_path / "d" / "uploads"
        assert cfg.render_dir == tmp_path / "d" / "renders"
        assert cfg.db_path == tmp_path / "d" / "mangatl.sqlite3"

    def test_ensure_dirs_is_idempotent(self, tmp_path: Path) -> None:
        """Ensure dirs is idempotent."""
        cfg = Settings(data_dir=tmp_path / "d", font_dir=tmp_path / "f")
        cfg.ensure_dirs()
        cfg.ensure_dirs()
        assert cfg.upload_dir.is_dir()

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env override."""
        monkeypatch.setenv("MANGATL_LLM_MODEL", "gemma-3-12b")
        assert Settings().llm_model == "gemma-3-12b"


class TestPipeline:
    """Orchestration, ordering and failure handling."""

    def test_happy_path(self, page_image: Path, tmp_path: Path) -> None:
        """Happy path."""
        pipeline, parts = make_pipeline()
        page = Page(id="p1", width=400, height=600)
        result = pipeline.run(page_image, page, out_dir=tmp_path)
        assert result.page.stage is Stage.RENDERED
        assert result.rendered_path is not None
        assert result.cleaned_path is not None
        assert result.context == ["EN:こんにちは", "EN:またね"]
        assert [r.target_text for r in result.page.regions] == ["EN:こんにちは", "EN:またね"]
        assert parts["detector"].calls == 1

    def test_regions_get_reading_order(self, page_image: Path, tmp_path: Path) -> None:
        """Regions get reading order."""
        pipeline, _ = make_pipeline()
        page = Page(id="p", width=400, height=600)
        pipeline.run(page_image, page, out_dir=tmp_path)
        assert [r.order for r in page.regions] == [0, 1]

    def test_low_confidence_dropped(self, page_image: Path, tmp_path: Path) -> None:
        """Low confidence dropped."""
        detector = FakeDetector(
            [
                TextRegion(id="keep", box=Box(0, 0, 50, 50), confidence=0.9),
                TextRegion(id="drop", box=Box(200, 200, 250, 250), confidence=0.05),
            ],
        )
        pipeline, _ = make_pipeline(detector=detector)
        page = Page(id="p", width=400, height=600)
        pipeline.run(page_image, page, out_dir=tmp_path)
        assert [r.id for r in page.regions] == ["keep"]

    def test_no_regions_still_renders(self, page_image: Path, tmp_path: Path) -> None:
        """No regions still renders."""
        pipeline, _ = make_pipeline(detector=FakeDetector([]))
        page = Page(id="p", width=400, height=600)
        result = pipeline.run(page_image, page, out_dir=tmp_path)
        assert result.page.stage is Stage.RENDERED
        assert result.context == []

    def test_all_blank_ocr_skips_translation(self, page_image: Path, tmp_path: Path) -> None:
        """All blank ocr skips translation."""
        translator = FakeTranslator()
        pipeline, _ = make_pipeline(ocr=FakeOcr(["", "  "]), translator=translator)
        page = Page(id="p", width=400, height=600)
        result = pipeline.run(page_image, page, out_dir=tmp_path)
        assert result.context == []
        assert translator.seen_context == []

    def test_context_and_glossary_forwarded(self, page_image: Path, tmp_path: Path) -> None:
        """Context and glossary forwarded."""
        translator = FakeTranslator()
        pipeline, _ = make_pipeline(translator=translator)
        entry = GlossaryEntry(source="兄", target="Nii-san")
        pipeline.run(
            page_image,
            Page(id="p", width=400, height=600),
            out_dir=tmp_path,
            context=["earlier"],
            glossary=[entry],
        )
        assert translator.seen_context == ["earlier"]
        assert translator.seen_glossary == [entry]

    def test_translator_length_mismatch_fails_page(self, page_image: Path, tmp_path: Path) -> None:
        """Translator length mismatch fails page."""
        pipeline, _ = make_pipeline(translator=FakeTranslator(drop_one=True))
        page = Page(id="p", width=400, height=600)
        result = pipeline.run(page_image, page, out_dir=tmp_path)
        assert result.page.stage is Stage.FAILED
        assert "returned 1 lines for 2 inputs" in result.page.error
        assert result.rendered_path is None

    def test_ocr_failure_is_captured(self, page_image: Path, tmp_path: Path) -> None:
        """Ocr failure is captured."""

        class Boom:
            def read(self, image_path: Path, box: Box) -> str:
                """Fake read used by the tests."""
                del image_path, box
                msg = "gpu fell over"
                raise RuntimeError(msg)

        pipeline, _ = make_pipeline(ocr=Boom())
        page = Page(id="p", width=400, height=600)
        result = pipeline.run(page_image, page, out_dir=tmp_path)
        assert result.page.stage is Stage.FAILED
        assert "RuntimeError: gpu fell over" in result.page.error

    def test_inpainter_receives_only_translatable_boxes(
        self,
        page_image: Path,
        tmp_path: Path,
    ) -> None:
        """Inpainter receives only translatable boxes."""
        inpainter = FakeInpainter()
        pipeline, _ = make_pipeline(ocr=FakeOcr(["あ", ""]), inpainter=inpainter)
        pipeline.run(page_image, Page(id="p", width=400, height=600), out_dir=tmp_path)
        assert len(inpainter.boxes) == 1


class TestDatabase:
    """SQLite round trips."""

    @pytest.fixture
    def db(self) -> Iterator[Database]:
        """Fake db used by the tests."""
        database = Database(":memory:")
        yield database
        database.close()

    def test_save_and_get(self, db: Database) -> None:
        """Save and get."""
        page = Page(
            id="p1",
            width=100,
            height=200,
            stage=Stage.TRANSLATED,
            regions=[
                TextRegion(
                    id="r0",
                    box=Box(1, 2, 30, 40),
                    kind=RegionKind.SFX,
                    polygon=((0, 0), (5, 0), (5, 5)),
                    confidence=0.75,
                    source_text="ドン",
                    target_text="BOOM",
                    vertical=False,
                    order=3,
                    notes="sfx",
                ),
            ],
        )
        db.save(page)
        loaded = db.get("p1")
        assert loaded is not None
        assert loaded.stage is Stage.TRANSLATED
        region = loaded.regions[0]
        assert region.box == Box(1, 2, 30, 40)
        assert region.kind is RegionKind.SFX
        assert region.polygon == ((0, 0), (5, 0), (5, 5))
        assert region.source_text == "ドン"
        assert region.order == 3
        assert not region.vertical

    def test_get_missing(self, db: Database) -> None:
        """Get missing."""
        assert db.get("nope") is None

    def test_upsert_replaces(self, db: Database) -> None:
        """Upsert replaces."""
        db.save(Page(id="p", width=10, height=10))
        db.save(Page(id="p", width=20, height=30, stage=Stage.RENDERED))
        loaded = db.get("p")
        assert loaded is not None
        assert loaded.width == 20
        assert loaded.stage is Stage.RENDERED

    def test_recent(self, db: Database) -> None:
        """Recent."""
        for i in range(3):
            db.save(Page(id=f"p{i}", width=10, height=10))
        assert len(db.recent()) == 3
        assert len(db.recent(limit=2)) == 2

    def test_delete(self, db: Database) -> None:
        """Delete."""
        db.save(Page(id="p", width=10, height=10))
        assert db.delete("p")
        assert not db.delete("p")

    def test_file_backed(self, tmp_path: Path) -> None:
        """File backed."""
        path = tmp_path / "db.sqlite3"
        first = Database(path)
        first.save(Page(id="p", width=5, height=5))
        first.close()
        second = Database(path)
        assert second.get("p") is not None
        second.close()


class TestAssembly:
    """Composition root."""

    def test_pick_font_prefers_comic_neue(self, tmp_path: Path) -> None:
        """Pick font prefers comic neue."""
        (tmp_path / "Bangers-Regular.ttf").write_bytes(b"x")
        (tmp_path / "ComicNeue-Bold.ttf").write_bytes(b"x")
        assert pick_font(tmp_path).name == "ComicNeue-Bold.ttf"

    def test_pick_font_falls_back_to_any_ttf(self, tmp_path: Path) -> None:
        """Pick font falls back to any ttf."""
        (tmp_path / "Whatever.ttf").write_bytes(b"x")
        assert pick_font(tmp_path).name == "Whatever.ttf"

    def test_pick_font_raises_when_empty(self, tmp_path: Path) -> None:
        """Pick font raises when empty."""
        with pytest.raises(FileNotFoundError, match="fetch-fonts"):
            pick_font(tmp_path)

    def test_build_translator_prefers_llm(self, settings: Settings) -> None:
        """Build translator prefers llm."""
        assert build_translator(settings).name == "local-llm"

    def test_build_translator_falls_back(self, settings: Settings) -> None:
        """Build translator falls back."""
        settings.llm_base_url = "  "
        assert build_translator(settings).name == "libretranslate"

    def test_build_pipeline(self, settings: Settings, font_file: Path) -> None:
        """Build pipeline."""
        settings.font_dir = font_file.parent
        assert isinstance(build_pipeline(settings), Pipeline)


class TestCli:
    """Command-line surface."""

    def test_parser_requires_command(self) -> None:
        """Parser requires command."""
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_page_size(self, page_image: Path) -> None:
        """Page size."""
        assert _page_size(page_image) == (400, 600)

    def test_batch_processes_in_order(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Batch processes in order."""
        pages = tmp_path / "chapter"
        pages.mkdir()
        for name in ("002.png", "001.png"):
            Image.new("RGB", (400, 600), "white").save(pages / name)
        (pages / "notes.txt").write_text("ignored", encoding="utf-8")
        pipeline, parts = make_pipeline()
        monkeypatch.setattr("mangatl.cli.build_pipeline", lambda _cfg: pipeline)
        assert run_batch(pages, settings, tmp_path / "out") == 0
        assert parts["detector"].calls == 2

    def test_batch_uses_default_out_dir(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Batch uses default out dir."""
        pages = tmp_path / "chapter"
        pages.mkdir()
        Image.new("RGB", (400, 600), "white").save(pages / "001.png")
        pipeline, _ = make_pipeline()
        monkeypatch.setattr("mangatl.cli.build_pipeline", lambda _cfg: pipeline)
        assert run_batch(pages, settings) == 0
        assert (settings.render_dir / "001.render.png").exists()

    def test_batch_reports_failures(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Batch reports failures."""
        pages = tmp_path / "chapter"
        pages.mkdir()
        Image.new("RGB", (400, 600), "white").save(pages / "001.png")
        pipeline, _ = make_pipeline(translator=FakeTranslator(drop_one=True))
        monkeypatch.setattr("mangatl.cli.build_pipeline", lambda _cfg: pipeline)
        assert run_batch(pages, settings, tmp_path / "out") == 1

    def test_main_batch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Main batch."""
        pages = tmp_path / "chapter"
        pages.mkdir()
        monkeypatch.setenv("MANGATL_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MANGATL_FONT_DIR", str(tmp_path / "fonts"))
        pipeline, _ = make_pipeline()
        monkeypatch.setattr("mangatl.cli.build_pipeline", lambda _cfg: pipeline)
        assert main(["batch", str(pages)]) == 0

    def test_main_serve(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Main serve."""
        started: list[tuple[str, int]] = []

        module = types.ModuleType("uvicorn")

        def run(app: object, *, host: str, port: int) -> None:
            """Fake run used by the tests."""
            del app
            started.append((host, port))

        module.run = run  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "uvicorn", module)
        monkeypatch.setenv("MANGATL_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MANGATL_FONT_DIR", str(tmp_path / "fonts"))
        pipeline, _ = make_pipeline()
        monkeypatch.setattr("mangatl.web.app.build_pipeline", lambda _cfg: pipeline)
        assert main(["serve", "--host", "0.0.0.0", "--port", "9999"]) == 0  # noqa: S104
        assert started == [("0.0.0.0", 9999)]  # noqa: S104


class TestWebApp:
    """HTTP surface."""

    @pytest.fixture
    def client(self, settings: Settings) -> Iterator[TestClient]:
        """Fake client used by the tests."""
        pipeline, _ = make_pipeline()
        db = Database(":memory:")
        app = create_app(settings, database=db, pipeline=pipeline)
        with TestClient(app) as test_client:
            yield test_client
        db.close()

    @pytest.fixture
    def png_bytes(self, tmp_path: Path) -> bytes:
        """Fake png bytes used by the tests."""
        path = tmp_path / "upload.png"
        Image.new("RGB", (400, 600), "white").save(path)
        return path.read_bytes()

    def test_index(self, client: TestClient) -> None:
        """Index."""
        response = client.get("/")
        assert response.status_code == 200
        assert "mangatl" in response.text

    def test_health(self, client: TestClient, settings: Settings) -> None:
        """Health."""
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["llm"] == settings.llm_base_url

    def test_upload_and_fetch(self, client: TestClient, png_bytes: bytes) -> None:
        """Upload and fetch."""
        response = client.post("/api/pages", files={"file": ("p.png", png_bytes, "image/png")})
        assert response.status_code == 201
        body = response.json()
        assert body["stage"] == "rendered"
        assert [r["target_text"] for r in body["regions"]] == ["EN:こんにちは", "EN:またね"]

        fetched = client.get(f"/api/pages/{body['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == body["id"]

        render = client.get(f"/api/pages/{body['id']}/render")
        assert render.status_code == 200
        assert render.headers["content-type"] == "image/png"

    def test_upload_rejects_non_image(self, client: TestClient) -> None:
        """Upload rejects non image."""
        response = client.post(
            "/api/pages", files={"file": ("x.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_upload_rejects_oversized(self, client: TestClient, settings: Settings) -> None:
        """Upload rejects oversized."""
        settings.max_upload_bytes = 8
        response = client.post("/api/pages", files={"file": ("p.png", b"x" * 64, "image/png")})
        assert response.status_code == 413

    def test_upload_surfaces_pipeline_failure(self, settings: Settings, png_bytes: bytes) -> None:
        """Upload surfaces pipeline failure."""
        pipeline, _ = make_pipeline(translator=FakeTranslator(drop_one=True))
        db = Database(":memory:")
        with TestClient(create_app(settings, database=db, pipeline=pipeline)) as client:
            response = client.post("/api/pages", files={"file": ("p.png", png_bytes, "image/png")})
        db.close()
        assert response.status_code == 500

    def test_list_pages(self, client: TestClient, png_bytes: bytes) -> None:
        """List pages."""
        client.post("/api/pages", files={"file": ("p.png", png_bytes, "image/png")})
        assert len(client.get("/api/pages").json()) == 1

    def test_page_not_found(self, client: TestClient) -> None:
        """Page not found."""
        assert client.get("/api/pages/missing").status_code == 404

    def test_render_not_found(self, client: TestClient) -> None:
        """Render not found."""
        assert client.get("/api/pages/missing/render").status_code == 404

    def test_delete(self, client: TestClient, png_bytes: bytes) -> None:
        """Delete."""
        page_id = client.post(
            "/api/pages",
            files={"file": ("p.png", png_bytes, "image/png")},
        ).json()["id"]
        assert client.delete(f"/api/pages/{page_id}").status_code == 204
        assert client.delete(f"/api/pages/{page_id}").status_code == 404

    def test_glossary_round_trip(self, client: TestClient) -> None:
        """Glossary round trip."""
        assert client.get("/api/glossary").json() == []
        put = client.put("/api/glossary", json=[{"source": "兄", "target": "Nii-san"}])
        assert put.json() == {"count": 1}
        assert client.get("/api/glossary").json()[0]["target"] == "Nii-san"

    def test_glossary_rejects_malformed(self, client: TestClient) -> None:
        """Glossary rejects malformed."""
        assert client.put("/api/glossary", json=[{"target": "x"}]).status_code == 422

    def test_glossary_rejects_non_object_entries(self, client: TestClient) -> None:
        """Glossary rejects non object entries."""
        assert client.put("/api/glossary", json=[1, 2]).status_code == 422

    def test_create_app_builds_its_own_deps(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create app builds its own deps."""
        pipeline, _ = make_pipeline()
        monkeypatch.setattr("mangatl.web.app.build_pipeline", lambda _cfg: pipeline)
        with TestClient(create_app(settings)) as client:
            assert client.get("/api/health").status_code == 200

    def test_create_app_reads_env_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create app reads env settings."""
        monkeypatch.setenv("MANGATL_DATA_DIR", str(tmp_path / "d"))
        monkeypatch.setenv("MANGATL_FONT_DIR", str(tmp_path / "f"))
        pipeline, _ = make_pipeline()
        monkeypatch.setattr("mangatl.web.app.build_pipeline", lambda _cfg: pipeline)
        with TestClient(create_app()) as client:
            assert client.get("/api/health").status_code == 200
