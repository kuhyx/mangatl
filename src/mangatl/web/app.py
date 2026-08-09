"""FastAPI application: drop in a manga page, get an English one back.

Every route is synchronous work behind an async handler because the pipeline
is GPU-bound and there is exactly one GPU. Queuing beyond that would only add
latency without adding throughput.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from mangatl.config import Settings
from mangatl.domain.models import Page
from mangatl.quality.glossary import Glossary
from mangatl.storage.db import Database
from mangatl.web.assembly import build_pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from mangatl.pipeline import Pipeline

HTTP_CREATED = 201
HTTP_NO_CONTENT = 204
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_PAYLOAD_TOO_LARGE = 413
HTTP_UNPROCESSABLE = 422
HTTP_SERVER_ERROR = 500

INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>mangatl</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;
      background:#111;color:#eee}
 h1{font-size:1.4rem} a{color:#8cf} .row{display:flex;gap:1rem;flex-wrap:wrap}
 img{max-width:100%;border:1px solid #333;border-radius:4px}
 button{padding:.6rem 1rem;font-size:1rem;cursor:pointer}
 #out{margin-top:1.5rem} .line{padding:.3rem 0;border-bottom:1px solid #222}
 .ja{color:#888;font-size:.9rem}
</style></head><body>
<h1>mangatl &mdash; local Japanese &rarr; English manga translation</h1>
<p>Everything runs on this machine. Nothing leaves it.</p>
<input type="file" id="file" accept="image/*">
<button id="go">Translate page</button>
<div id="out"></div>
<script>
document.getElementById('go').onclick = async () => {
  const f = document.getElementById('file').files[0];
  const out = document.getElementById('out');
  if (!f) { out.textContent = 'Pick an image first.'; return; }
  out.textContent = 'Working. A dense page takes a while on the critique pass...';
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/api/pages', {method: 'POST', body: fd});
  if (!r.ok) { out.textContent = 'Failed: ' + await r.text(); return; }
  const d = await r.json();
  out.innerHTML = '<div class="row"><img src="/api/pages/' + d.id + '/render"></div>' +
    d.regions.map(x => '<div class="line"><div class="ja">' + x.source_text +
    '</div><div>' + x.target_text + '</div></div>').join('');
};
</script></body></html>
"""


def get_settings() -> Settings:
    """Provide application settings. Overridden in tests."""
    return Settings()


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    pipeline: Pipeline | None = None,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Runtime configuration; read from the environment if absent.
        database: Pre-built database, mainly for tests.
        pipeline: Pre-built pipeline, mainly for tests.

    Returns:
        The configured application.
    """
    config = settings if settings is not None else get_settings()
    config.ensure_dirs()
    db = database if database is not None else Database(config.db_path)
    engine = pipeline if pipeline is not None else build_pipeline(config)
    glossary_path = config.data_dir / "glossary.json"

    app = FastAPI(title="mangatl", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the single-page UI."""
        return INDEX_HTML

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Report which local services this instance is pointed at."""
        return {
            "status": "ok",
            "llm": config.llm_base_url,
            "model": config.llm_model,
            "device": config.device,
            "critique_pass": config.critique_pass,
        }

    _register_glossary_routes(app, glossary_path)
    _register_page_routes(app, config, db, engine, glossary_path)
    return app


def _register_glossary_routes(app: FastAPI, glossary_path: Path) -> None:
    """Attach the glossary read/replace endpoints."""

    @app.get("/api/glossary")
    async def read_glossary() -> list[dict[str, str]]:
        """Return the current series glossary."""
        return [asdict(e) for e in Glossary.load(glossary_path).entries]

    @app.put("/api/glossary")
    async def write_glossary(entries: list[dict[str, str]]) -> dict[str, int]:
        """Replace the series glossary.

        Args:
            entries: Objects with ``source``, ``target`` and optional ``note``.

        Returns:
            The number of terms stored.

        Raises:
            HTTPException: If any entry is malformed.
        """
        try:
            glossary = Glossary.from_json(_dump(entries))
        except (TypeError, ValueError) as exc:
            raise HTTPException(HTTP_UNPROCESSABLE, str(exc)) from exc
        glossary.save(glossary_path)
        return {"count": len(glossary)}


def _register_page_routes(
    app: FastAPI,
    config: Settings,
    db: Database,
    engine: Pipeline,
    glossary_path: Path,
) -> None:
    """Attach the page upload, read, render and delete endpoints."""

    @app.post("/api/pages", status_code=HTTP_CREATED)
    async def create_page(file: UploadFile) -> dict[str, Any]:
        """Accept a page image and run the whole pipeline over it.

        Args:
            file: Uploaded image.

        Returns:
            The page id, stage, and every region with its source and target
            text.

        Raises:
            HTTPException: If the upload is too large or not a readable image,
                or if the pipeline failed.
        """
        payload = await file.read()
        if len(payload) > config.max_upload_bytes:
            raise HTTPException(HTTP_PAYLOAD_TOO_LARGE, "file too large")
        page_id = secrets.token_hex(8)
        stored = config.upload_dir / f"{page_id}.png"
        stored.write_bytes(payload)
        size = _image_size(stored)
        if size is None:
            stored.unlink(missing_ok=True)
            raise HTTPException(HTTP_BAD_REQUEST, "not a readable image")
        page = Page(id=page_id, width=size[0], height=size[1])
        result = engine.run(
            stored,
            page,
            out_dir=config.render_dir,
            glossary=Glossary.load(glossary_path).entries,
        )
        db.save(result.page)
        if result.rendered_path is None:
            raise HTTPException(HTTP_SERVER_ERROR, result.page.error)
        return _page_payload(result.page)

    _register_page_read_routes(app, config, db)


def _register_page_read_routes(app: FastAPI, config: Settings, db: Database) -> None:
    """Attach the read-only page endpoints."""

    @app.get("/api/pages")
    async def list_pages() -> list[dict[str, Any]]:
        """List recently processed pages, newest first."""
        return [_page_payload(p) for p in db.recent()]

    @app.get("/api/pages/{page_id}")
    async def read_page(page_id: str) -> dict[str, Any]:
        """Fetch one page.

        Args:
            page_id: Page identifier.

        Returns:
            The page payload.

        Raises:
            HTTPException: If no such page exists.
        """
        page = db.get(page_id)
        if page is None:
            raise HTTPException(HTTP_NOT_FOUND, "no such page")
        return _page_payload(page)

    @app.get("/api/pages/{page_id}/render")
    async def read_render(page_id: str) -> FileResponse:
        """Serve the finished typeset image.

        Args:
            page_id: Page identifier.

        Returns:
            The rendered PNG.

        Raises:
            HTTPException: If the render is missing.
        """
        path = config.render_dir / f"{page_id}.render.png"
        if not path.exists():
            raise HTTPException(HTTP_NOT_FOUND, "no render for that page")
        return FileResponse(path, media_type="image/png")

    @app.delete("/api/pages/{page_id}", status_code=HTTP_NO_CONTENT)
    async def delete_page(page_id: str) -> None:
        """Delete a page record.

        Args:
            page_id: Page identifier.

        Raises:
            HTTPException: If no such page exists.
        """
        if not db.delete(page_id):
            raise HTTPException(HTTP_NOT_FOUND, "no such page")


def _dump(entries: list[dict[str, str]]) -> str:
    """Serialise glossary entries back to JSON for the loader."""
    return json.dumps(entries, ensure_ascii=False)


def _image_size(path: Path) -> tuple[int, int] | None:
    """Return image dimensions, or ``None`` when the file is not an image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            return (img.width, img.height)
    except (UnidentifiedImageError, OSError):
        return None


def _page_payload(page: Page) -> dict[str, Any]:
    """Shape a page for the JSON API."""
    return {
        "id": page.id,
        "width": page.width,
        "height": page.height,
        "stage": str(page.stage),
        "error": page.error,
        "regions": [
            {
                "id": r.id,
                "box": [r.box.x1, r.box.y1, r.box.x2, r.box.y2],
                "order": r.order,
                "confidence": r.confidence,
                "source_text": r.source_text,
                "target_text": r.target_text,
            }
            for r in page.regions
        ],
    }
