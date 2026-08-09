"""SQLite persistence.

Deliberately stdlib-only: no ORM, no migration framework, no Redis. This runs
on one workstation, and a single file you can copy is worth more than a
service you have to keep alive.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from mangatl.domain.models import Box, Page, RegionKind, Stage, TextRegion

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id      TEXT PRIMARY KEY,
    width   INTEGER NOT NULL,
    height  INTEGER NOT NULL,
    stage   TEXT NOT NULL,
    error   TEXT NOT NULL DEFAULT '',
    regions TEXT NOT NULL DEFAULT '[]',
    created REAL NOT NULL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS pages_created ON pages (created DESC);
"""


def _region_to_dict(region: TextRegion) -> dict[str, object]:
    """Flatten a region for JSON storage."""
    return {
        "id": region.id,
        "box": [region.box.x1, region.box.y1, region.box.x2, region.box.y2],
        "kind": str(region.kind),
        "polygon": [list(p) for p in region.polygon],
        "confidence": region.confidence,
        "source_text": region.source_text,
        "target_text": region.target_text,
        "vertical": region.vertical,
        "order": region.order,
        "notes": region.notes,
    }


def _region_from_dict(data: dict[str, object]) -> TextRegion:
    """Rebuild a region from its stored form."""
    box = [int(v) for v in list(data["box"])]  # type: ignore[call-overload]
    polygon = tuple((int(p[0]), int(p[1])) for p in list(data["polygon"]))  # type: ignore[call-overload]
    return TextRegion(
        id=str(data["id"]),
        box=Box(box[0], box[1], box[2], box[3]),
        kind=RegionKind(str(data["kind"])),
        polygon=polygon,
        confidence=float(data["confidence"]),  # type: ignore[arg-type]
        source_text=str(data["source_text"]),
        target_text=str(data["target_text"]),
        vertical=bool(data["vertical"]),
        order=int(data["order"]),  # type: ignore[call-overload]
        notes=str(data["notes"]),
    )


class Database:
    """Thin SQLite wrapper holding pages and their regions."""

    def __init__(self, path: Path | str) -> None:
        """Open (and if needed create) the database at ``path``.

        Args:
            path: File path, or ``":memory:"`` for an ephemeral database.
        """
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def save(self, page: Page) -> None:
        """Insert or replace ``page`` and all of its regions."""
        regions = json.dumps([_region_to_dict(r) for r in page.regions], ensure_ascii=False)
        self._conn.execute(
            "INSERT INTO pages (id, width, height, stage, error, regions) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "width=excluded.width, height=excluded.height, stage=excluded.stage, "
            "error=excluded.error, regions=excluded.regions",
            (page.id, page.width, page.height, str(page.stage), page.error, regions),
        )
        self._conn.commit()

    def get(self, page_id: str) -> Page | None:
        """Load one page by id, or ``None`` when it does not exist."""
        row = self._conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_page(row)

    def recent(self, limit: int = 50) -> list[Page]:
        """List the most recently created pages, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM pages ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_page(row) for row in rows]

    def delete(self, page_id: str) -> bool:
        """Delete one page.

        Args:
            page_id: Page to remove.

        Returns:
            ``True`` when a row was actually deleted.
        """
        cursor = self._conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_page(self, row: sqlite3.Row) -> Page:
        """Rebuild a page from a database row."""
        return Page(
            id=str(row["id"]),
            width=int(row["width"]),
            height=int(row["height"]),
            stage=Stage(str(row["stage"])),
            error=str(row["error"]),
            regions=[_region_from_dict(d) for d in json.loads(row["regions"])],
        )
