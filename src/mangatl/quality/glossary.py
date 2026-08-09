"""Series glossary: the cheapest large win in translation consistency.

A local model will happily call the same character "Big Brother", "Onii-chan"
and "Nii-san" on three consecutive pages. Injecting a term list into the
prompt fixes most of it; verifying the output afterwards fixes the rest.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from mangatl.domain.models import GlossaryEntry

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path


class Glossary:
    """An ordered, de-duplicated set of terms scoped to one series."""

    def __init__(self, entries: Iterable[GlossaryEntry] = ()) -> None:
        """Build a glossary.

        Args:
            entries: Initial terms. Later entries with the same source term
                replace earlier ones.
        """
        self._entries: dict[str, GlossaryEntry] = {}
        for entry in entries:
            self.add(entry)

    def __len__(self) -> int:
        """Number of distinct source terms held."""
        return len(self._entries)

    def __iter__(self) -> Iterator[GlossaryEntry]:
        """Iterate terms longest-source-first, which is the matching order."""
        return iter(self.entries)

    @property
    def entries(self) -> list[GlossaryEntry]:
        """Terms sorted longest source first so specific terms win."""
        return sorted(self._entries.values(), key=lambda e: (-len(e.source), e.source))

    def add(self, entry: GlossaryEntry) -> None:
        """Insert or replace a term."""
        self._entries[entry.source] = entry

    def remove(self, source: str) -> bool:
        """Drop a term.

        Args:
            source: Source-language term to remove.

        Returns:
            ``True`` when a term was actually removed.
        """
        return self._entries.pop(source, None) is not None

    def relevant_to(self, lines: Sequence[str]) -> list[GlossaryEntry]:
        """Return only the terms that actually occur in ``lines``.

        Sending the whole glossary on every page wastes context and dilutes
        attention; sending only what appears keeps the prompt sharp.

        Args:
            lines: Source lines about to be translated.

        Returns:
            Matching terms, longest source first.
        """
        blob = "".join(lines)
        return [e for e in self.entries if e.source in blob]

    def violations(self, source: str, target: str) -> list[GlossaryEntry]:
        """Find terms present in ``source`` but not honoured in ``target``.

        Args:
            source: Source-language line.
            target: Its translation.

        Returns:
            Entries whose source term appeared but whose required target
            rendering did not. Matching is case-insensitive on the target
            side, because casing varies legitimately mid-sentence.
        """
        lowered = target.lower()
        return [e for e in self.entries if e.source in source and e.target.lower() not in lowered]

    def to_json(self) -> str:
        """Serialise the glossary to a JSON array string."""
        return json.dumps([asdict(e) for e in self.entries], ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> Glossary:
        """Load a glossary from a JSON array string.

        Args:
            blob: JSON produced by :meth:`to_json`.

        Returns:
            The loaded glossary.

        Raises:
            ValueError: If the payload is not a JSON array of objects.
        """
        data = json.loads(blob)
        if not isinstance(data, list):
            msg = "glossary JSON must be an array"
            raise TypeError(msg)
        entries: list[GlossaryEntry] = []
        for item in data:
            if not isinstance(item, dict):
                msg = "glossary entries must be objects"
                raise TypeError(msg)
            if "source" not in item or "target" not in item:
                msg = "glossary entries need both 'source' and 'target'"
                raise ValueError(msg)
            entries.append(
                GlossaryEntry(
                    source=str(item["source"]),
                    target=str(item["target"]),
                    note=str(item.get("note", "")),
                ),
            )
        return cls(entries)

    def save(self, path: Path) -> None:
        """Write the glossary to ``path`` as UTF-8 JSON."""
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Glossary:
        """Read a glossary from ``path``, returning an empty one if absent."""
        if not path.exists():
            return cls()
        return cls.from_json(path.read_text(encoding="utf-8"))
