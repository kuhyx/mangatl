"""Guard: the web UI paints no words (kuhy's textgate rule).

Anything kuhy sees carries no Claude-written text. The page's proper noun in
``<title>`` is allowed; translated regions are content (filled in at run time
from the API) and so never appear in the static page. The old wording survives
only as ``aria-label``, which nothing paints.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from mangatl.web.app import INDEX_HTML

WORD = re.compile(r"[A-Za-z]{2,}")
# String literals the script paints into the page.
PAINTED_LITERAL = re.compile(r"""textContent\s*=\s*(['"])(.*?)\1""")
UNPAINTED = {"script", "style", "title"}


class _Page(HTMLParser):
    """Collect painted text nodes, script bodies and interactive controls."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.painted: list[str] = []
        self.scripts: list[str] = []
        self.controls: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track nesting and remember controls kuhy can click."""
        if tag in {"button", "label", "input"}:
            self.controls.append((tag, dict(attrs)))
        if tag not in {"input", "meta", "img", "br"}:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        """Pop back to the matching open tag."""
        if tag in self.stack:
            while self.stack.pop() != tag:
                pass

    def handle_data(self, data: str) -> None:
        """Sort text into painted nodes and script source."""
        if self.stack and self.stack[-1] == "script":
            self.scripts.append(data)
        elif not UNPAINTED.intersection(self.stack):
            self.painted.append(data)


def _parse() -> _Page:
    page = _Page()
    page.feed(INDEX_HTML)
    return page


def test_no_painted_words() -> None:
    """No text node outside <title>/<script>/<style> holds a word."""
    words = [w for text in _parse().painted for w in WORD.findall(text)]
    assert words == []


def test_script_paints_no_words() -> None:
    """Status strings the script writes into the page are glyphs, not sentences."""
    source = "".join(_parse().scripts)
    literals = [m.group(2) for m in PAINTED_LITERAL.finditer(source)]
    assert literals, "guard found no painted literals: the regex no longer matches"
    assert [w for lit in literals for w in WORD.findall(lit)] == []


def test_controls_are_spoken_not_painted() -> None:
    """Every icon-only control keeps its meaning as a spoken label.

    The native file input paints "Choose File / No file chosen", so it must be
    hidden behind an icon label.
    """
    controls = _parse().controls
    visible = [(tag, a) for tag, a in controls if "hidden" not in a]
    assert {tag for tag, _ in visible} == {"button", "label"}
    assert all(a.get("aria-label") for _, a in visible)
