"""LibreTranslate fallback.

Kept because it is genuinely offline, AGPL, and trivial to run, so it is a
useful safety net when the LLM server is down. It is *not* the recommended
path: Argos/LibreTranslate is a general-web NMT model, and generic ja->en
models fall apart on the colloquial, subject-dropping, context-dependent
register manga actually uses. It also translates line by line, which throws
away the whole-page context that carries most of the quality.

Use it for triage. Do not ship its output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mangatl.config import Settings
    from mangatl.domain.models import GlossaryEntry


class LibreTranslateTranslator:
    """Line-by-line NMT via a local LibreTranslate instance."""

    name = "libretranslate"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        """Build the translator.

        Args:
            settings: Runtime configuration.
            client: Optional pre-built ``httpx.Client``.
        """
        self._settings = settings
        self._client = client

    def _ensure_client(self) -> Any:
        """Return the HTTP client, creating one on first use."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._settings.llm_timeout_s)
        return self._client

    def _apply_glossary(self, text: str, glossary: Sequence[GlossaryEntry]) -> str:
        """Force glossary terms into output NMT could not be told about."""
        out = text
        for entry in sorted(glossary, key=lambda e: -len(e.source)):
            out = out.replace(entry.source, entry.target)
        return out

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> list[str]:
        """Translate each line independently.

        Args:
            lines: Source lines in reading order.
            context: Ignored. LibreTranslate has no cross-line context, which
                is precisely why this backend is the fallback.
            glossary: Applied as a post-hoc string substitution.

        Returns:
            One translated line per input line.
        """
        del context
        client = self._ensure_client()
        out: list[str] = []
        for line in lines:
            if not line.strip():
                out.append("")
                continue
            response = client.post(
                f"{self._settings.libretranslate_url}/translate",
                json={"q": line, "source": "ja", "target": "en", "format": "text"},
            )
            response.raise_for_status()
            translated = str(response.json().get("translatedText", ""))
            out.append(self._apply_glossary(translated, glossary))
        return out
