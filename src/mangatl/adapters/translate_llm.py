"""Translation backed by a local OpenAI-compatible LLM server.

This is the recommended default. On an RTX 3090 a 14B model quantised to
Q5_K_M served by ``llama-server`` translates a dense manga page in a few
seconds and is materially better than any dedicated NMT model on colloquial
Japanese, because it can use the whole page as context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from mangatl.quality.glossary import Glossary
from mangatl.quality.refine import ChatBackend, Refiner, RefineSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mangatl.config import Settings
    from mangatl.domain.models import GlossaryEntry
    from mangatl.quality.prompt import StyleGuide


class HttpChatBackend(ChatBackend):
    """Chat backend that speaks the OpenAI ``/chat/completions`` shape."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        """Build a backend.

        Args:
            settings: Runtime configuration.
            client: Optional pre-built ``httpx.Client``. When omitted, one is
                created lazily so importing this module never requires httpx.
        """
        self._settings = settings
        self._client = client

    def _ensure_client(self) -> Any:
        """Return the HTTP client, creating one on first use."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._settings.llm_timeout_s)
        return self._client

    @override
    def chat(self, system: str, user: str) -> str:
        """Send one turn to the local server and return the reply text.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            The assistant message content.

        Raises:
            RuntimeError: If the server reply has no choices.
        """
        client = self._ensure_client()
        response = client.post(
            f"{self._settings.llm_base_url}/chat/completions",
            json={
                "model": self._settings.llm_model,
                "temperature": self._settings.llm_temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            msg = "local LLM returned no choices; is the model loaded?"
            raise RuntimeError(msg)
        content = choices[0]["message"]["content"]
        return str(content)


class LlmTranslator:
    """Whole-page, context-aware, glossary-enforcing translator."""

    name = "local-llm"

    def __init__(
        self,
        settings: Settings,
        *,
        backend: ChatBackend | None = None,
        style: StyleGuide | None = None,
    ) -> None:
        """Build the translator.

        Args:
            settings: Runtime configuration.
            backend: Chat backend; defaults to :class:`HttpChatBackend`.
            style: Series style guide.
        """
        self._refiner = Refiner(
            backend if backend is not None else HttpChatBackend(settings),
            settings=RefineSettings(
                critique=settings.critique_pass,
                repair_glossary=True,
                round_trip=settings.round_trip_qe,
            ),
            style=style,
        )

    def translate_page(
        self,
        lines: Sequence[str],
        *,
        context: Sequence[str] = (),
        glossary: Sequence[GlossaryEntry] = (),
    ) -> list[str]:
        """Translate one page of lines.

        Args:
            lines: Source lines in reading order.
            context: Preceding dialogue, oldest first.
            glossary: Terms that must be honoured.

        Returns:
            One translated line per input line.
        """
        report = self._refiner.run(lines, context=context, glossary=Glossary(glossary))
        return report.lines
