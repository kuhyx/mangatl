"""Composition root: choose concrete adapters and build the pipeline.

Kept in its own module so the FastAPI app can be constructed in tests without
ever touching a model file, and so switching a backend is a one-line change
in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mangatl.adapters.translate_libre import LibreTranslateTranslator
from mangatl.adapters.translate_llm import LlmTranslator
from mangatl.adapters.typeset import PillowTypesetter
from mangatl.adapters.vision import AotInpainter, MangaOcr, YoloBubbleDetector
from mangatl.pipeline import Pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from mangatl.config import Settings
    from mangatl.ports import Translator

FONT_PREFERENCE = ("ComicNeue-Bold.ttf", "Bangers-Regular.ttf", "NotoSansJP-Bold.ttf")


def pick_font(font_dir: Path) -> Path:
    """Choose a lettering font from the bundled open-licensed set.

    Args:
        font_dir: Directory the install script populated.

    Returns:
        The first preferred font present, else any TTF in the directory.

    Raises:
        FileNotFoundError: If the directory contains no usable font.
    """
    for name in FONT_PREFERENCE:
        candidate = font_dir / name
        if candidate.exists():
            return candidate
    fallback = sorted(font_dir.glob("*.ttf"))
    if fallback:
        return fallback[0]
    msg = f"no lettering font found in {font_dir}; run scripts/fetch-fonts.sh"
    raise FileNotFoundError(msg)


def build_translator(settings: Settings) -> Translator:
    """Pick the translation backend.

    The local LLM is always preferred. LibreTranslate is selected only when
    ``llm_base_url`` has been blanked out, because its quality on manga
    dialogue is not good enough to ship.

    Args:
        settings: Runtime configuration.

    Returns:
        The chosen translator.
    """
    if not settings.llm_base_url.strip():
        return LibreTranslateTranslator(settings)
    return LlmTranslator(settings)


def build_pipeline(settings: Settings) -> Pipeline:
    """Construct the full pipeline from configuration.

    Args:
        settings: Runtime configuration.

    Returns:
        A pipeline wired to the real adapters.
    """
    return Pipeline(
        detector=YoloBubbleDetector(settings),
        ocr=MangaOcr(settings),
        translator=build_translator(settings),
        inpainter=AotInpainter(settings),
        typesetter=PillowTypesetter(pick_font(settings.font_dir)),
    )
