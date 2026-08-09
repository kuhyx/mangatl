"""Concrete implementations of the pipeline ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def require(module: str, extra: str = "ml") -> ModuleType:
    """Import a heavy optional dependency, or fail in a way the pipeline handles.

    The adapters import ``torch``/``ultralytics``/``manga_ocr`` lazily, so a box
    without the ``[ml]`` extra only discovers the gap mid-request. ``ImportError``
    is not in the set :meth:`mangatl.pipeline.Pipeline.run` catches, so left alone
    it escapes as an unhandled 500 instead of a failed page. Re-raising as
    ``RuntimeError`` keeps the promise that a page fails gracefully, and says how
    to fix it.

    Args:
        module: Fully-qualified module name to import.
        extra: Name of the optional-dependency extra that provides it.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the module is not installed.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        msg = (
            f"{module!r} is not installed; it ships in the optional '{extra}' extra. "
            f"Install it with: pip install -e '.[{extra}]'"
        )
        raise RuntimeError(msg) from exc
