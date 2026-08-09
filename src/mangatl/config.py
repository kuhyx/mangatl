"""Runtime configuration, all overridable by ``MANGATL_*`` environment vars."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROOT = Path.home() / ".local" / "share" / "mangatl"


class Settings(BaseSettings):
    """Everything the app needs to know about the local machine.

    Every default points at a service running on localhost. Nothing in this
    application talks to a third-party API unless you deliberately repoint
    ``llm_base_url`` at one.
    """

    model_config = SettingsConfigDict(
        env_prefix="MANGATL_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Field(default=DEFAULT_ROOT)
    """Where uploads, renders and the SQLite database live."""

    font_dir: Path = Field(default=DEFAULT_ROOT / "fonts")
    """Directory holding the bundled OFL/Apache lettering fonts."""

    llm_base_url: str = Field(default="http://127.0.0.1:8081/v1")
    """OpenAI-compatible endpoint served by llama.cpp or vLLM."""

    llm_model: str = Field(default="qwen3-14b-instruct")
    """Model name passed through to the local server."""

    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    """Low but non-zero: manga dialogue needs some latitude, not none."""

    llm_timeout_s: float = Field(default=180.0, gt=0.0)
    """Whole-page requests on a 14B model are slow; do not set this low."""

    libretranslate_url: str = Field(default="http://127.0.0.1:5000")
    """Offline fallback only. Quality is materially worse than the LLM."""

    detector_weights: Path = Field(default=DEFAULT_ROOT / "models" / "bubble-seg.pt")
    """YOLO segmentation weights for balloon and text detection."""

    ocr_model: str = Field(default="kha-white/manga-ocr-base")
    """Purpose-built Japanese manga OCR. Apache-2.0, ~444 MB."""

    inpaint_weights: Path = Field(default=DEFAULT_ROOT / "models" / "aot-inpainting.safetensors")
    """AOT-GAN weights. MIT-licensed conversion, unlike some LaMa checkpoints."""

    device: str = Field(default="cuda")
    """Torch device string. Set to ``cpu`` to run without a GPU."""

    max_upload_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    """Reject oversized uploads before touching disk."""

    critique_pass: bool = Field(default=True)
    """Second-pass self-review. Roughly doubles latency, clearly worth it."""

    round_trip_qe: bool = Field(default=False)
    """Back-translation scoring. Triples latency; enable for final QC only."""

    @property
    def upload_dir(self) -> Path:
        """Directory holding original uploaded pages."""
        return self.data_dir / "uploads"

    @property
    def render_dir(self) -> Path:
        """Directory holding cleaned and typeset output pages."""
        return self.data_dir / "renders"

    @property
    def db_path(self) -> Path:
        """SQLite database file path."""
        return self.data_dir / "mangatl.sqlite3"

    def ensure_dirs(self) -> None:
        """Create every directory the app writes to."""
        for path in (self.data_dir, self.upload_dir, self.render_dir, self.font_dir):
            path.mkdir(parents=True, exist_ok=True)
