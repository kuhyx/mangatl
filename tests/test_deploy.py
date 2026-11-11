"""Fail-closed checks for ``scripts/deploy.sh``.

Only the refusal paths run here: the happy path restarts real systemd units
and loads ~10 GB into VRAM, so it is checked by running the deploy itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent.parent / "scripts" / "deploy.sh"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the deploy script with overridden paths, never the live ones."""
    return subprocess.run(  # noqa: S603 -- fixed repo script, test-owned args
        ["/usr/bin/bash", str(DEPLOY)],
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_deploy_refuses_without_venv(tmp_path: Path) -> None:
    """A missing app venv stops the deploy before anything is touched."""
    result = _run(
        {
            "MANGATL_SRC": str(tmp_path / "src"),
            "MANGATL_VENV": str(tmp_path / "missing-venv"),
        }
    )
    assert result.returncode != 0
    assert "install-arch.sh" in result.stderr


def test_deploy_refuses_dirty_checkout(tmp_path: Path) -> None:
    """A checkout that cannot fast-forward fails the deploy."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "pip").write_text("#!/bin/sh\nexit 0\n")
    (venv / "pip").chmod(0o755)
    result = _run(
        {
            "MANGATL_SRC": str(tmp_path / "not-a-repo"),
            "MANGATL_VENV": str(tmp_path / "venv"),
        }
    )
    assert result.returncode != 0
