"""Checks for ``scripts/check_cpu_torch.sh``, the no-CUDA-wheels gate.

Each test hands the script a fake interpreter whose ``-m pip list`` prints a
scripted freeze list, so no real venv or torch install is needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parent.parent / "scripts" / "check_cpu_torch.sh"


def _run(tmp_path: Path, freeze: str, *, exit_code: int = 0) -> subprocess.CompletedProcess[str]:
    """Run the check against a fake python that prints ``freeze``."""
    (tmp_path / "freeze.txt").write_text(freeze)
    python = tmp_path / "python"
    python.write_text(f'#!/bin/sh\ncat "{tmp_path / "freeze.txt"}"\nexit {exit_code}\n')
    python.chmod(0o755)
    return subprocess.run(  # noqa: S603 -- fixed repo script, test-owned args
        ["/usr/bin/bash", str(CHECK), str(python)],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_cpu_only_set_passes(tmp_path: Path) -> None:
    """CPU torch plus the BSD nvidia-ml-py exception is clean."""
    result = _run(
        tmp_path,
        "numpy==2.5.1\ntorch==2.15.0+cpu\ntorchvision==0.29.1+cpu\nnvidia-ml-py==13.610.43\n",
    )
    assert result.returncode == 0, result.stderr
    assert "CPU-only" in result.stdout


@pytest.mark.parametrize(
    "line",
    [
        "nvidia-cublas==13.1.1.3",
        "nvidia_cudnn_cu13==9.20.0.48",
        "cuda-bindings==13.3.1",
        "cuda-toolkit==13.0.3.0",
        "triton==3.7.1",
        "torch==2.13.0",
        "torch==2.13.0+cu130",
        "torchvision==0.28.0",
    ],
)
def test_cuda_wheel_fails(tmp_path: Path, line: str) -> None:
    """Every proprietary CUDA wheel, or a non-CPU torch, fails and is named."""
    result = _run(tmp_path, f"numpy==2.5.1\n{line}\n")
    assert result.returncode == 1
    assert line in result.stderr


def test_broken_interpreter_fails_closed(tmp_path: Path) -> None:
    """An interpreter that cannot list packages is an error, never a pass."""
    result = _run(tmp_path, "", exit_code=1)
    assert result.returncode == 2
    assert "cannot run" in result.stderr


def test_installer_takes_torch_from_cpu_index_first() -> None:
    """The installer pulls CPU torch before `.[ml]`, gates on it, defaults to cpu."""
    script = (CHECK.parent / "install-arch.sh").read_text()
    torch_step = script.index('--index-url "$TORCH_INDEX" torch torchvision')
    assert 'TORCH_INDEX="https://download.pytorch.org/whl/cpu"' in script
    assert torch_step < script.index('pip install --quiet -e "${REPO_DIR}[ml]"')
    assert torch_step < script.index("scripts/check_cpu_torch.sh")
    assert "${MANGATL_DEVICE:-cpu}" in script
