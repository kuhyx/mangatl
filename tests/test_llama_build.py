"""Checks that llama-server is built and kept runnable without the CUDA toolkit.

The FOSS cleanup removed the proprietary ``cuda`` package, which left a
CUDA-linked llama-server unable to load ``libcudart``. The installer now builds
llama.cpp on Vulkan, and ``scripts/check_llama_libs.sh`` fails on any shared
library the binary cannot load, so install and deploy catch a stale build.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
CHECK = SCRIPTS / "check_llama_libs.sh"
INSTALL = SCRIPTS / "install-arch.sh"


def _run(tmp_path: Path, ldd_output: str, *, ldd_exit: int = 0) -> subprocess.CompletedProcess[str]:
    """Run the check with a fake ``ldd`` that prints ``ldd_output``."""
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    (tmp_path / "ldd.txt").write_text(ldd_output)
    ldd = tmp_path / "ldd"
    ldd.write_text(f'#!/bin/sh\ncat "{tmp_path / "ldd.txt"}"\nexit {ldd_exit}\n')
    ldd.chmod(0o755)
    return subprocess.run(  # noqa: S603 -- fixed repo script, test-owned args
        ["/usr/bin/bash", str(CHECK), str(binary)],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_all_libraries_resolved_passes(tmp_path: Path) -> None:
    """A binary whose libraries all resolve is runnable."""
    result = _run(tmp_path, "\tlibvulkan.so.1 => /usr/lib/libvulkan.so.1 (0x1)\n")
    assert result.returncode == 0, result.stderr


def test_missing_cuda_runtime_fails(tmp_path: Path) -> None:
    """The exact deploy failure: libcudart.so.13 not found."""
    result = _run(tmp_path, "\tlibcudart.so.13 => not found\n\tlibc.so.6 => /usr/lib/libc.so.6\n")
    assert result.returncode == 1
    assert "libcudart.so.13" in result.stderr


def test_missing_binary_fails(tmp_path: Path) -> None:
    """No binary at all is not runnable."""
    result = subprocess.run(  # noqa: S603 -- fixed repo script, test-owned args
        ["/usr/bin/bash", str(CHECK), str(tmp_path / "absent")],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1
    assert "absent" in result.stderr


def test_ldd_failure_fails(tmp_path: Path) -> None:
    """An ldd that cannot read the binary fails closed."""
    result = _run(tmp_path, "not a dynamic executable\n", ldd_exit=1)
    assert result.returncode == 1


def test_installer_builds_vulkan_not_cuda() -> None:
    """The installer neither installs the CUDA toolkit nor builds against it."""
    text = INSTALL.read_text()
    assert "-DGGML_VULKAN=ON" in text
    assert "-DGGML_CUDA=OFF" in text
    assert "-DGGML_CUDA=ON" not in text
    assert " cuda " not in text
    assert "cudnn" not in text
    assert "check_llama_libs.sh" in text
    # Vulkan also lists the iGPU; without a pin the layers split across both.
    assert "--device Vulkan0" in text


def test_deploy_checks_llama_libraries() -> None:
    """Deploy refuses to start an LLM whose binary cannot load its libraries."""
    assert "check_llama_libs.sh" in (SCRIPTS / "deploy.sh").read_text()
