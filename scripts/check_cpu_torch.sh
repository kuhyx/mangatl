#!/usr/bin/env bash
#
# Fail if a Python environment holds NVIDIA's proprietary CUDA wheels.
#
#   scripts/check_cpu_torch.sh [PYTHON]
#
# PYTHON defaults to the app venv's interpreter. Exit 0 when the installed set
# is CPU-only; exit 1 and list every offender otherwise. Offenders:
#   - any nvidia-* or cuda-* distribution (cuDNN, cuBLAS, cuda-bindings, ...)
#   - triton, which only CUDA torch pulls in and which bundles NVIDIA's ptxas
#   - torch / torchvision without a +cpu local version: PyPI's plain Linux
#     wheels are the CUDA build, so "no +cpu" means "not the CPU build"
# One exception: nvidia-ml-py is BSD-licensed pure-Python NVML bindings (one
# pynvml.py, no binaries) that ultralytics requires unconditionally.

set -euo pipefail

readonly PYTHON="${1:-$HOME/.local/share/mangatl/venv/bin/python}"
readonly ALLOWED="nvidia-ml-py"

die() { printf 'check_cpu_torch: %s\n' "$*" >&2; exit 2; }

main() {
  local freeze line name version offenders=()
  freeze="$("$PYTHON" -m pip list --format=freeze 2>/dev/null)" \
    || die "cannot run '$PYTHON -m pip list'"
  while IFS= read -r line; do
    [[ "$line" == *==* ]] || continue
    # pip normalises nothing in freeze output; compare lowercase, _ as -.
    name="${line%%==*}"
    name="${name,,}"
    name="${name//_/-}"
    version="${line#*==}"
    [[ "$name" == "$ALLOWED" ]] && continue
    case "$name" in
      nvidia-* | cuda-* | triton) offenders+=("$line") ;;
      torch | torchvision) [[ "$version" == *+cpu ]] || offenders+=("$line") ;;
    esac
  done <<<"$freeze"
  if ((${#offenders[@]})); then
    printf 'CUDA wheels in %s:\n' "$PYTHON" >&2
    printf '  %s\n' "${offenders[@]}" >&2
    exit 1
  fi
  printf 'ok: %s is CPU-only\n' "$PYTHON"
}

main "$@"
