#!/usr/bin/env bash
#
# Fail when llama-server cannot load one of its shared libraries.
#
# A llama.cpp build links against whatever GPU runtime was present at build
# time. When that runtime goes away (the CUDA toolkit was removed for the FOSS
# cleanup), the binary still exists but dies with exit 127 at start. This
# check turns that into a clear error for the installer (rebuild) and deploy.
#
# Usage: check_llama_libs.sh <path to llama-server>

set -euo pipefail

die() { printf 'check_llama_libs: %s\n' "$*" >&2; exit 1; }

main() {
  local binary="${1:?usage: check_llama_libs.sh <llama-server>}"
  [[ -x "$binary" ]] || die "$binary is missing or not executable"
  local libs
  libs="$(ldd "$binary" 2>&1)" || die "ldd cannot read $binary: $libs"
  local missing
  missing="$(grep 'not found' <<<"$libs" | awk '{print $1}' | sort -u | tr '\n' ' ')" || true
  [[ -z "$missing" ]] || die "$binary cannot load: ${missing% } -- rebuild it with scripts/install-arch.sh"
}

main "$@"
