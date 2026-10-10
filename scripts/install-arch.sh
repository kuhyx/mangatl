#!/usr/bin/env bash
# mangatl unattended installer for Arch Linux + NVIDIA.
#
# Assumes you are NOT present. Every choice is made for you, defaults favour
# translation quality over speed, and nothing here contacts a third party
# except to download open-licensed models and fonts.
#
#   ./scripts/install-arch.sh
#
# Re-runnable: every step is idempotent and skips work already done.

set -euo pipefail

readonly PREFIX="${MANGATL_PREFIX:-$HOME/.local/share/mangatl}"
readonly MODELS="$PREFIX/models"
readonly FONTS="$PREFIX/fonts"
readonly VENV="$PREFIX/venv"
readonly LLAMA_DIR="$PREFIX/llama.cpp"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_DIR

# Qwen3-14B at Q5_K_M: ~10 GB on a 24 GB card, leaving comfortable room for
# manga-ocr, the detector and the inpainter to stay resident alongside it.
readonly LLM_REPO="Qwen/Qwen3-14B-GGUF"
readonly LLM_FILE="Qwen3-14B-Q5_K_M.gguf"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!! \033[0m%s\n' "$*" >&2; }
die()  { printf '\033[1;31mXXX \033[0m%s\n' "$*" >&2; exit 1; }

need_root() {
  if [[ $EUID -eq 0 ]]; then
    die "do not run this as root; it installs into \$HOME and calls sudo only where needed"
  fi
}

# --------------------------------------------------------------------------
# 1. Base system
# --------------------------------------------------------------------------
# The kernel-module package is NOT a fixed name. Arch dropped the plain
# `nvidia` package in favour of the open modules, so hardcoding it aborts the
# whole install (`set -e` + `--noconfirm`) on any current box. Pick whichever
# module package is already installed, else the first one that still exists in
# the repos.
nvidia_module_pkg() {
  local candidates=(nvidia-open-dkms nvidia-open nvidia-dkms nvidia)
  local p
  for p in "${candidates[@]}"; do
    if pacman -Qi "$p" &>/dev/null; then
      printf '%s' "$p"
      return
    fi
  done
  for p in "${candidates[@]}"; do
    if pacman -Si "$p" &>/dev/null; then
      printf '%s' "$p"
      return
    fi
  done
  warn "no NVIDIA kernel-module package found; skipping it"
}

install_packages() {
  log "Installing base packages"
  local pkgs=(
    base-devel git cmake ninja
    python python-pip python-virtualenv
    nvidia-utils cuda cudnn
    libjpeg-turbo libpng freetype2
    curl jq
  )
  local module
  module="$(nvidia_module_pkg)"
  [[ -n "$module" ]] && pkgs+=("$module")
  local missing=()
  for p in "${pkgs[@]}"; do
    pacman -Qi "$p" &>/dev/null || missing+=("$p")
  done
  if ((${#missing[@]})); then
    sudo pacman -S --needed --noconfirm "${missing[@]}"
  else
    log "  all present, nothing to do"
  fi
}

check_gpu() {
  log "Checking GPU"
  if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found. If you just installed the driver, REBOOT and re-run."
    warn "Continuing in CPU mode: usable, but a 14B model on CPU is unusably slow."
    export MANGATL_DEVICE=cpu
    return
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  local vram
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if ((vram < 12000)); then
    warn "Only ${vram} MiB of VRAM. Falling back to an 8B model."
    warn "Edit LLM_REPO/LLM_FILE in this script if you want a different size."
  fi
}

# --------------------------------------------------------------------------
# 2. Python environment
# --------------------------------------------------------------------------
setup_venv() {
  log "Creating Python environment at $VENV"
  mkdir -p "$PREFIX" "$MODELS" "$FONTS"
  [[ -d "$VENV" ]] || python -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip wheel

  log "Installing mangatl and its ML extras (this pulls torch, expect ~4 GB)"
  # Braces are load-bearing: "$REPO_DIR[ml]" parses as an array subscript and
  # expands to the empty string, silently installing nothing.
  pip install --quiet -e "${REPO_DIR}[ml]"
  pip install --quiet huggingface_hub
}

# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
fetch_models() {
  log "Fetching models"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  # manga-ocr caches itself on first use; warm it now so the first real page
  # is not a five-minute download.
  if [[ ! -d "$HOME/.cache/huggingface/hub/models--kha-white--manga-ocr-base" ]]; then
    log "  manga-ocr (Apache-2.0, ~444 MB)"
    python "$REPO_DIR/scripts/fetch_models.py" ocr
  else
    log "  manga-ocr already cached"
  fi

  if [[ ! -f "$MODELS/bubble-seg.pt" ]]; then
    log "  speech-bubble segmentation (GPL-3.0)"
    python "$REPO_DIR/scripts/fetch_models.py" detector "$MODELS"
  else
    log "  bubble detector already present"
  fi

  if [[ ! -f "$MODELS/aot-inpainting.safetensors" ]]; then
    log "  AOT-GAN inpainting (MIT, ~22 MB)"
    python "$REPO_DIR/scripts/fetch_models.py" inpainter "$MODELS"
  else
    log "  inpainter already present"
  fi

  if [[ ! -f "$MODELS/$LLM_FILE" ]]; then
    log "  translation LLM $LLM_FILE (Apache-2.0, ~10 GB) -- this is the long one"
    python "$REPO_DIR/scripts/fetch_models.py" gguf "$MODELS" "$LLM_REPO" "$LLM_FILE"
  else
    log "  translation LLM already present"
  fi

}

# --------------------------------------------------------------------------
# 4. Fonts -- open-licensed only
# --------------------------------------------------------------------------
fetch_fonts() {
  log "Fetching lettering fonts (OFL / Apache only)"
  bash "$REPO_DIR/scripts/fetch-fonts.sh" "$FONTS"
}

# --------------------------------------------------------------------------
# 5. llama.cpp server
# --------------------------------------------------------------------------
build_llama() {
  log "Building llama.cpp with CUDA"
  if [[ -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    log "  already built"
    return
  fi
  [[ -d "$LLAMA_DIR" ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"
  [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || die "llama-server did not build"
}

# --------------------------------------------------------------------------
# 6. systemd user services
# --------------------------------------------------------------------------
install_units() {
  log "Installing systemd user units"
  local unit_dir="$HOME/.config/systemd/user"
  mkdir -p "$unit_dir"

  cat > "$unit_dir/mangatl-llm.service" <<EOF
[Unit]
Description=mangatl translation LLM (llama.cpp)
After=network.target

[Service]
Type=simple
# -c 8192 gives the model room for a dense page plus prior-page context.
# --n-gpu-layers 999 offloads everything; a 3090 has the VRAM for it.
ExecStart=$LLAMA_DIR/build/bin/llama-server \\
  --model $MODELS/$LLM_FILE \\
  --alias qwen3-14b-instruct \\
  --host 127.0.0.1 --port 8081 \\
  --ctx-size 8192 --n-gpu-layers 999 \\
  --flash-attn on --parallel 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

  cat > "$unit_dir/mangatl.service" <<EOF
[Unit]
Description=mangatl web UI
After=mangatl-llm.service
Wants=mangatl-llm.service

[Service]
Type=simple
Environment=MANGATL_DATA_DIR=$PREFIX
Environment=MANGATL_FONT_DIR=$FONTS
Environment=MANGATL_DETECTOR_WEIGHTS=$MODELS/bubble-seg.pt
Environment=MANGATL_LLM_BASE_URL=http://127.0.0.1:8081/v1
Environment=MANGATL_LLM_MODEL=qwen3-14b-instruct
Environment=MANGATL_DEVICE=${MANGATL_DEVICE:-cuda}
ExecStart=$VENV/bin/mangatl serve --host 127.0.0.1 --port 8781
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable mangatl-llm.service mangatl.service
  log "  enabled. Start with: systemctl --user start mangatl-llm mangatl"
}

verify() {
  log "Verifying installation"
  local ok=1
  [[ -f "$MODELS/$LLM_FILE" ]]      || { warn "missing LLM weights";      ok=0; }
  [[ -f "$MODELS/bubble-seg.pt" ]]  || { warn "missing detector weights"; ok=0; }
  compgen -G "$FONTS/*.ttf" >/dev/null || { warn "no fonts installed";    ok=0; }
  [[ -x "$VENV/bin/mangatl" ]]      || { warn "mangatl not on PATH";      ok=0; }
  [[ -x "$LLAMA_DIR/build/bin/llama-server" ]] || { warn "no llama-server"; ok=0; }
  ((ok)) || die "installation incomplete, see warnings above"
  log "All components present."
}

main() {
  need_root
  install_packages
  check_gpu
  setup_venv
  fetch_models
  fetch_fonts
  build_llama
  install_units
  verify
  cat <<EOF

  Done.

    systemctl --user start mangatl-llm mangatl
    xdg-open http://127.0.0.1:8781

  Batch a chapter, carrying context page to page:

    $VENV/bin/mangatl batch ~/scans/ch-01 --out ~/out/ch-01

  First request after a cold start is slow: the LLM has to load ~10 GB
  into VRAM. Watch it with:  journalctl --user -fu mangatl-llm

EOF
}

main "$@"
