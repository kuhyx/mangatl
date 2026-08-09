#!/usr/bin/env bash
# The reading you probably did not intend: DO NOT BUILD THIS.
#
# You asked for an app. You now have one. But before you invest a weekend in
# it, spend twenty minutes here first, because the honest answer is that
# manga-image-translator plus BallonsTranslator already gets you ~80% of the
# result today, and the remaining 20% is exactly the part mangatl exists for.
#
# This script stands up both, so you can decide from evidence instead of
# from a design doc.
#
#   ./scripts/try-existing-first.sh
#
# What you are actually evaluating:
#
#   manga-image-translator  GPL-3.0. The de-facto engine. Full auto pipeline,
#     REST API, Docker image with every model baked in. Its own maintainer
#     concedes the typesetting is weak: text is fitted to the detected text
#     region rather than the balloon, and there is "no good solution" yet.
#     No project persistence, no collaboration, no CBZ, no PSD.
#
#   BallonsTranslator      GPL-3.0. The power-user desktop editor. v1.5.8
#     (July 2025) added context-aware LLM translation and glossary support,
#     which overlaps heavily with mangatl's quality layer. Qt desktop only:
#     single user, no web UI, no server API.
#
# If those two cover your actual workflow, mangatl is a hobby project rather
# than a necessity, and that is a completely fine thing to conclude.

set -euo pipefail

readonly WORK="${MANGATL_TRY_DIR:-$HOME/scanlation-eval}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!! \033[0m%s\n' "$*" >&2; }

check_docker() {
  command -v docker &>/dev/null || {
    warn "docker not installed. sudo pacman -S docker && sudo systemctl enable --now docker"
    exit 1
  }
  docker info &>/dev/null || {
    warn "cannot talk to the docker daemon. Add yourself to the docker group and re-login."
    exit 1
  }
  pacman -Qi nvidia-container-toolkit &>/dev/null || {
    warn "installing nvidia-container-toolkit for GPU passthrough"
    sudo pacman -S --needed --noconfirm nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  }
}

verify_gpu_in_docker() {
  log "Verifying GPU passthrough into containers"
  if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    log "  GPU visible inside containers"
  else
    warn "GPU not visible inside docker. CPU will work but be very slow."
    warn "On Arch: check that the host driver version matches the CUDA base image."
  fi
}

run_mit() {
  log "Starting manga-image-translator on http://127.0.0.1:5003"
  mkdir -p "$WORK/mit/in" "$WORK/mit/out"
  cat > "$WORK/mit/docker-compose.yml" <<'EOF'
services:
  manga-image-translator:
    image: zyddnys/manga-image-translator:main
    # The image is ~15 GB: every model is baked in, so first pull is slow
    # but there is nothing to download at runtime.
    command: >
      python -m manga_translator server
      --host 0.0.0.0 --port 5003 --use-gpu
    ports: ["5003:5003"]
    volumes:
      - ./in:/app/in
      - ./out:/app/out
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
  (cd "$WORK/mit" && docker compose up -d)
  log "  drop pages in $WORK/mit/in, results land in $WORK/mit/out"
}

install_ballons() {
  log "Installing BallonsTranslator (desktop, runs natively)"
  if [[ -d "$WORK/BallonsTranslator" ]]; then
    log "  already cloned"
  else
    git clone --depth 1 https://github.com/dmMaze/BallonsTranslator "$WORK/BallonsTranslator"
  fi
  cat <<EOF

  BallonsTranslator needs its own venv and a model download on first run:

    cd $WORK/BallonsTranslator
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python launch.py

  In Settings, point the translator at your local llama-server
  (OpenAI-compatible, http://127.0.0.1:8081/v1) rather than a cloud key,
  then turn on the glossary and context options added in v1.5.8.

EOF
}

decision_guide() {
  cat <<'EOF'

  How to decide, after you have run twenty real pages through both:

    Stay with the existing tools if...
      - you work alone
      - one chapter at a time is fine
      - you are happy touching up typesetting in GIMP/Krita afterwards
      - a desktop app is not a problem

    Build on mangatl if...
      - you want a browser UI you can reach from another machine
      - you want a chapter processed as one unit, with translated context
        carried page to page automatically
      - you want a series glossary enforced and verified, not just suggested
      - you want the project state to survive closing the window
      - you want to hand pages to someone else for QC

  The second list is the entire reason mangatl exists. If none of it
  matters to you, the honest recommendation is: do not build it.

EOF
}

main() {
  mkdir -p "$WORK"
  check_docker
  verify_gpu_in_docker
  run_mit
  install_ballons
  decision_guide
}

main "$@"
