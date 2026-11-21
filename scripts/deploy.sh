#!/usr/bin/env bash
#
# Deploy mangatl for real: fast-forward the live checkout, refresh the app
# venv when dependencies changed, restart the user units and wait for both
# health checks. Exit 0 only when the web UI and the LLM both answer.
#
# The live service runs an editable install of ~/src/mangatl, so pulling that
# checkout is what ships new code; the restart makes the running process pick
# it up. Re-runnable: a second run is a pull no-op plus a restart.
#
# Never touches data: the database, uploads, renders and models in
# ~/.local/share/mangatl stay as they are.
#
# Note: this leaves the LLM running (~10 GB VRAM). `./run.sh stop` frees it.

set -euo pipefail

readonly SRC="${MANGATL_SRC:-$HOME/src/mangatl}"
readonly VENV="${MANGATL_VENV:-$HOME/.local/share/mangatl/venv}"
readonly LLAMA_SERVER="${MANGATL_LLAMA_SERVER:-$HOME/.local/share/mangatl/llama.cpp/build/bin/llama-server}"
readonly LLM_UNIT="mangatl-llm.service"
readonly WEB_UNIT="mangatl.service"
readonly LLM_HEALTH="http://127.0.0.1:8081/health"
readonly WEB_HEALTH="http://127.0.0.1:8781/api/health"
# A cold 14B load reads the model off disk into VRAM.
readonly READY_TIMEOUT_S=300

log() { printf '==> %s\n' "$*"; }
die() { printf 'deploy: %s\n' "$*" >&2; exit 1; }

preflight() {
  [[ -x "$VENV/bin/pip" ]] || die "no app venv at $VENV -- run scripts/install-arch.sh"
  local unit
  for unit in "$LLM_UNIT" "$WEB_UNIT"; do
    systemctl --user cat "$unit" &>/dev/null \
      || die "$unit is not installed -- run scripts/install-arch.sh"
  done
}

# Fast-forward the live checkout; refuses a dirty or diverged tree.
pull() {
  local before after
  before="$(git -C "$SRC" rev-parse HEAD)" || die "$SRC is not a git checkout"
  git -C "$SRC" pull --ff-only --quiet || die "git pull --ff-only failed in $SRC"
  after="$(git -C "$SRC" rev-parse HEAD)"
  log "checkout at ${after:0:7} (was ${before:0:7})"
  # The editable install picks up code changes on its own; only new or
  # changed dependencies need pip.
  if [[ "$before" != "$after" ]] \
    && ! git -C "$SRC" diff --quiet "$before" "$after" -- pyproject.toml; then
    log "pyproject.toml changed: reinstalling into $VENV"
    "$VENV/bin/pip" install --quiet -e "$SRC" || die "pip install failed"
  fi
}

# Poll a URL until its body reports ok; fail fast if the unit dies.
wait_healthy() {
  local url="$1" unit="$2" waited=0
  until curl -sf --max-time 3 "$url" | grep -q '"status": *"ok"'; do
    if ! systemctl --user is-active --quiet "$unit"; then
      journalctl --user -u "$unit" -n 20 --no-pager >&2 || true
      die "$unit is not running"
    fi
    ((waited >= READY_TIMEOUT_S)) && die "$unit not healthy within ${READY_TIMEOUT_S}s"
    sleep 2
    waited=$((waited + 2))
  done
  log "$unit healthy"
}

main() {
  preflight
  pull
  # After the pull, so the check script exists in the checkout. A binary that
  # cannot load its GPU runtime would die with exit 127 on start.
  bash "$SRC/scripts/check_llama_libs.sh" "$LLAMA_SERVER" \
    || die "llama-server is not runnable -- rerun scripts/install-arch.sh"
  systemctl --user daemon-reload
  # Start, not restart, the LLM: its binary and model do not change with the
  # repo, and a restart would reload 10 GB for nothing.
  systemctl --user start "$LLM_UNIT" || die "could not start $LLM_UNIT"
  wait_healthy "$LLM_HEALTH" "$LLM_UNIT"
  systemctl --user restart "$WEB_UNIT" || die "could not restart $WEB_UNIT"
  wait_healthy "$WEB_HEALTH" "$WEB_UNIT"
}

main "$@"
