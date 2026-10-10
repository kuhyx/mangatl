#!/usr/bin/env bash
#
# Start mangatl for a translating session, and stop it again afterwards.
#
# The systemd units are deliberately NOT enabled at login: llama-server pins
# ~10 GB of VRAM, which you want back when you are doing anything else on the
# GPU. This script is the manual switch.
#
#   ./run.sh            start both services, wait until ready, open the UI
#   ./run.sh stop       stop both services and release the VRAM
#   ./run.sh status     show what is running
#   ./run.sh logs       follow the LLM log
#
# Re-runnable: starting something already running is a no-op.

set -euo pipefail

readonly LLM_UNIT="mangatl-llm.service"
readonly WEB_UNIT="mangatl.service"
readonly LLM_HEALTH="http://127.0.0.1:8081/health"
readonly WEB_URL="http://127.0.0.1:8781"
# The 14B model has to be read off disk and pushed into VRAM on a cold start.
readonly READY_TIMEOUT_S=300

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!! \033[0m%s\n' "$*" >&2; }
die()  { printf '\033[1;31mXXX \033[0m%s\n' "$*" >&2; exit 1; }

require_units() {
  if ! systemctl --user cat "$LLM_UNIT" &>/dev/null; then
    die "$LLM_UNIT is not installed -- run ./scripts/install-arch.sh first"
  fi
}

# Poll an endpoint until it answers or we give up. Returns 1 on timeout so the
# caller can report which stage failed rather than hanging forever.
wait_for() {
  local url="$1" label="$2" waited=0
  while ! curl -sf --max-time 3 -o /dev/null "$url"; do
    if ! systemctl --user is-active --quiet "$3"; then
      warn "$label died on startup. Recent log:"
      journalctl --user -u "$3" -n 20 --no-pager >&2
      return 1
    fi
    if ((waited >= READY_TIMEOUT_S)); then
      warn "$label did not come up within ${READY_TIMEOUT_S}s"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    # A cold 14B load is slow; say something so it does not look hung.
    ((waited % 20 == 0)) && log "  still waiting for $label (${waited}s)"
  done
  return 0
}

start() {
  require_units
  log "Starting the translation LLM (this loads ~10 GB into VRAM)"
  systemctl --user start "$LLM_UNIT"
  wait_for "$LLM_HEALTH" "llama-server" "$LLM_UNIT" || die "LLM failed to start"
  log "  LLM ready"

  log "Starting the web UI"
  systemctl --user start "$WEB_UNIT"
  wait_for "$WEB_URL/api/health" "web UI" "$WEB_UNIT" || die "web UI failed to start"
  log "  UI ready at $WEB_URL"

  if command -v xdg-open &>/dev/null; then
    xdg-open "$WEB_URL" &>/dev/null &
  fi

  cat <<EOF

  mangatl is up.

    UI:      $WEB_URL
    Stop it: $0 stop      (releases the VRAM)

EOF
}

stop() {
  log "Stopping mangatl"
  systemctl --user stop "$WEB_UNIT" "$LLM_UNIT" 2>/dev/null || true
  log "  stopped; VRAM released"
}

status() {
  printf '%-22s %s\n' "$LLM_UNIT" "$(systemctl --user is-active "$LLM_UNIT")"
  printf '%-22s %s\n' "$WEB_UNIT" "$(systemctl --user is-active "$WEB_UNIT")"
  if command -v nvidia-smi &>/dev/null; then
    printf '%-22s %s\n' "VRAM in use" "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  fi
}

main() {
  case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    logs)   journalctl --user -fu "$LLM_UNIT" ;;
    *)      die "usage: $0 [start|stop|status|logs]" ;;
  esac
}

main "$@"
