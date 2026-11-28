## Commands

Needs `pip install -e ".[dev]"` (add `[ml]` only for integration tests / `make cover`).

- run: `./run.sh` (starts the llama.cpp + web user services; `./run.sh stop` releases VRAM)
- deploy: `scripts/deploy.sh` (pulls ~/src/mangatl, restarts the user units, exits 0 only when the LLM and web health checks pass)
- test: `python -m pytest --no-cov`
- test-changed: `scripts/test_changed.sh`
- lint: n/a: dropped (kuhy, 2026-11-27)
- coverage: n/a: dropped (kuhy, 2026-11-27)
- coverage-gaps: n/a: dropped (kuhy, 2026-11-27)
