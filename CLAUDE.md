## Commands

Needs `pip install -e ".[dev]"` (add `[ml]` only for integration tests / `make cover`).

- run: `./run.sh` (starts the llama.cpp + web user services; `./run.sh stop` releases VRAM)
- test: `python -m pytest --no-cov`
- test-changed: `scripts/test_changed.sh`
- lint: `make lint types`
- coverage: `COVERAGE_FILE=.coverage.unit python -m pytest --cov-fail-under=0`
- coverage-gaps: `coverage-gaps coverage.xml`
