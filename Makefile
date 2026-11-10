# The gate must test THIS tree with an interpreter that has the project's
# deps, whatever venv happens to be on PATH (a caller's unrelated venv breaks
# mypy and pytest plugins). Order: the installed app venv (has torch, which the
# integration half needs), then ./.venv, then plain `python`. Override with
# `make PY=...`. The app venv holds an editable install of the main checkout,
# so src/ is forced to the front of PYTHONPATH: the worktree code is what runs.
MANGATL_PREFIX ?= $(HOME)/.local/share/mangatl
PY ?= $(firstword $(wildcard $(MANGATL_PREFIX)/venv/bin/python $(CURDIR)/.venv/bin/python) python)
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: check fmt lint types test cover integration install serve clean

check: fmt lint types cover ## Run every quality gate, in the order CI runs them

fmt:
	ruff format --check .

lint:
	ruff check .

types:
	$(PY) -m mypy

test:
	$(PY) -m pytest

integration:  ## Needs the [ml] extra: exercises the real torch network
	$(PY) -m pytest tests/integration --override-ini="testpaths=tests/integration"

# The 100% gate belongs on the union of both suites: the generative code cannot
# run without [ml], and the unit suite must keep running without it. Skips the
# integration half (and so cannot reach 100%) if torch is absent.
cover:
	rm -f .coverage .coverage.unit .coverage.integration
	COVERAGE_FILE=.coverage.unit $(PY) -m pytest --cov-fail-under=0
	COVERAGE_FILE=.coverage.integration $(PY) -m pytest tests/integration \
		--override-ini="testpaths=tests/integration" \
		--cov=mangatl --cov-branch --cov-fail-under=0
	$(PY) -m coverage combine .coverage.unit .coverage.integration
	$(PY) -m coverage report --show-missing --fail-under=100

install:
	./scripts/install-arch.sh

serve:
	mangatl serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
