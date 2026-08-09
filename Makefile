.PHONY: check fmt lint types test cover integration install serve clean

check: fmt lint types cover ## Run every quality gate, in the order CI runs them

fmt:
	ruff format --check .

lint:
	ruff check .

types:
	mypy

test:
	python -m pytest

integration:  ## Needs the [ml] extra: exercises the real torch network
	python -m pytest tests/integration --override-ini="testpaths=tests/integration"

# The 100% gate belongs on the union of both suites: the generative code cannot
# run without [ml], and the unit suite must keep running without it. Skips the
# integration half (and so cannot reach 100%) if torch is absent.
cover:
	rm -f .coverage .coverage.unit .coverage.integration
	COVERAGE_FILE=.coverage.unit python -m pytest --cov-fail-under=0
	COVERAGE_FILE=.coverage.integration python -m pytest tests/integration \
		--override-ini="testpaths=tests/integration" \
		--cov=mangatl --cov-branch --cov-fail-under=0
	coverage combine .coverage.unit .coverage.integration
	coverage report --show-missing --fail-under=100

install:
	./scripts/install-arch.sh

serve:
	mangatl serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
