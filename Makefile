.PHONY: check fmt lint types test install serve clean

check: fmt lint types test ## Run every quality gate, in the order CI runs them

fmt:
	ruff format --check .

lint:
	ruff check .

types:
	mypy

test:
	python -m pytest

install:
	./scripts/install-arch.sh

serve:
	mangatl serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
