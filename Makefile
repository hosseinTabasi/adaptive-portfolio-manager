.PHONY: install test run figures lint

PYTHON ?= /workspace/.venv/bin/python
PIP ?= /workspace/.venv/bin/pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q -m "not slow"

run:
	$(PYTHON) -m src.app --config configs/config.yaml

figures:
	$(PYTHON) -m src.app --config configs/config.yaml --skip-train-if-models

lint:
	$(PYTHON) -m ruff check src tests
