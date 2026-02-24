.PHONY: install run test lint clean

CONDA ?= $(shell which conda 2>/dev/null || echo conda)
ENV_NAME ?= tsseg-app-env
PYTHON ?= python

# ── Install ──────────────────────────────────────────────────────────────────

install:
	$(CONDA) create -n $(ENV_NAME) python=3.10 -y
	$(CONDA) run -n $(ENV_NAME) pip install -r requirements.txt
	$(CONDA) run -n $(ENV_NAME) pip install -e ".[dev]"
	@echo "✅ Environment '$(ENV_NAME)' ready. Activate with: conda activate $(ENV_NAME)"

install-dev: install
	$(CONDA) run -n $(ENV_NAME) pip install pytest ruff

# ── Run ──────────────────────────────────────────────────────────────────────

run:
	$(PYTHON) app.py

run-share:
	GRADIO_SHARE=1 $(PYTHON) app.py

# ── Quality ──────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
