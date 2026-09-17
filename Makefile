.PHONY: setup data data-demo pipeline test test-fast app

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Project standard is Python 3.13; 3.11+ is required (pandas 3.x needs it).
# Override with `make setup PYTHON_BOOTSTRAP=/path/to/python` if none of the
# auto-detected interpreters below are suitable.
PYTHON_BOOTSTRAP ?=
MIN_PY_MAJOR := 3
MIN_PY_MINOR := 11

# iCloud Drive skips syncing any folder whose name ends in .nosync. When
# ~/Documents is iCloud-synced, a venv placed directly under it gets partially
# uploaded/evicted mid-use and breaks. In that case we create $(VENV).nosync
# instead and symlink $(VENV) -> $(VENV).nosync, so every other target below
# can keep referring to plain $(VENV).
ICLOUD_DOCS := $(HOME)/Library/Mobile Documents/com~apple~CloudDocs/Documents

DATA_FILES := \
	data/synthetic/accounts.csv \
	data/synthetic/transactions.csv \
	data/synthetic/complaints.csv \
	data/synthetic/account_monthly_snapshot.csv \
	data/synthetic/metric_definitions.csv

SAMPLE_FILES := \
	data/sample/accounts.parquet \
	data/sample/transactions.parquet \
	data/sample/complaints.parquet \
	data/sample/account_monthly_snapshot.parquet

setup:
	@if [ -d "$(ICLOUD_DOCS)" ]; then \
		target="$(VENV).nosync"; \
		echo "iCloud-synced Documents detected -> will create $$target and symlink $(VENV) to it"; \
	else \
		target="$(VENV)"; \
		echo "Documents is not iCloud-synced -> will create $$target"; \
	fi; \
	if [ -n "$(PYTHON_BOOTSTRAP)" ]; then \
		if ! command -v "$(PYTHON_BOOTSTRAP)" >/dev/null 2>&1; then \
			echo "PYTHON_BOOTSTRAP=$(PYTHON_BOOTSTRAP) not found or not executable." >&2; \
			exit 1; \
		fi; \
		candidates="$(PYTHON_BOOTSTRAP)"; \
	else \
		candidates="python3.13 python3.12 python3.11 python3 /opt/miniconda3/bin/python3 /opt/homebrew/bin/python3"; \
	fi; \
	found=""; \
	for candidate in $$candidates; do \
		if ! command -v "$$candidate" >/dev/null 2>&1; then \
			echo "Skipping $$candidate: not found"; \
			continue; \
		fi; \
		resolved=$$(command -v "$$candidate"); \
		ver=$$("$$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null); \
		major=$$(printf '%s' "$$ver" | cut -d. -f1); \
		minor=$$(printf '%s' "$$ver" | cut -d. -f2); \
		if [ -z "$$major" ]; then \
			echo "Skipping $$candidate ($$resolved): could not determine Python version"; \
			continue; \
		fi; \
		if [ "$$major" -lt $(MIN_PY_MAJOR) ] || { [ "$$major" -eq $(MIN_PY_MAJOR) ] && [ "$$minor" -lt $(MIN_PY_MINOR) ]; }; then \
			echo "Skipping $$candidate ($$resolved): Python $$ver, need $(MIN_PY_MAJOR).$(MIN_PY_MINOR)+"; \
			continue; \
		fi; \
		if ! "$$candidate" -c "import venv, ensurepip" >/dev/null 2>&1; then \
			echo "Skipping $$candidate ($$resolved): missing venv/ensurepip module"; \
			continue; \
		fi; \
		rm -rf "$$target"; \
		if "$$candidate" -m venv "$$target" >/tmp/metricguard-setup-venv.log 2>&1; then \
			echo "Using $$candidate ($$resolved, Python $$ver)"; \
			found="$$candidate"; \
			break; \
		fi; \
		echo "Skipping $$candidate ($$resolved): Python $$ver has venv/ensurepip modules but venv creation failed (broken interpreter install - see /tmp/metricguard-setup-venv.log)"; \
		rm -rf "$$target"; \
	done; \
	if [ -z "$$found" ]; then \
		echo "" >&2; \
		echo "ERROR: Python $(MIN_PY_MAJOR).$(MIN_PY_MINOR)+ with venv support required. Tried: $$candidates." >&2; \
		echo "Install a working Python $(MIN_PY_MAJOR).$(MIN_PY_MINOR)+ (project standard: Python 3.13), or point make at one directly:" >&2; \
		echo "    make setup PYTHON_BOOTSTRAP=/path/to/python3.13" >&2; \
		exit 1; \
	fi; \
	if [ "$$target" != "$(VENV)" ]; then ln -sfn "$$target" $(VENV); fi
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt

# Full-size synthetic data is gitignored (too large to commit); regenerate it
# deterministically (fixed seed in src/generate_synthetic_data.py) rather than
# downloading it. Only runs when a data file is actually missing.
data: $(DATA_FILES)

$(DATA_FILES): src/generate_synthetic_data.py
	$(PYTHON) src/generate_synthetic_data.py --size full

# Demo-size synthetic data (Parquet, data/sample/, amendment A10) is small
# enough to commit and is used for CI / the deployed app. metric_definitions.csv
# is shared with the full dataset (data/synthetic/) and is not written here --
# see config.METRIC_DEFINITIONS_PATH. Only runs when a sample file is missing.
data-demo: $(SAMPLE_FILES)

$(SAMPLE_FILES): src/generate_synthetic_data.py
	$(PYTHON) src/generate_synthetic_data.py --size demo

pipeline: data
	$(PYTHON) src/run_pipeline.py

test: data
	$(PYTHON) -m pytest tests/ -v

# Runs every test today: no test is marked `slow` yet. Becomes a real fast
# subset once Codex registers the `slow` marker in pyproject.toml and tags
# the full-size-data tests with it (task CX1 / amendment A6).
test-fast: data
	$(PYTHON) -m pytest tests/ -v -m "not slow"

app: data
	$(PYTHON) -m streamlit run app/streamlit_app.py
