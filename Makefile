.PHONY: setup data pipeline test test-fast app

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

setup:
	@candidates="$(PYTHON_BOOTSTRAP)"; \
	if [ -n "$$candidates" ]; then \
		if ! command -v "$$candidates" >/dev/null 2>&1; then \
			echo "PYTHON_BOOTSTRAP=$$candidates not found or not executable." >&2; \
			exit 1; \
		fi; \
	else \
		candidates="python3.13 python3.12 python3.11 python3"; \
	fi; \
	found=""; \
	for candidate in $$candidates; do \
		if command -v "$$candidate" >/dev/null 2>&1; then \
			ver=$$("$$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null); \
			major=$$(echo "$$ver" | cut -d. -f1); \
			minor=$$(echo "$$ver" | cut -d. -f2); \
			if [ -n "$$major" ] && [ "$$major" -ge $(MIN_PY_MAJOR) ] && [ "$$minor" -ge $(MIN_PY_MINOR) ]; then \
				echo "Using $$candidate (Python $$ver)"; \
				found="$$candidate"; \
				break; \
			else \
				echo "Skipping $$candidate (Python $${ver:-unknown}, need $(MIN_PY_MAJOR).$(MIN_PY_MINOR)+)"; \
			fi; \
		fi; \
	done; \
	if [ -z "$$found" ]; then \
		echo "" >&2; \
		echo "ERROR: no Python $(MIN_PY_MAJOR).$(MIN_PY_MINOR)+ interpreter found (tried: python3.13, python3.12, python3.11, python3)." >&2; \
		echo "Install Python $(MIN_PY_MAJOR).$(MIN_PY_MINOR)+ (project standard: Python 3.13), or point make at one directly:" >&2; \
		echo "    make setup PYTHON_BOOTSTRAP=/path/to/python3.13" >&2; \
		exit 1; \
	fi; \
	if [ -d "$(ICLOUD_DOCS)" ]; then \
		echo "iCloud-synced Documents detected -> creating $(VENV).nosync and symlinking $(VENV) to it"; \
		"$$found" -m venv $(VENV).nosync; \
		ln -sfn $(VENV).nosync $(VENV); \
	else \
		echo "Documents is not iCloud-synced -> creating $(VENV)"; \
		"$$found" -m venv $(VENV); \
	fi
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt

# Full-size synthetic data is gitignored (too large to commit); regenerate it
# deterministically (fixed seed in src/generate_synthetic_data.py) rather than
# downloading it. Only runs when a data file is actually missing.
data: $(DATA_FILES)

$(DATA_FILES): src/generate_synthetic_data.py
	$(PYTHON) src/generate_synthetic_data.py

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
