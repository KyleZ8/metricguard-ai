.PHONY: setup data pipeline test test-fast app

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# iCloud Drive skips syncing any folder whose name ends in .nosync. When
# ~/Documents is iCloud-synced, a venv placed directly under it gets partially
# uploaded/evicted mid-use and breaks. In that case we create .venv.nosync
# instead and symlink .venv -> .venv.nosync, so every other target below can
# keep referring to plain .venv.
ICLOUD_DOCS := $(HOME)/Library/Mobile Documents/com~apple~CloudDocs/Documents

DATA_FILES := \
	data/synthetic/accounts.csv \
	data/synthetic/transactions.csv \
	data/synthetic/complaints.csv \
	data/synthetic/account_monthly_snapshot.csv \
	data/synthetic/metric_definitions.csv

setup:
	@if [ -d "$(ICLOUD_DOCS)" ]; then \
		echo "iCloud-synced Documents detected -> creating .venv.nosync and symlinking .venv to it"; \
		python3 -m venv .venv.nosync; \
		ln -sfn .venv.nosync .venv; \
	else \
		echo "Documents is not iCloud-synced -> creating .venv"; \
		python3 -m venv .venv; \
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

test-fast: data
	$(PYTHON) -m pytest tests/ -v -m "not slow"

app: data
	$(PYTHON) -m streamlit run app/streamlit_app.py
