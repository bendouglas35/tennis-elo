PYTHON ?= python3
export PYTHONPATH := src

.PHONY: setup data build baseline leakage test clean

setup:
	$(PYTHON) -m pip install -e ".[dev]"

# Download atp_matches_YYYY.csv (2000+) into data/raw.
# If GitHub is unreachable, copy the CSVs yourself and run:
#   make data FROM=/path/to/folder/with/csvs
data:
ifdef FROM
	$(PYTHON) -m tennis.cli download --from-dir $(FROM)
else
	$(PYTHON) -m tennis.cli download --source $(or $(SOURCE),sackmann)
endif

build:
	$(PYTHON) -m tennis.cli build

baseline:
	$(PYTHON) -m tennis.cli baseline

leakage:
	$(PYTHON) -m tennis.cli leakage-check

test:
	$(PYTHON) -m pytest -q

clean:
	rm -f data/*.duckdb data/*.duckdb.wal
