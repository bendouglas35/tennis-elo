PYTHON ?= python3
export PYTHONPATH := src

.PHONY: setup data build build-aux v05 clean-note baseline leakage pace-tune elo tune validate final-test test clean

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

# v0.5: qualifying (data/raw_quali) and Challenger (data/raw_chall) results for the ratings.
build-aux:
	$(PYTHON) -m tennis.cli build-aux

# v0.5 rolling-origin comparison of the feature steps (development years only).
v05:
	$(PYTHON) -m tennis.cli models v05

baseline:
	$(PYTHON) -m tennis.cli baseline

leakage:
	$(PYTHON) -m tennis.cli leakage-check

pace-tune:
	$(PYTHON) -m tennis.cli pace-tune

elo:
	$(PYTHON) -m tennis.cli elo-ladder

tune:
	$(PYTHON) -m tennis.cli models tune

validate:
	$(PYTHON) -m tennis.cli models validate

# One-off milestone: refuses to run twice (pass --force only to reproduce).
final-test:
	$(PYTHON) -m tennis.cli final-test

test:
	$(PYTHON) -m pytest -q

clean:
	rm -f data/*.duckdb data/*.duckdb.wal
