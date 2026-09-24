VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: all test run fresh clean

all:            ## environment, tests, full experiment
	VENV=$(VENV) ./run.sh

test:
	$(PY) -m pytest -q

run:
	TOKENIZERS_PARALLELISM=false $(PY) -m hsm.run

fresh:          ## recompute corpus vectors instead of loading the cache
	FRESH=1 VENV=$(VENV) ./run.sh

clean:
	rm -rf data/cache results/runs results/per_query.csv
