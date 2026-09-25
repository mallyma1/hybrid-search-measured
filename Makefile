VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: all test run figures fresh clean

all:            ## environment, tests, full experiment, charts
	VENV=$(VENV) ./run.sh

test:           ## unit tests only
	$(PY) -m pytest -q

run:            ## experiment only, in an existing environment
	TOKENIZERS_PARALLELISM=false $(PY) -m hsm.run

figures:        ## redraw results/figures/*.png from the saved results
	$(PY) -m hsm.plot

fresh:          ## recompute corpus vectors instead of loading the cache
	FRESH=1 VENV=$(VENV) ./run.sh

clean:          ## remove caches and per-query outputs (keeps results.md, results.json, figures)
	rm -rf data/cache results/runs results/per_query.csv
