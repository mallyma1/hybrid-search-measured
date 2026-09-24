#!/usr/bin/env bash
# Reproduce everything from scratch: environment, tests, data download, all
# arms, metrics and significance tests. Writes results/.
#
#   ./run.sh                 # full run, reuses cached corpus vectors if present
#   FRESH=1 ./run.sh         # drop cached corpus vectors first (timed from zero)
#   VENV=/path/to/venv ./run.sh
#   ./run.sh --no-rerank     # skip the cross-encoder experiment
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
  else
    python3.12 -m venv "$VENV"
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet --python "$VENV/bin/python" -r requirements.txt
else
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi

"$VENV/bin/python" -m pytest -q

if [ "${FRESH:-0}" = "1" ]; then
  rm -rf data/cache
fi
export TOKENIZERS_PARALLELISM=false
# Exit code 3 means a time budget stopped a checkpointed stage; resume until done.
while true; do
  status=0
  "$VENV/bin/python" -m hsm.run "$@" || status=$?
  [ "$status" -eq 3 ] || exit "$status"
  echo "[run.sh] resuming from checkpoints"
done
