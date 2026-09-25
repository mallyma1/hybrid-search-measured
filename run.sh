#!/usr/bin/env bash
# Reproduce everything from scratch: environment, tests, data download, all
# arms, metrics, significance tests and charts. Writes results/.
#
#   ./run.sh                 # full run, reuses cached corpus vectors if present
#   FRESH=1 ./run.sh         # drop cached corpus vectors first (timed from zero)
#   VENV=/path/to/venv ./run.sh
#   ./run.sh --no-rerank     # skip the cross-encoder experiment
#   ./run.sh --help          # print this help and exit
#
# Other arguments are passed to `python -m hsm.run` (--device, --max-seconds).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac

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
  [ "$status" -eq 3 ] || break
  echo "[run.sh] resuming from checkpoints"
done
[ "$status" -eq 0 ] || exit "$status"

"$VENV/bin/python" -m hsm.plot   # results/figures/ndcg10.png and metrics.png
