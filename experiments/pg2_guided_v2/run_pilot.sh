#!/usr/bin/env bash
# Manual-only launcher. validate must pass before the sole run command.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: run_pilot.sh ABS_SPEC ABS_AUTH ABS_EXCLUSIVE_LAUNCH_DIR" >&2
  exit 2
fi
: "${PG2_PYTHON:?Set PG2_PYTHON to the verified absolute Python interpreter}"
test -x "$PG2_PYTHON"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PG2_PYTHON" -m query_minco.training.pg2_guided_v2.cli validate --spec "$1" --authorization "$2"
"$PG2_PYTHON" -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); r=pathlib.Path(sys.argv[2]).resolve(); assert p.parent == r/"research_outputs/pg2_guided_v2/_preparation"; p.mkdir(exist_ok=False)' "$3" "$repo"
set +e
timeout --signal=TERM --kill-after=15s 3615s "$PG2_PYTHON" -u -m query_minco.training.pg2_guided_v2.cli run --spec "$1" --authorization "$2" > "$3/stdout.log" 2> "$3/stderr.log"
rc=$?
set -e
"$PG2_PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); f=(p/"PROCESS_EXIT.json").open("x"); json.dump({"returncode":int(sys.argv[2])},f); f.close()' "$3" "$rc"
exit "$rc"
