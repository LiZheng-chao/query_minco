#!/usr/bin/env bash
# NOT EXECUTED. Separate approved file and clean CPU0 are mandatory.
set -euo pipefail
ROOT=/home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
PLAN="$ROOT/research_outputs/pg1_v2_r12/plans_clean002"
ARM=Rtail
if test "$#" -ge 1; then PLAN="$1"; fi
if test "$#" -ge 2; then ARM="$2"; fi
case "$ARM" in R0|Rtail) ;; *) printf 'Arm must be R0 or Rtail\n' >&2; exit 64;; esac
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$ROOT:$ROOT/tests" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0
SPEC="$PLAN/$ARM"_RUN_SPEC.json
AUTH="$PLAN/$ARM"_AUTHORIZATION_APPROVED.json
test -f "$SPEC" && test -f "$AUTH" || { printf 'Missing exact spec or separately approved authorization; no training started.\n' >&2; exit 65; }
"$PY" -B -s -m query_minco.training.pg1_v2_r12.cli validate --spec "$SPEC" --authorization "$AUTH"
LOG="$PLAN/$ARM"_launch001
mkdir "$LOG"
set +e
timeout --signal=TERM --kill-after=10s 3600s "$PY" -B -s -m query_minco.training.pg1_v2_r12.cli run --spec "$SPEC" --authorization "$AUTH" > "$LOG/stdout.log" 2> "$LOG/stderr.log"
rc=$?
set -e
"$PY" -B -s - "$LOG/PROCESS_EXIT.json" "$rc" <<'PY'
import json,sys
with open(sys.argv[1],'x') as f:json.dump(dict(exit_code=int(sys.argv[2])),f)
PY
printf 'Pilot exited %s; logs=%s\n' "$rc" "$LOG"
exit "$rc"
