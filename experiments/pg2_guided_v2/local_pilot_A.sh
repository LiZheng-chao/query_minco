#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "LOCAL_CPU_PILOT_A_FAILED line=$LINENO exit=$rc" >&2; exit "$rc"' ERR
ROOT=/home/lizheng/query_minco_mainline
cd "$ROOT"
export PG2_PYTHON=/home/lizheng/anaconda3/envs/yopo/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SPEC="$ROOT/research_outputs/pg2_guided_v2/_preparation/local_cpu_pilot32_preflight_A_run002/RUN_SPEC_WITH_PREFLIGHT.json"
AUTH="$ROOT/research_outputs/pg2_guided_v2/_preparation/local_cpu_pilot32_preflight_A_run002/AUTHORIZATION_APPROVED.json"
test -f "$SPEC"
test -f "$AUTH"
bash "$ROOT/experiments/pg2_guided_v2/run_pilot.sh" "$SPEC" "$AUTH" "$ROOT/research_outputs/pg2_guided_v2/_preparation/local_cpu_pilot32_launch_A_run002"
