#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "LOCAL_CPU_PILOT_B_FAILED line=$LINENO exit=$rc" >&2; exit "$rc"' ERR
ROOT=/home/lizheng/query_minco_mainline
cd "$ROOT"
export PG2_PYTHON=/home/lizheng/anaconda3/envs/yopo/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PG2_PYTHON" - <<'PY'
from pathlib import Path
from query_minco.training.io import read,require
p=Path("/home/lizheng/query_minco_mainline/research_outputs/pg2_guided_v2/runs/A42_A_PG2v2_CPU_Pilot32_run002")
require(not(p/"FAILURE_RECORD.json").exists(),"PREVIOUS_ARM_FAILED")
r=read(p/"RESULT.json");a=read(p/"POST_RUN_ACCEPTANCE.json")
require(r.get("status")=="COMPLETED_AWAITING_AUDIT" and r.get("optimizer_updates")==32 and a.get("passed") is True,"PREVIOUS_ARM_NOT_ACCEPTED")
PY
SPEC="$ROOT/research_outputs/pg2_guided_v2/_preparation/local_cpu_pilot32_preflight_B_run002/RUN_SPEC_WITH_PREFLIGHT.json"
AUTH="$ROOT/research_outputs/pg2_guided_v2/_preparation/local_cpu_pilot32_preflight_B_run002/AUTHORIZATION_APPROVED.json"
test -f "$SPEC"
test -f "$AUTH"
bash "$ROOT/experiments/pg2_guided_v2/run_pilot.sh" "$SPEC" "$AUTH" "$ROOT/research_outputs/pg2_guided_v2/_preparation/local_cpu_pilot32_launch_B_run002"
