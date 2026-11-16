#!/usr/bin/env bash
set -u
cd /home/lizheng/query_minco_mainline || exit 2
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
E0_RUN="research_outputs/g4cp3_manual_E0_$(date +%Y%m%d_%H%M%S)_$$"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B -m query_minco.audit.run_provenance_v1 \
  --output "$E0_RUN/process" --cwd /home/lizheng/query_minco_mainline --timeout-s 60 -- \
  "$PY" -B experiments/g4cp3_observed_v1/manual_e0.py \
  --allow-sensors RESEARCHER_MANUAL_E0_SENSORS_NO_ARM --output "$E0_RUN/run"
code=$?
printf 'E0_RUN=%s\nTRUE_EXIT=%s\n' "$E0_RUN" "$code"
printf '%s\n' 'Exit 3 means sensor capture succeeded but initial tube remains unproved: NO-GO for motion.'
exit "$code"
