#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || -z "$1" ]]; then
    echo 'Usage: postprocess_run.sh RUN_DIRECTORY' >&2
    exit 2
fi
run=$(cd -- "$1" && pwd -P)
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd -- "$repo"
exec "${QUERY_MINCO_PYTHON:-/home/lizheng/anaconda3/envs/yopo/bin/python}" -B \
    -m experiments.sim_dev_oracle_forest10m_v1.plot \
    --run "$run" --output "$run/plots" --after-cleanup
