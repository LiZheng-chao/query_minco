#!/usr/bin/env bash
# Researcher-run native GPU rendering and CPU-only frozen inference. No ROS/training.
set -Eeuo pipefail
set -o noclobber
umask 077
ulimit -c 0
cd /home/lizheng/query_minco_mainline
export PYTHONPATH=/home/lizheng/query_minco_mainline
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/local/cuda-11.8/lib64:/usr/lib/wsl/lib
export LC_ALL=C.UTF-8
unset PYTHONHOME
base=/home/lizheng/query_minco_mainline/research_outputs/rsps_forest15m_v1
entry=/home/lizheng/query_minco_mainline/experiments/risk_aware_set_physics_v1/forest15m_v1.py
case "${1:-}" in
 smoke) kind=smoke; rid=smoke001; seconds=1800 ;;
 forest) kind=forest; rid=forest_paired001; seconds=14400 ;;
 check-smoke|check-forest)
   export CUDA_VISIBLE_DEVICES=
   if [[ "$1" == check-smoke ]]; then kind=smoke; rid=smoke001; else kind=forest; rid=forest_paired001; fi
   exec timeout --signal=TERM --kill-after=15s 300s /home/lizheng/anaconda3/envs/yopo/bin/python -B -s "$entry" --kind "$kind" --run-id "$rid" --check
   ;;
 *) printf 'Usage: forest15m_commands_v1.sh smoke|check-smoke|forest|check-forest\n' >&2; exit 64 ;;
esac
test ! -e "$base/$rid"
coord="$base/command_$rid"
mkdir "$coord"
trap 'rc=$?; printf "STOP exit=%s line=%s; preserve all outputs; no retry/resume.\n" "$rc" "$LINENO" | tee "$coord/STOP.txt"; exit "$rc"' ERR
trap 'printf "INTERRUPTED; preserve all outputs.\n" | tee "$coord/INTERRUPTED.txt"; exit 130' INT TERM
printf 'Running %s. Native GPU depth rendering only; all five frozen models infer on CPU. No optimizer or ROS.\n' "$kind"
timeout --signal=TERM --kill-after=5s 30s /usr/lib/wsl/lib/nvidia-smi > "$coord/GPU_INFO.log" 2>&1
timeout --signal=TERM --kill-after=15s "${seconds}s" \
 /home/lizheng/anaconda3/envs/yopo/bin/python -B -s "$entry" --kind "$kind" --run-id "$rid" \
 > "$coord/stdout.log" 2> "$coord/stderr.log"
export CUDA_VISIBLE_DEVICES=
timeout --signal=TERM --kill-after=15s 300s \
 /home/lizheng/anaconda3/envs/yopo/bin/python -B -s "$entry" --kind "$kind" --run-id "$rid" --check \
 > "$coord/CHECK.stdout.log" 2> "$coord/CHECK.stderr.log"
cat "$coord/CHECK.stdout.log"
printf 'Evidence: %s/%s/SUMMARY.json and PER_MAP.csv, PER_CANDIDATE.csv, PER_SCENE.json\n' "$base" "$rid"
