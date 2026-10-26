#!/usr/bin/env bash
# Manual execution only. Requires an already running native YOPO simulation stack.
set -Eeuo pipefail
ROOT=/home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
OUT="$ROOT/research_outputs/rsps_forest_shadow_v1"
action="${1:?use start|stop|check}"
id="${2:?supply exclusive run ID, e.g. shadow001}"
[[ "$id" =~ ^shadow[0-9]{3}$ ]] || { echo "Expected shadowNNN" >&2; exit 2; }
RUN="$OUT/$id"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:/opt/ros/noetic/lib/python3/dist-packages"
unset ROS_NAMESPACE
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1
if [[ "$action" == check ]]; then
  export CUDA_VISIBLE_DEVICES=
  exec "$PY" -B -s "$ROOT/experiments/risk_aware_set_physics_v1/forest_shadow_v1.py" check --run "$RUN"
fi
if [[ "$action" == stop ]]; then
  [[ -f "$RUN/START.json" ]] || { echo "No started shadow run" >&2; exit 2; }
  # Only this bridge is stopped. Launcher trap stops its own native sensor.
  "$PY" -B -s - "$RUN" <<'PY'
import json,os,signal,sys
from pathlib import Path
r=Path(sys.argv[1]);pid=json.loads((r/'START.json').read_text())['pid']
cmd=Path('/proc/%d/cmdline'%pid).read_bytes()
assert b'query_minco.inference.ros_shadow' in cmd and str(r).encode() in cmd, 'PID identity changed'
os.kill(pid,signal.SIGINT)
PY
  exit 0
fi
[[ "$action" == start ]] || exit 2
[[ ! -e "$RUN" ]] || { echo "Run exists; choose a new ID. Never overwrite." >&2; exit 2; }
# No ROS executable is invoked before exclusive preparation succeeds.
CUDA_VISIBLE_DEVICES= "$PY" -B -s "$ROOT/experiments/risk_aware_set_physics_v1/forest_shadow_v1.py" prepare --run "$RUN"
exec > >(tee -a "$RUN/logs/launcher.log") 2>&1
export ROS_LOG_DIR="$RUN/roslog" ROS_HOME="$RUN/runtime" HOME="$RUN/runtime" TMPDIR="$RUN/runtime"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib:/usr/local/cuda-11.8/lib64:/usr/lib/wsl/lib
echo "Manual shadow start: $RUN"
set -x
export CUDA_VISIBLE_DEVICES=0
sensor_pid="" bridge_pid=""
cleanup() {
  for pid in "$bridge_pid" "$sensor_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
finish() {
  rc=$?
  trap - EXIT
  cleanup
  "$PY" -B -s - "$RUN" "$rc" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'LAUNCHER_EXIT.json'
if not p.exists():
 with p.open('x') as f:json.dump(dict(returncode=int(sys.argv[2])),f)
PY
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Refuse absent/native-wrong/controller-conflicting stacks before GPU startup.
CUDA_VISIBLE_DEVICES= "$PY" -B -s - "$RUN" <<'PY'
import json,sys,rosgraph
from pathlib import Path
from query_minco.inference.ros_shadow import topology_check
v=topology_check(rosgraph.Master('/query_shadow_precheck').getSystemState()[0])
with (Path(sys.argv[1])/'BEFORE_TOPOLOGY.json').open('x') as f:json.dump(v,f,indent=2)
PY
awk '/MemAvailable/ {if ($2 < 3145728) exit 1}' /proc/meminfo
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >"$RUN/logs/GPU_BEFORE.csv"
awk -F, '{if ($2-$1 < 512) exit 1}' "$RUN/logs/GPU_BEFORE.csv"
# A second native renderer uses a process-private config overlay. Baseline files remain read-only.
# Share the ROS network, but expose a read-only filesystem except this exclusive Run.
setsid timeout --signal=TERM --kill-after=5s 180s /usr/bin/bwrap --die-with-parent \
  --ro-bind / / --proc /proc --dev-bind /dev /dev --bind "$RUN" "$RUN" \
  --ro-bind "$RUN/sensor.yaml" /home/lizheng/YOPO/Simulator/src/config/config.yaml \
  --chdir /home/lizheng/YOPO/Simulator \
  /home/lizheng/YOPO/Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda \
  __name:=query_shadow_sensor /mock_map:=/query_shadow/unused_map /tf_static:=/query_shadow/tf_static \
  >"$RUN/logs/sensor.log" 2>&1 &
sensor_pid=$!
# Bridge CPU has no CUDA devices visible, independently of native rendering.
binding_sha=$("$PY" -B -s -c 'from query_minco.inference.shadow import sha,FROZEN;print(sha(FROZEN))')
setsid timeout --signal=INT --kill-after=5s 150s env CUDA_VISIBLE_DEVICES= \
  /usr/bin/bwrap --die-with-parent --ro-bind / / --proc /proc --dev /dev --bind "$RUN" "$RUN" \
  --chdir "$ROOT" "$PY" -B -s -m query_minco.inference.ros_shadow --run "$RUN" --binding-sha "$binding_sha" \
  --seconds 120 --max-frames 300 >"$RUN/logs/bridge.log" 2>&1 &
bridge_pid=$!
early=false
resource_violation=false
while kill -0 "$bridge_pid" 2>/dev/null; do
  if ! kill -0 "$sensor_pid" 2>/dev/null; then
    early=true
    kill -INT -- "-$bridge_pid" 2>/dev/null || true
    break
  fi
  if ! /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >"$RUN/runtime/gpu_current.csv"; then
    resource_violation=true
  else
    cat "$RUN/runtime/gpu_current.csv" >>"$RUN/logs/gpu_memory.csv"
    if ! awk -F, '{if ($1 > $2-256) exit 1}' "$RUN/runtime/gpu_current.csv"; then resource_violation=true; fi
  fi
  if [[ "$resource_violation" == true ]]; then
    kill -INT -- "-$bridge_pid" 2>/dev/null || true
    break
  fi
  sleep 1
done
set +e
wait "$bridge_pid"
bridge_exit=$?
set -e
"$PY" -B -s - "$RUN" "$bridge_exit" "$early" "$resource_violation" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'PROCESS_EXIT.json'
with p.open('x') as f:json.dump(dict(bridge_exit=int(sys.argv[2]),sensor_exited_early=sys.argv[3]=='true',resource_violation=sys.argv[4]=='true'),f,indent=2)
PY
cleanup
wait "$sensor_pid" || true
[[ "$bridge_exit" == 0 && "$early" == false && "$resource_violation" == false ]] || exit 1
echo "Shadow run ended; execute the separate check command after this launcher exits."
