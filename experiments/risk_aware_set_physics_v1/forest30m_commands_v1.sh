#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
action=$1
id=$2
[[ "$id" =~ ^forest30m_shadow[0-9]{3}$ ]] || exit 2
RUN="$ROOT/research_outputs/rsps_forest_shadow_v1/$id"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT:/opt/ros/noetic/lib/python3/dist-packages"
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1
unset ROS_NAMESPACE
if [[ "$action" == check ]];then
 CUDA_VISIBLE_DEVICES= exec "$PY" -B -s "$ROOT/experiments/risk_aware_set_physics_v1/forest30m_shadow_v1.py" check --run "$RUN"
fi
[[ "$action" == start && -f "$RUN/BINDING.json" && ! -e "$RUN/LAUNCHER_STARTED.json" ]] || exit 2
"$PY" -B -s - "$RUN" <<'PY'
import json,sys,time
from pathlib import Path
with (Path(sys.argv[1])/'LAUNCHER_STARTED.json').open('x') as f:json.dump(dict(wall_time_ns=time.time_ns()),f)
PY
exec >"$RUN/logs/launcher.log" 2>&1
export ROS_LOG_DIR="$RUN/roslog" ROS_HOME="$RUN/runtime" HOME="$RUN/runtime" TMPDIR="$RUN/runtime"
bridge_pid=""
finish(){
 rc=$?;trap - EXIT
 if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null;then kill -INT -- "-$bridge_pid" 2>/dev/null || true;wait "$bridge_pid" || true;fi
 "$PY" -B -s - "$RUN" "$rc" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'LAUNCHER_EXIT.json').open('x') as f:json.dump(dict(returncode=int(sys.argv[2])),f)
PY
 exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
awk '/MemAvailable/ {if($2<3145728)exit 1}' /proc/meminfo
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >"$RUN/logs/GPU_BEFORE.csv"
awk -F, '{if($2-$1<512)exit 1}' "$RUN/logs/GPU_BEFORE.csv"
binding_sha=$(sha256sum "$ROOT/research_outputs/rsps_forest15m_v1/preparation001/configs/FROZEN_MODELS.json" | cut -d' ' -f1)
setsid timeout --signal=INT --kill-after=5s 150s env CUDA_VISIBLE_DEVICES= /usr/bin/bwrap --die-with-parent --ro-bind / / --proc /proc --dev /dev --bind "$RUN" "$RUN" --chdir "$ROOT" \
 "$PY" -B -s -m query_minco.inference.ros_shadow30 --run "$RUN" --binding-sha "$binding_sha" --seconds 120 --max-frames 300 >"$RUN/logs/bridge.log" 2>&1 &
bridge_pid=$!
violation=false
while kill -0 "$bridge_pid" 2>/dev/null;do
 /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >"$RUN/runtime/gpu_current.csv" || violation=true
 cat "$RUN/runtime/gpu_current.csv" >>"$RUN/logs/gpu_memory.csv"
 awk -F, '{if($2-$1<256)exit 1}' "$RUN/runtime/gpu_current.csv" || violation=true
 if [[ "$violation" == true ]];then kill -INT -- "-$bridge_pid" 2>/dev/null || true;break;fi
 sleep 1
done
set +e
wait "$bridge_pid";rc=$?
set -e
"$PY" -B -s - "$RUN" "$rc" "$violation" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'PROCESS_EXIT.json').open('x') as f:json.dump(dict(bridge_exit=int(sys.argv[2]),sensor_exited_early=False,resource_violation=sys.argv[3]=='true'),f)
PY
[[ "$rc" == 0 && "$violation" == false ]]
