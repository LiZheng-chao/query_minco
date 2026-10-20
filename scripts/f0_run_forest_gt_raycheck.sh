#!/usr/bin/env bash
# F0 isolated GT-ray smoke test. Offline export from native Maps::forest(),
# not from /mock_map. Never modifies original Simulator/Controller/YOPO.
set -eo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash query_minco/scripts/f0_run_forest_gt_raycheck.sh /absolute/path/to/preflight_observations_v2/pair_<timestamp>" >&2
  exit 2
fi
ROOT="${HOME}/reach_planner"
PAIR_DIR="$(realpath "$1")"
SCRIPTS="$ROOT/query_minco/scripts"
CONFIG="$ROOT/query_minco/configs/forest_f0_camera_128x160_v1.yaml"
EXPECTED_SHA="45f3184f236cc11c0739fe689f84c4f1b0e58f1ada8ef11fd18ba25734e5bf81"
for f in "$CONFIG" "$PAIR_DIR/metadata.json" "$PAIR_DIR/observation_pair.npz" \
         "$SCRIPTS/f0_export_forest_voxels.cpp" "$SCRIPTS/f0_validate_forest_voxels.py"; do
  if [ ! -f "$f" ]; then echo "Missing required file: $f" >&2; exit 3; fi
done
CURRENT_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
if [ "$CURRENT_SHA" != "$EXPECTED_SHA" ]; then
  echo "BLOCKED: F0 YAML hash changed: $CURRENT_SHA" >&2
  exit 4
fi
# Protect host memory while native generator materializes ~52M PCL points.
if pgrep -f '[s]ensor_simulator_cuda' >/dev/null; then
  echo "BLOCKED: stop live forest sensor_simulator_cuda with Ctrl+C before offline export." >&2
  exit 5
fi
AVAILABLE_KB="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
if [ -z "$AVAILABLE_KB" ] || [ "$AVAILABLE_KB" -lt 3145728 ]; then
  echo "BLOCKED: need >=3 GiB host MemAvailable (currently ${AVAILABLE_KB:-unknown} KB)." >&2
  exit 6
fi
if ! pkg-config --exists pcl_common pcl_io; then
  echo "BLOCKED: PCL development pkg-config packages not found (pcl_common/pcl_io)." >&2
  exit 7
fi
mkdir -p "$ROOT/query_minco/outputs/forest_f0_v1/tools"
OUT="$ROOT/query_minco/outputs/forest_f0_v1/voxel_replicas/seed3_$(date -u +%Y%m%dT%H%M%SZ)"
BIN="$ROOT/query_minco/outputs/forest_f0_v1/tools/f0_export_forest_voxels"
if [ -e "$OUT" ]; then echo "BLOCKED: output exists $OUT" >&2; exit 8; fi
# Compile from original native map generator in the CURRENT workspace.
echo "Compiling standalone native forest exporter..."
g++ -std=c++17 -O2 -pthread \
  -I"$ROOT/Simulator/src/include" -I/opt/ros/noetic/include \
  "$SCRIPTS/f0_export_forest_voxels.cpp" \
  "$ROOT/Simulator/src/src/maps.cpp" \
  "$ROOT/Simulator/src/src/perlinnoise.cpp" \
  $(pkg-config --cflags --libs pcl_common pcl_io) \
  -L/opt/ros/noetic/lib -Wl,-rpath,/opt/ros/noetic/lib \
  -lrosconsole -lrostime -lroscpp -lyaml-cpp \
  -o "$BIN"

# Generator produces a compact grid; never persists full 52M-point cloud.
echo "Exporting voxel replica..."
"$BIN" "$CONFIG" "$OUT"
echo "Cross-checking voxel map with captured depth..."
/usr/bin/python3 "$SCRIPTS/f0_validate_forest_voxels.py" \
  --map-dir "$OUT" \
  --observation-npz "$PAIR_DIR/observation_pair.npz" \
  --observation-metadata "$PAIR_DIR/metadata.json" \
  --expected-config-sha256 "$EXPECTED_SHA" \
  --expected-point-count 51716825

echo "GT-ray smoke completed. Files are under: $OUT"
