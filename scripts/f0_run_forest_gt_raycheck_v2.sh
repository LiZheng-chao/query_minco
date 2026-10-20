#!/usr/bin/env bash
# F0 isolated GT-ray smoke test v2. CMake instead of pkg-config.
# Offline export from native Maps::forest(),
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
# PCL is normally discoverable through CMake's PCLConfig.cmake on ROS Noetic.
# A missing pkg-config .pc file is NOT proof that PCL itself is absent.
if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
  echo "BLOCKED: CMake and g++ are required." >&2
  exit 7
fi
mkdir -p "$ROOT/query_minco/outputs/forest_f0_v1/tools"
OUT="$ROOT/query_minco/outputs/forest_f0_v1/voxel_replicas/seed3_$(date -u +%Y%m%dT%H%M%SZ)"
if [ -e "$OUT" ]; then echo "BLOCKED: output exists $OUT" >&2; exit 8; fi
# Build in a completely isolated CMake directory, without reusing the stale
# Simulator/build CMake cache or touching the Simulator/ source tree.
echo "Compiling standalone native forest exporter with CMake/PCLConfig..."
BUILD_SOURCE="$ROOT/query_minco/outputs/forest_f0_v1/tools/f0_gt_cmake_source_v2"
BUILD_DIR="$ROOT/query_minco/outputs/forest_f0_v1/tools/f0_gt_cmake_build_v2"
mkdir -p "$BUILD_SOURCE" "$BUILD_DIR"
cat > "$BUILD_SOURCE/CMakeLists.txt" <<'F0_CMAKE'
cmake_minimum_required(VERSION 3.10)
project(f0_native_forest_exporter LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED F0_ROOT OR NOT EXISTS "${F0_ROOT}/Simulator/src/src/maps.cpp")
  message(FATAL_ERROR "F0_ROOT must point to a complete reach_planner checkout")
endif()
find_package(PCL REQUIRED COMPONENTS common io)
find_package(catkin REQUIRED COMPONENTS roscpp pcl_conversions)
find_package(yaml-cpp REQUIRED)
add_executable(f0_export_forest_voxels
  "${F0_ROOT}/query_minco/scripts/f0_export_forest_voxels.cpp"
  "${F0_ROOT}/Simulator/src/src/maps.cpp"
  "${F0_ROOT}/Simulator/src/src/perlinnoise.cpp")
target_include_directories(f0_export_forest_voxels PRIVATE
  "${F0_ROOT}/Simulator/src/include"
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(f0_export_forest_voxels PRIVATE
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
  yaml-cpp)
F0_CMAKE

# Using /opt/ros/noetic avoids referring back to obsolete YOPO_exec_gap overlays.
if ! cmake -S "$BUILD_SOURCE" -B "$BUILD_DIR" \
    -DF0_ROOT="$ROOT" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/ros/noetic; then
  echo "BLOCKED: CMake dependency discovery failed." >&2
  echo "Check whether libpcl-dev and ROS pcl_conversions development files are installed:" >&2
  echo '  dpkg-query -W libpcl-dev ros-noetic-pcl-conversions 2>&1' >&2
  echo '  find /usr /opt/ros/noetic -name PCLConfig.cmake 2>/dev/null | head' >&2
  echo "Do not change original Simulator build files." >&2
  exit 7
fi
if ! cmake --build "$BUILD_DIR" --target f0_export_forest_voxels -- -j2; then
  echo "BLOCKED: C++ compilation/linking failed; share the first compiler error." >&2
  exit 9
fi
COMPILED_BIN="$BUILD_DIR/f0_export_forest_voxels"
if [ ! -x "$COMPILED_BIN" ]; then
  echo "BLOCKED: compiled exporter executable not found: $COMPILED_BIN" >&2
  exit 9
fi
# Keep the previously published BIN path as a convenience while avoiding
# clobbering it: run the binary directly from the isolated build directory.
BIN="$COMPILED_BIN"

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
