from pathlib import Path
import json,hashlib
r=Path.cwd();o=r/'research_outputs/g4cp7b_fix_20261112_v1';e=json.loads((o/'ENTRY.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key in ['source','history']:assert all(sha(p)==h for p,h in e[key].items())
b=json.loads((o/'SOURCE_BINDING_v2.json').read_text());assert all(sha(p)==h for p,h in b.items())
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',old_source_files=len(e['source']),failed_run_files=len(e['history']),scope='checked mainline/YOPO source plus original failed run;other historical outputs full rehash NOT_RUN,no writes there'),indent=2))
reports={
'P7B_DEPTH_STALE_ROOT_CAUSE.md':"""# Failure root cause and evidence limits
Actual run p7b_native_safety_20261112_122048_479 failed depth_STALE before injection,with0 integration. Plant stop/cleanup succeeded. Not a request_timeout PASS.
Accepted depth25frames:source last1794457258.6507668,receipt ROS1794457258.6548376;fault ROS1794457258.859631. Thus last accepted depth was~208.86ms old.
Earlier received frames have millisecond age;FAILURE_TIMELINE.json retains every accepted source gap/age.
Renderer stdout says51,716,825points,average depth5.40283ms,lidar3.21903ms;one warning50.55ms. These averages do not bound all callback/serialization delays.
Old logs lack per-frame source start/publish timestamps. Exact source-vs-transport-vs-consumer attribution of the missing interval is UNVERIFIED. GPU utilization and original baseline same-scene comparison NOT_RUN.

Confirmed code issue:one ROS spin processes both odom render callbacks and timerMapCallback;the latter publishes full giant PointCloud2 whenever subscribed.
Truth subscriber forces that publication even though it rejects count>2M. Original constructor also converts51Mpoints to PointCloud2. Serializing/transferring this data can block/starve callbacks and resource buffers.
This is a specific high-risk mechanism,not proven sole cause of the exact gap. No claim it is experimentally eliminated yet.
Fix isolated overlay:no full PointCloud2 conversion/publication path;export complete native GPU occupancy once at startup. Original native CUDA map/kernel/library/config unchanged.
Telemetry now records source stamp,renderer odom receipt,render start/end and publish time. Probe records depth callback receipt and callback work.
Source->renderer receipt approximates producer queue/transport;render includes native readback;publish->consumer uses timestamp after publish call and may be slightly negative from ordering,it is not pure network latency.
Latest non-ARM S test reports all phase P50/P95/P99/max plus source/receipt gaps and GPU samples. Real S test NOT_RUN by Codex.
0.2s watchdog unchanged. Renderer algorithm,depth_fps33,lidar settings,forest seed/density unchanged.
""",
'P7B_NATIVE_TRUTH_FIX.md':"""# Complete native-grid truth evaluation
Old node rejects51M>2M before worker and may not receive full giant ROS message before failure. Its final PASS only meant journal/worker cleanup;no qualified map/heartbeat existed.
New overlay exports actual native GPU counters with original predicate counter>occupy_threshold into u8 occupancy,in chunks1Mints (bounded additional~5MB),not a raw-point subsample.
All native occupied cells are retained. CPU/native map construction consumes all original valid points unchanged.
Grid metadata preserves native float origin,resolution,shape,threshold;memory cap128Mcells fails explicitly rather than drops information.
Original native mapQuery z<=0 floor handling is reproduced in evaluator. Query outside finite base grid rejects;it does not silently assume mirrored/free infinity.
Python readonly memmap hashes whole exported grid/metadata once,checks file size/mtime for mutation,queries bounded local0.8m volume against exact closed occupied voxel boxes. Unqueried cells are at least0.8m away.
No KD-tree or51M Python tuple materialization and no /mock_map subscription.
Files native_grid.u8/json are runtime-only evaluator inputs,never online route geometry. Birth authorization remains bounded0.60m and uses separate GT check.
Process RESULT separates PROCESS_CLEANUP_PASS,MAP_QUALIFIED,BIRTH_CLEARANCE_PASS,TRUTH_MONITOR_ACTIVE and PHYSICAL_SAFETY_PASS (not inferred from cleanup).
Owner S requires fresh truth/map/birth heartbeat;C/D require valid new prerequisites. Changed/dead/stale map or monitor rejects ARM or stops owned Plant.
Raw geometry export is from the same GridMap GPU allocation used by native raycast,no second map generator/controller.
Native world is declared static during an episode;external unseen dynamic scene edits are outside contract. Runtime file mutation rejects.
""",
'P7B_CODE_DIFF_AND_REGRESSION.md':"""# Code and build
All additions versioned under native_grid_truth_v1,ros/experiments/tests g4cp7b_fix_v1. Old YOPO/P7b source and failed-run logs unchanged.
Renderer header overlay adds inline export method,not data fields;native GridMap layout and compiled libraycast_cuda remain unchanged.
RENDERER_DIFF_v2.patch,build_v2/BUILD_IDENTITY and compile/link process logs bind exact source/binary/config/library.
Build_v2 COMPILE/LINK PASS,never executed simulation. Initial build failed because timing code was inserted in constructor as well as render function;corrected insertion,failed logs retained.
6 final static tests PASS in tests_v2 and logs/qualify_v2:occupied boundary,last voxel,floor,outside domain,mutation,free-volume bound,no-ARM S guard,and native render reuse assertions.
Installed baseline math/control untouched. End-to-end old/new GPU depth numerical replay NOT_RUN;requires manual native source test.
Sensor probe uses strict increasing source stamps,source/receipt gap<=0.2s,age<=0.2s,>=500frames over20s,rate>=25Hz,>=90percent matched telemetry and calibration contract.
Current limits unchanged,not a claim of exact33Hz hardware capability.
S passes reference stageC only to provide fresh HELD hold requests,owner never sends prepare/arm;integration must remain0.
""",
'P7B_FOREST_SAFETY_RESULTS.md':"""# Actual versus unexecuted
Original failed run:depth_STALE,0integration,0fault injection;Plant terminated,cleanup PASS. Truth map/birth/active NOT_QUALIFIED.
New renderer build PASS;6 static tests PASS;source binding/no-ROS S preflight PASS.
New S real forest stability NOT_RUN.
New C seven real faults NOT_RUN.
New D/E1-A real0.28m tracking/replanning NOT_RUN.
PHYSICAL_SAFETY_PASS is not assigned from any static/cleanup result.
S now a required exact-binding prerequisite to C;C7 a required exact-binding prerequisite to D.
No old P7b/P6 pass reused as current scene qualification.
""",
'P7B_E1A_E1B_READINESS.md':"""# Gates
S ready for researcher non-ARM native run. It includes real original forest generation,complete occupancy export,real depth/timing and independent truth heartbeat.
If generation/export/map qualification cannot finish within current owner startup8s,entry refuses;do not extend watchdog or reinterpret failure.
C only after current S PASS. D only after current C7 PASS.
E1-A uses existing original3-piece MINCO,0.28m fixed target,one second plan at1.8s,whole tube inside approved0.60m birth sphere.
Actual terrain birth clearance remains unknown until S. Insufficient clearance rejects without reseeding.
E1-B depth-based obstacle avoidance beyond launch ball remains NOT_IMPLEMENTED/NO_GO;not confused with straight launch-region tracking.
E2>=20replans,>=2m remains NOT_IMPLEMENTED/NO_GO. No manual command offered.
Exact remaining steps beyond E1-A:integrate bounded depth free-space history/candidate route admission outside birth volume and qualify sensor uncertainty/UNKNOWN behavior;then actual same-version switches/collision-stop tests. No new simulator proposed.
""",
'P7B_FINAL_GO_NO_GO.md':"""# Final status
NATIVE_RENDERER_OVERLAY_BUILD=PASS
COMPLETE_NATIVE_GRID_TRUTH_STATIC=PASS
STATIC_TESTS=PASS6
WATCHDOG_SECONDS=0.2_UNCHANGED
REAL_STABILITY_S=NOT_RUN
REAL_SEVEN_FAULT_C=NOT_RUN
E1_A=CONDITIONAL_ON_S_AND_C_NOT_RUN
E1_B=NO_GO_NOT_IMPLEMENTED
E2=NO_GO_NOT_IMPLEMENTED
PHYSICAL_SAFETY_PASS=NOT_RUN
P4_STRICT_CERTIFICATION=UNCHANGED_FAIL
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
HISTORICAL_PROTECTION=PASS
Do not claim depth_STALE eliminated until actual source/receiver timing passes. Source fix removes giant-map ROS load;remaining GPU/queue/consumer latency is measurable in S.
Active execution identity SOURCE_BINDING_v2.json. Prior build/log/binding snapshots retained,not used as current pass.
"""
}
reports['P7B_MANUAL_COMMANDS.md']="""# Manual phases,never automatically chained
All commands require researcher execution. Private master11367 refuses occupied port.
Fixed original seed3 config;no controller/gain/network/physics update. Output paths exclusive.
S cannot emit PREPARE/ARM,requires zero integration. Native grid export may be tens of MB;do not commit runtime files.

1.Non-ARM20s sensor/truth stability:
~~~bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FIX_S="research_outputs/p7b_fix_S_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$FIX_S/process" --cwd "$PWD" --timeout-s 120 -- "$PY" -B experiments/g4cp7b_fix_v1/manual.py --stage S --allow-physics RESEARCHER_MANUAL_S_ONLY --output "$FIX_S/run"
echo "$FIX_S/run/RESULT.json"
~~~
Review run/RESULT.json and none/sensor_probe/RESULT.json,GPU_SAMPLES.json,truth_data/HEARTBEAT.json,truth_data/RESULT.json plus native_grid metadata and child exits.
Require S PASS,0integration,map/birth/active qualification. Missing GPU utility is reported in GPU samples,not invented utilization.
If S fails,return full output before proceeding.

2.After researcher review of S PASS,new7fault campaign:
~~~bash
cd /home/lizheng/query_minco_mainline
test -n "$FIX_S" || { echo "Set FIX_S to successful new S directory"; exit 2; }
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FIX_C="research_outputs/p7b_fix_C_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$FIX_C/process" --cwd "$PWD" --timeout-s 600 -- "$PY" -B experiments/g4cp7b_fix_v1/manual.py --stage C --allow-physics RESEARCHER_MANUAL_C_ONLY --prior "$FIX_S/run/RESULT.json" --output "$FIX_C/run"
echo "$FIX_C/run/RESULT.json"
~~~

3.Only after current C7 final PASS,manual E1-A:
~~~bash
cd /home/lizheng/query_minco_mainline
test -n "$FIX_C" || { echo "Set FIX_C to successful new C7 directory"; exit 2; }
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FIX_D="research_outputs/p7b_fix_E1A_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$FIX_D/process" --cwd "$PWD" --timeout-s 120 -- "$PY" -B experiments/g4cp7b_fix_v1/manual.py --stage D --allow-physics RESEARCHER_MANUAL_D_ONLY --prior "$FIX_C/run/RESULT.json" --output "$FIX_D/run"
echo "$FIX_D/run/RESULT.json"
~~~
Do not change source/config/binary between phases. Old failed/new wrong-identity results are rejected. No E1-B/E2 command is claimed ready.
"""
for name,text in reports.items():(o/name).write_text(text)
files=[p for a in e['allowed']+[str(o.relative_to(r))] for p in (r/a).rglob('*') if p.is_file()]
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(r)):sha(p) for p in files},indent=2))
print('7 reports;build and static tests PASS;manual S/C7/D ready;real runs NOT_RUN')
