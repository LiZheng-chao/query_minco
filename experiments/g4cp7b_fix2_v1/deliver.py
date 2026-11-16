from pathlib import Path
import json,hashlib
r=Path.cwd();o=r/'research_outputs/g4cp7b_fix2_20261112_v1';e=json.loads((o/'ENTRY.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key in ['source','history']:assert all(sha(p)==h for p,h in e[key].items())
b=json.loads((o/'SOURCE_BINDING_v3.json').read_text());assert all(sha(p)==h for p,h in b.items())
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',protected_source=len(e['source']),protected_original_run_files=len(e['history']),scope='mainline/YOPO source and existing S/C/E1A evidence;other outputs full rehash NOT_RUN'),indent=2))
reports={
'FIX2_DEPTH_RELAY_ROOT_CAUSE.md':"""# Depth evidence and minimal fix
Original Probe204,Supervisor182;22missing accepted stamps. Every missing stamp has corresponding Supervisor odom.
All22 corresponding odom receipts occur AFTER independent Probe depth receipt (ORDER_DIAGNOSTIC.json);early examples19-23ms later.
Old Supervisor silently returns on s not in poses. This is a confirmed arrival-order handling defect and consistent with these losses.
Old Relay input/output and Supervisor callback-entry were not logged,so exact historical hop of each loss is UNVERIFIABLE;do not claim recovered packet traces.
Small odom TCP buffering/Nagle is a plausible contributor,not a measured exclusive cause.

New Relay logs source/output stamp,receipt ROS+monotonic,lock wait,processing,publish decision,connection count and configured queue.
New Supervisor logs callback arrival before lock,lock acquisition,exact-pair processing,accepted/rejected reason and pending counts.
Native Renderer/Probe timing remains from prior compiled overlay.
ROS internal queue occupancy is not exposed:reported explicitly unavailable,not guessed.
Depth/Odom pairing handles both callback orders,bounded512poses/4depths;source age<=0.2s;no nearest pose,no receipt-time refresh,no rollback acceptance.
TCP_NODELAY enabled for new relay/supervisor connections,depth buffer1MiB;queue sizes remain bounded.
Relay shutdown catches ROSInterruptException so orderly stop cannot become spurious exit1.
Watchdog0.2s unchanged. Real Fix2 S NOT_RUN.
analyze_run.py can compare every new source stamp across all recorded layers without modifying raw logs.
""",
'FIX2_BIRTH_POINT_QUALIFICATION.md':"""# Deterministic birth protocol,registered before search
Original seed3,maze5,tree_dist4,resolution0.1 and full exported grid unchanged.
Protocol:XY[-2,2] at0.25m spacing,Z2;increasing squared distance to origin,then x,y ascending. First conservative clearance>=0.60m selected.
No planner,policy,MINCO optimization or successful trajectory screening used.
Old[0,0,2] clearance0.3428449337m retained as failed candidate.
First qualified candidate,index8:[0.25,-0.25,2],lower bound0.6218632332m.
Full record BIRTH_SELECTION.json contains all8 evaluated candidates,grid SHA+metadata SHA.
Grid shape650x652x144=61027200,C-order x*ny*nz+y*nz+z.
Origin[-32.8320579529,-32.8835983276,-0.0577124991],resolution0.10000000149.
Native occupancy predicate and closed boxes retained,including virtual floor and1e-5 numerical reserve.
Readonly offline qualification is not new physical PASS.
Plant init XYZ and native position-nodelet init XYZ both use selected sealed pose;Renderer consumes actual Plant odom.
Birth contract center and map_id match the sealed selection. New S and pre-ARM enforce live map_id equality;different map rejects.
Only0.60m engineering birth ball is authorized;GT grid not supplied to online route choice. P4 strict proof still FAIL.
""",
'FIX2_SOURCE_CODE_CHANGES.md':"""# Versioned implementation
New supervisor/reference/relay/controller/owner under g4cp7b_fix2_v1;pairing,selected-birth contract and depth-planning adapter under forest_fix2_v1.
No prior source or result overwritten. Native renderer/physics/controller algorithms and compiled renderer binary reused,not rebuilt or replaced in this round.
Owner switches only sensor/control transport integration,selected init state and qualified birth map check.
Original renderer/config/tree SHA remain bound by SOURCE_BINDING_v3.json. Initial binding snapshots retained but not active.
E1-B development DepthPlanner reuses conservative depth ledger,frozen A* route checks,original three-piece MINCO and full sensor-space certificate.
It has no GT geometry input,network or fixed left/right anchors. It is not wired as an execution mode and may reject near-field UNKNOWN.
No claim of actual obstacle avoidance from this backend or birth-ball straight tracking.
""",
'FIX2_RUNTIME_AND_SAFETY_TESTS.md':"""# Tests and status
Final logs/qualify_v2:exit0,9 STATIC_PASS,Python compile and no-ROS S preflight PASS.
Tests:old early-drop reproduction,both callback orders,expired/no-nearest,queue bound,rollback rejection,birth/controller registration,Relay cleanup/tracing,E1-B no-oracle/UNKNOWN rejection.
No actual real forest S,C,E1A,E1B,E2 executed by Codex.
Old S is FAIL;old C/E1A are prerequisite rejection,not failed physical flight.
Full real delay decomposition remains pending new S. Keep all raw logs on failure.
Birth offline qualified,live same-map/pose qualification pending. No lower radius/threshold.
""",
'FIX2_FOREST_E1_READINESS.md':"""# Gates
S only is released now. Must complete20s with actual Depth,stable exact-pair consumption,qualified full map and selected birth clearance.
C7 remains code-gated on matching successful new S. E1-A remains code-gated on matching C7;commands intentionally withheld until actual prerequisite PASS.
Old failed C/E1A rejection is correct and not relabeled flight failure.
E1-A original0.28m fixed target and>=2actual plans/one switch retained;<=1m/s and original dynamics limits.
E1-B backend implemented for further offline/background qualification,but execution integration and actual bypass/avoidance tests incomplete.
It cannot be claimed obstacle avoidance merely because it rejects UNKNOWN or generates a candidate.
E2 not ready. Remaining E1-B work is shared sensor-space proof in reference+admission,switch timing and same-version stop qualification beyond birth volume.
No new simulator proposed;all future motion researcher-manual only.
""",
'FIX2_FINAL_GO_NO_GO.md':"""# Decision
DEPTH_PAIRING_STATIC=PASS
EXACT_HISTORICAL_RELAY_HOP=UNRECORDED_UNVERIFIABLE
BIRTH_OFFLINE_QUALIFIED=YES
SELECTED_BIRTH=[0.25,-0.25,2]
CLEARANCE_LOWER_BOUND_M=0.6218632332444738
REAL_FIX2_S=NOT_RUN
C7=NOT_RUN_GATED
E1_A=NOT_RUN_GATED
E1_B=DEVELOPMENT_BACKEND_ONLY_NOT_READY
E2=NO_GO
STATIC_TESTS=PASS9
HISTORICAL_PROTECTION=PASS
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
Do not claim consumer loss eliminated before real S trace confirms it.
"""
}
reports['FIX2_MANUAL_COMMANDS.md']="""# Released:Fix2 S only
Keep this shell variable for returning the exact run path. No PREPARE/ARM emitted by S;require integration0.
Original renderer is reused with original config,only birth pose/consumer code changes. Map must match frozen snapshot.
~~~bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FIX2_S="research_outputs/p7b_fix2_S_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$FIX2_S/process" --cwd "$PWD" --timeout-s 120 -- "$PY" -B experiments/g4cp7b_fix2_v1/manual.py --stage S --allow-physics RESEARCHER_MANUAL_S_ONLY --output "$FIX2_S/run"
echo "$FIX2_S/run/RESULT.json"
~~~
Inspect whole RESULT plus none/sensor_probe/RESULT,truth_data/RESULT,relay_data/EVENTS,supervisor_data/EVENTS,all child exit logs.
Analyze actual new trace to a NEW exclusive report directory:
~~~bash
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
"$PY" -B experiments/g4cp7b_fix2_v1/analyze_run.py --run "$FIX2_S/run/none" --output "research_outputs/fix2_chain_$(date +%Y%m%d_%H%M%S)_$$"
~~~
Return raw run even on failure. The analyzer is read-only toward raw evidence.
Current active identity SOURCE_BINDING_v3.json.
Private ROS master11371 refuses occupied port;never kill unrelated ROS.
C/E1-A manual commands will be released only after reviewed S PASS. Do not reuse prior failed S or prior C.
"""
for name,body in reports.items():(o/name).write_text(body)
files=[p for folder in e['allowed']+[str(o.relative_to(r))] for p in (r/folder).rglob('*') if p.is_file()]
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(r)):sha(p) for p in files},indent=2))
print('7 reports;offline birth qualified;9 static tests;S only released')
