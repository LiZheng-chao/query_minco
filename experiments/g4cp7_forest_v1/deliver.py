from pathlib import Path
import json,hashlib
r=Path.cwd();o=r/'research_outputs/g4cp7_forest_integration_20261112_v1';e=json.loads((o/'ENTRY.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key in ['source','history']:assert all(sha(p)==h for p,h in e[key].items())
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',sources=len(e['source']),P6_artifacts=len(e['history']),scope='checked source and P6 runs;other historical output full rehash NOT_RUN'),indent=2))
reports={
'P7_FOREST_NATIVE_INTEGRATION.md':"""# Native integration
Original ~/YOPO/Simulator binary/config reused. Preregistered seed3,maze_type5,tree_dist4,resolution0.1,tree.ply,spawn0/0/2. No model-dependent scene selection.
native_preflight.py starts original renderer from Simulator cwd,actual native remaps,owned HELD Host and independent /mock_map observer. No ARM or Controller.
Native source/config/resource identities in NATIVE_SOURCE_BINDING.json;check-only PASS. Actual native forest source/spawn qualification NOT_RUN.
native_truth.py uses closed-voxel diagonal reserve and finite-domain rejection;GT goes only to evaluation,never route selection.
Observer is PROTOTYPE:2M point cap,heavy cloud processing still under callback lock,source map-content epoch not fully bound. Actual false/missed/late collision qualification NOT_RUN.
Preflight does not consume truth heartbeat as motion authority. Its truth RESULT may FAIL on shutdown because STOP_CONFIRMED is not yet supplied;do not confuse clean sensor capture with truth qualification.
Forest execution owner is not yet assembled. This is incomplete integration,not a completed forest loop.
""",
'P7_RUNTIME_CODE_DEFECTS.md':"""# P6 actual review and defects
P6 A6 PASS,B runtime/cleanup PASS with exact source binding.
B progress0.394093m,terminal error0.005910m,max tracking error0.019628m,max speed0.413676m/s.
432 exact final SO3/Plant ACK matches,forces agree<2e-5,median/max ACK latency8.685/46.709ms.
Full streams/peaks in P6_RECEIPT.json. Sampled jerk FD is diagnostic.
Execution uses genuine producer-stamped native IMU;no Shadow difference substitute.

NO_EXACT_POSE186 historical aggregate lacks per-rejection callback traces. Root mechanism:latest-depth lookup without waiting for reversed callback order.
New bounded Pairing handles either order,exact stamp and0.2s freshness;5 tests PASS;P6 actual arrival metadata replay104/104 matched.
Not proof that all186 historical events are resolved;new real Shadow NOT_RUN.
CONTINUOUS_EXECUTION_LIMIT130:all130 classified HISTORICAL_INPUT_UNAVAILABLE. Old failed state/acceleration/knots were not stored. No guessed per-limit cause or threshold change.
P7 Shadow now stores per-rejection source/receipt stamps,state,acceleration and goal for reproduction.
P5 rolling pose+.4 target now fixed once in P7 Shadow. P6 execution was already fixed.
P6 code/results unchanged.
""",
'P7_SENSOR_AND_CONTROL_CONTRACT.md':"""# Contracts
Native topics /sim/odom,/depth_image,/camera_info,/mock_map;world pose,physical render body-origin and qxyzw,not display TF offset.
Depth160x90 32FC1 optical-Z,K80/80/80/45,exact source stamp. Pairing bounded512 poses/4pending/4ready,<=0.2s;no nearest-pose substitution.
P7 Shadow candidate-only,FD acceleration explicitly shadow-only. Fresh-pair candidate output sensor-rate,not100Hz execution claim.
P6 actual execution original3-piece MINCO,T2,source IMU,1/8/100 limits and original position nodelet preserved.
P7 native truth observer currently only emits heartbeat;authoritative collision-to-Plant stop wiring remains incomplete.
No UNKNOWN->FREE or GT route oracle.
P4 strict certification remains FAIL. P5 birth engineering premise is separate;finite birth-volume permission is pending researcher answer,not assumed.
""",
'P7_E1_E2_READINESS.md':"""# Specific remaining blockers
E1-A/B/E2 NOT_READY,actual NOT_RUN.
1.Run new native seed3 HELD source/spawn qualification,not a repeat of frozen3101 E0.
2.Specify birth point versus bounded launch volume. A point alone cannot prove entire0.30m tracking tube. No wider volume silently authorized.
3.Assemble forest owner using native Simulator cwd/remaps,require fresh truth PASS BEFORE ARM and route truth failure/stale to exact owned Plant stop.
4.Move cloud ingestion outside odom callback lock;bind actual map SHA/scene epoch and qualify collision-stop latency under new identity.
5.Integrate sensor-space certificate into both reference/admission;existing full-ball single-view check still rejects near field. Do not bypass it.
6.Add>=1 actual replanning/switch to fixed E1 target;then qualify affected native sensor/truth/admission paths. P6 open PASS does not qualify new forest.
Minimum subsequent delivery is the wired owner/reference/admission plus independently tested observer,not another dynamics/controller.
E1-A launch-region straight-through tracking must be reported separately from E1-B actual obstacle avoidance.
E2>=20 replans,>=2m,<=1m/s. No motion command offered before these prerequisites.
""",
'P7_MANUAL_FOREST_COMMANDS.md':"""# Native forest HELD preflight only
New seed3 source/spawn check,not angular-coverage proof. No ARM/Controller.
~~~bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
P7_PREFLIGHT="research_outputs/p7_native_preflight_$(date +%Y%m%d_%H%M%S)_$$"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$P7_PREFLIGHT/process" --cwd "$PWD" --timeout-s 60 -- "$PY" -B experiments/g4cp7_forest_v1/native_preflight.py --allow-sensors RESEARCHER_MANUAL_E0_SENSORS_NO_ARM --output "$P7_PREFLIGHT/run"
echo "$P7_PREFLIGHT"
~~~
Inspect E0_RESULT/OWNER_RESULT separately from truth_data/HEARTBEAT and RESULT. Truth size/coverage/shutdown failures must not be ignored.
Exit3=unproved initial observed tube,not motion permission.
Private master11361 refuses occupied port. Actual run NOT_RUN;only check-only passed.
No E1 motion CLI is claimed ready.
""",
'P7_FINAL_GO_NO_GO.md':"""# Final
P6_A_PHYSICAL=PASS_REVIEWED
P6_B_PHYSICAL=PASS_REVIEWED
P6_IDENTITY=PASS
P7_PAIRING_STATIC=PASS5
P7_PAIRING_REPLAY=104/104
HISTORICAL_130_LIMIT_CLASSIFICATION=UNRESOLVED_INPUTS_NOT_SAVED
NATIVE_FOREST_HELD_CODE=IMPLEMENTED
NATIVE_FOREST_LIVE=NOT_RUN
NATIVE_TRUTH_OBSERVER=PROTOTYPE_NOT_QUALIFIED
FOREST_EXECUTION_OWNER=INCOMPLETE
E1_A_E1_B_E2=NOT_RUN_NO_GO
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
HISTORICAL_PROTECTION=PASS
Finite birth volume question pending;no authorization assumed.
"""
}
for name,body in reports.items():(o/name).write_text(body)
for p in (r/'ros/g4cp7_forest_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
files=[p for a in e['allowed']+[str(o.relative_to(r))] for p in (r/a).rglob('*') if p.is_file()]
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(r)):sha(p) for p in files},indent=2))
print('6 reports;P6 verified;P7 forest execution incomplete;protected history PASS')
