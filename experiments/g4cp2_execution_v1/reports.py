"""Assemble truthful static/runtime distinctions;never launch ROS/ARM."""
from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1'
def doc(name,text):
 with (O/name).open('x') as f:f.write(text.strip()+'\n')
def main():
 entry=json.loads((O/'ENTRY.json').read_text());protection=json.loads((O/'HISTORICAL_PROTECTION.json').read_text());assert protection['status']=='PASS'
 new=[p for base in entry['allowed_new'][:-1] for p in (R/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 binding=dict(entry['source_before']);binding.update({str(p):sha(p) for p in new})
 for p in [O/'SCENES.json',O/'LEGACY_B_PHYSICAL_WINDOW.json',O/'FOREST_SENSOR_ROUTE_QUALIFICATION.json',
  O/'forest_sensor_build/native_forest_sensor',O/'forest_sensor_build/config.yaml',O/'forest_sensor_build/forest.ply',
  R/'research_outputs/g4cp1_startup_control_20261111_v1/build_equilibrium_v3/gated_host',
  R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/native_open_sensor',
  R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/config.yaml',
  R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/floor_only.ply',
  R.parent/'YOPO/Controller/devel/lib/libquadrotor_dynamics.so',R.parent/'YOPO/Controller/devel/lib/libso3_control_nodelet.so',
  R.parent/'YOPO/Controller/src/so3_control/config/gains_hummingbird.yaml',R.parent/'YOPO/Controller/src/so3_control/config/corrections_hummingbird.yaml',
  R.parent/'YOPO/Simulator/src/src/test_simulator_cuda.cpp',
  R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101/COMPACT_COUNTS.i32',
  R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101/COMPACT_GRID.json']:
  binding[str(p)]=sha(p)
 write(O/'SOURCE_BINDING.json',binding)
 doc('G4C_P2_RUNTIME_FIX_AUDIT.md',"""
# P2 runtime defect audit
PASS - current HEAD4a2d962f599550d0e48f1b9dbfb127565d2c1594 and real untracked sources audited. Old runs,YOPO,G4B,P1,Hard,watchdog values,gains,model weights and training code are unchanged. All additions are in the ENTRY whitelist.
PASS - read/replayed complete P1 A/B event streams and actual child results. No historical result file was corrected in place.

| historical evidence | interpretation |
|---|---|
|A5 distinct ACK,steps0|actual held handshake PASS|
|Aprepare exit1,depth_STALE,then I/O on closed file|cleanup FAIL;parent PASS omitted final child reconciliation|
|Brecorded5048 steps,event tail5058|actual integrated physical window PASS,not ground start|
|Bprepare exit-9|cleanup FAIL,despite stable physical window|

P1 recorded no absolute stop-monotonic timestamp. Source order and event cessation support termination-induced staleness in A,but the exact historical stop/fault boundary cannot be retrospectively certified. We do not erase the fault or fabricate a stop timestamp.
B whole available integrated-event replay:speed max1.2703544e-7m/s,acceleration max7.0508264e-7m/s^2,drift max8.6106938e-8m. Complete stream counts/gaps/ages are in P1_FULL_WINDOW_REPLAY.json. Stored sensor sampling is100Hz/20Hz;unlogged first-ms integration details are not reconstructed. LEGACY_B_PHYSICAL_WINDOW.json authorizes only a reviewed physical-window prerequisite,explicitly retaining cleanup FAIL.

## Version4 repair
- lifecycle.py:RUNNING -> QUALIFIED(provisional) -> TERMINATING -> TERMINATED. Final status combines runtime qualification,expected versus unexpected faults,actual terminal plant state,all child exit codes and node teardown files.
- Journal.callback holds a shared reentrant lock. Closing first fences late callbacks,unregisters subscribers,drains in-flight work,then closes under the lock. Late callbacks cannot write a closed file.
- Watchdogs remain live while RUNNING/QUALIFIED. Parent freezes/kills/waits for the exclusively owned native Plant FIRST,then writes STOP_CONFIRMED.json with absolute monotonic/ROS clocks. Only this confirmation allows Python nodes to stop health checking and close normally.
- An actual pre-stop fault remains in event/metric records and prevents D success. Expected C faults are labeled,not erased. A stage may have runtime PASS and cleanup FAIL;overall status is FAIL.
- Full-window metrics include accepted source/receipt gaps,tail freshness,duplicates,rollback,max measured v/a,position drift,tracking error and diagnostic finite-difference jerk. Final heartbeat alone is never the denominator.
- Spawn errors,guard exceptions,unexpected child exits and surviving owned descendants produce explicit FAIL. Controller fault injection kills only the PID-start-time-bound owned tree;no global pkill.
- Port probe uses SO_REUSEADDR consistently with roscore and refuses a live listener. XMLRPC master PID must belong to this run's owned process tree;we do not attach to an unrelated ROS master.

PASS - static race,duplicate,latency,stream cessation,false-PASS,abnormal exit and teardown-failure tests. NOT_RUN - actual v4 ROS cleanup;real C/D process consumption. No new network or optimizer updates.
""")
 doc('G4C_P2_C_FAULT_INJECTION.md',"""
# C real-physics fault campaign
PASS static - source,interfaces,fault schedules and gate predicates implemented. NOT_RUN physical - Codex did not launch ROS or ARM.

manual.py --stage C starts6 independent episodes,one owned master,and a fresh native/controller/sensor/reference/supervisor/relay/truth chain in each episode.
Cases:request_timeout,depth_loss,odom_rollback,nan_control,reference_exit,controller_exit.
Fault onset file is created only after real ARMED phase,positive native integration and at least1s of physical feedback. SIGKILL targets are recorded with PID start identities. Reference timeout stops validated requests;Depth loss drops actual messages;Odom rollback preserves payload but injects an explicit invalid stamp;NaN request is intentionally rejected before controller publication.
No mock Odom or prerecorded state source is used in manual runs. Reference and supervisor are separate processes;reference death cannot be hidden by an internal HOLD generator.

Expected detection reasons are case-specific. A generic abort,truth failure or missing process cannot count as successful injection. Require>=500 native steps and at least5 accepted pre-fault samples per required stream. Record onset,first detection/latch,stop request,confirmed process freeze/exit and all event counters.
Keep original.2s watchdog. Additional total fault-to-confirmed-stop acceptance is<=.35s to include measured scheduler/owner overhead;this does not extend command/sensor freshness. Native watchdog exit3 is expected only for a matching C fault;all other abnormal exits remain FAIL.
Plant is frozen/terminated,not merely deprived of Path messages. This is simulation fail-stop,not a real-aircraft braking certificate.
Node journals then close gracefully;unexpected cleanup failures independently fail the episode. Stop at first failed case. C_CAMPAIGN PASS requires all6 episode SHA-bound final results and a clean master exit. D rejects partial/failed/stale-source C results.

Raw files per episode:FAULT.json,optional FAULT_PROCESS_TARGETS.json,STOP_CONFIRMED.json,supervisor_data/EVENTS.jsonl,reference_data records,truth_data records,all child PROCESS_START/RESULT and stdout/stderr,RESULT.json.
""")
 doc('G4C_P2_D_REAL_MINCO_TRACKING.md',"""
# D actual0.4m MINCO tracking
PASS static - executable source and numerical/input contracts implemented. NOT_RUN physical - no v4 movement was executed.

D requires exact-source C_CAMPAIGN PASS with all6 referenced episode hashes. It starts a fresh equilibrium_hover native plant and repeats HELD/PREPARE/ACK before manual-authorized ARM. Reference waits for1s of integrated settling,then uses the actual Odom/IMU PVA and measured quaternion. Goal is initial anchor+[.4,0,0] in WORLD. FixedT2 and3 quintics remain.
The deterministic boundary-reference formulas construct only the two physical knot parameters. Original MincoAdapter remains the sole trajectory solver. This avoids the old Raw21 absolute-A x-support limitation for rotated/lateral classical references;the neural A decoder/ABI/network are untouched.
A Plan validates finite values,durations,head/tail constraints,C4 continuity,independent numerical_reference agreement and continuous norm_max limits1/8/100 before publication.
All execution samples use original sensor_contract.sample_world -> validation.evaluate. No second sampler,frame transform or continuous-extrema implementation is introduced.

Request envelopes carry exact source epoch,curve,initial frame and PVA. Supervisor validates and atomically commits a new curve under its callback lock,checks original current PVA and old/new switch limits(.05m,.1m/s,.5m/s^2),then checks envelope PVA against that exact curve at that exact stamp. Controller publication occurs only after admission. Original state/curve time is retained rather than resetting the epoch to conceal latency.
Continuous limits are checked at admission. A curve can be executed for its whole2s horizon while sensor/Request freshness remains.2s;an expired horizon is rejected. This separates old CommandGate's planning-source-age concept from curve execution time without relaxing watchdogs.

Post-run acceptance uses ALL logged physical feedback up to confirmed stop:
progress>=.25m,terminal position error<=.10m,max position tracking error<=.15m,terminal speed<=.10m/s;measured v<=1,a<=8,planned continuous jerk<=100. Actual jerk FD is reported separately and is not a certificate of unsampled physical extrema.
Reference completes2s motion and.5s settling before provisional qualification. Child results,truth records,active faults and cleanup can still turn final status FAIL.
The analytical rest-to-rest0.4m/T2 reference has peaks about.375m/s,.57735m/s^2,3m/s^3;static original-MINCO tests pass these bounds. This is not a tracking result.
Outputs include actual p/v/a,validated references,controller/final SO3,ACKs,RPM input/actual,thrust,plans,MINCO residuals,latencies and all rejected requests.
""")
 doc('G4C_P2_E_FOREST_CLOSED_LOOP.md',"""
# E deterministic sparse forest implementation
PASS static implementation - native sensor compiled,pose transform,observed-space queries,deterministic A*,MINCO reconstruction,5Hz replan source,switch admission and independent truth observer are present.
FAIL initial observed-route qualification - current single-view startup cannot certify the full swept body/tracking tube around its own near-field camera blind region. NOT_RUN physical. E entry refuses ARM while FOREST_SENSOR_ROUTE_QUALIFICATION.json is FAIL.

Fixed asset:previously exposed G3D sparse seed3101,pose0,qualified spawn[-3.298862934,-.322489500,1.799999952]. No map/density was tuned to model or planner outcomes. The old cloud/counts/source manifest SHA are verified;new PLY is byte-preserving export of compact points,with identical native config except asset path. Native sensor is compiled into this output only. No YOPO source or old map/config was edited.
Planner sees only real depth/mask/actual pose,state and goal. Truth map/counts are visible only to a separate collision observer and offline qualification.
ObservedSpace rejects no-return20m,invalid pixels,out-of-FOV and near-plane space. It checks the whole conservative projected sphere rectangle,not a center pixel. Whole-curve checks pad radius.15 with tracking envelope.15 and speed-derived between-sample distance,plus.1m depth voxel reserve.
A* is deterministic on a local grid,with data-dependent path direction and certified edges. No fixed left/right Anchor,neural candidate or GT oracle chooses a path. References are state/goal conditioned and passed through original MINCO.
New plans run nominally5Hz;reference publication100Hz continues on the current admitted curve. If planning misses.15s or input age.2s,it rejects and stops rather than hide a timeout. There is no unbounded online leaf-variable optimization.
Proposed actual success remains>=20 replans,>=2m world-X forward progress,measured v<=1,a<=8,planned jerk<=100,no truth-monitor failure,and complete rejection/latency records. None is claimed achieved.

## Real-data counterexample retained
The frozen real forest image and0.4m reference pass original offline Hard,while sensor tube certification returns UNKNOWN_OR_BLOCKED_SWEPT_TUBE. This is a sensing/bootstrap qualification failure,not task infeasibility and not a neural failure.
We do not inject a GT free-space bubble,mark UNKNOWN FREE,shrink radius,drop the tracking envelope or adjust a map to obtain PASS. Additional qualified multiview/observed-map initialization is required before E can arm;that extension remains DESIGN_ONLY.
The current implementation uses a conservative instantaneous view,not a claimed complete fused local map. Historical observation fusion with provenance and a validated observable startup tube is the remaining E engineering gap.
""")
 doc('G4C_P2_CONTROL_AND_FRAME_AUDIT.md',"""
# Control and frame authority
PASS static - actual installed native interfaces/source contracts were reused;NOT_RUN - v4 live controller consumption.

Native integrated Plant Odom/IMU -> explicit fault relay -> supervisor/reference.
Native Depth + CameraInfo -> same relay -> exact source-Odom epoch lookup.
Reference JSON envelope -> locked supervisor admission -> PositionCommand -> unchanged SO3ControlNodelet -> Raw SO3 -> strict freshness/authorization gate -> Final SO3 -> unchanged P1 native Plant with final-command ACK.
Independent truth observer reads native Odom directly;it has no route/score output to the reference.

World p/v/a and gravity are transformed by existing sensor_contract.rotation using measured quaternion. Optical-to-body is the existing [[0,0,1],[-1,0,0],[0,-1,0]] physical calibration. No identity substitution,channel flip or display-TF +.05z offset is used. Depth is optical Z metres,K80/80/80/45,160x90;20m cap/no-return is UNKNOWN for planning.
Actual raw/final SO3 messages retain native /quadrotor header label while force numerically means WORLD. This source-derived exception is explicit;force is not rotated again.
Initial rotor state is equilibrium_hover,not a demonstrated zero-RPM takeoff. Mass.98,g9.81,kf8.98132e-9,tau1/30,native integration1000Hz,odom100Hz,controller gains and Hard are unchanged.

One authoritative actual curve path:
trajectory.build -> original MincoAdapter;
Plan.sample/at -> original sample_world/evaluate;
continuous dynamics -> original norm_max;
independent solve comparison -> original numerical_reference;
offline forest evaluation -> frozen Hard v2;
native feedback truth clearance -> existing certify_position/grid_boxes.
The new classical reference knot construction is not a new MINCO solver or a neural decoder change.

Admission/publication is synchronized. Frozen curve content hashes bind request-to-plan;late/changed/expired curves,nonfinite requests,pose mismatch and switch discontinuity latch abort. No control message is published merely because a Path exists.
Truth monitoring uses sampled native state plus a declared v/a interpolation tube;its continuous-collision conclusion is conditional on those feedback bounds. Actual unsampled acceleration/jerk extrema and motor-rate physical collisions are not certified by100Hz data. These limits remain explicit and are additional reasons not to claim E scientific readiness.
""")
 doc('G4C_P2_FINAL_GO_NO_GO.md',"""
# Final GO/NO-GO
|Item|Status|Decision|
|---|---|---|
|P1 A actual HELD5ACK/0steps|actual physical handshake PASS|history retained|
|P1 B stable integrated window5058tail|actual physical window PASS|equilibrium initial state only|
|P1 A/B complete child cleanup|FAIL|not rewritten as clean runs|
|P2 runtime/lifecycle/race/math tests|static PASS|source qualified for manual C|
|P2 native forest sensor compile|static PASS|not sensor runtime proof|
|P2 C real fault campaign|NOT_RUN|researcher manual command supplied|
|P2 D0.4m tracking|NOT_RUN|implemented;requires C PASS|
|P2 E planner/search/ROS source|static implementation PASS|code present,not executed|
|P2 E initial observed safe tube|FAIL|entry blocks E ARM|
|P2 E actual forest closed loop|NOT_RUN|NO-GO until D plus observed-route qualification|
|historical protection|PASS|669 old sources/50745 old outputs,external checks retained|
|new network/optimizer steps|PASS|0|
|new K2/K6/Anchor/Loss changes|NOT_RUN|none implemented|

RUNTIME_FIX_STATIC=PASS
C_IMPLEMENTED=YES
C_ACTUAL_PHYSICS=NOT_RUN
D_IMPLEMENTED=YES
D_ACTUAL_PHYSICS=NOT_RUN
E_IMPLEMENTED=YES
E_SENSOR_BOOTSTRAP_QUALIFICATION=FAIL
E_ACTUAL_PHYSICS=NOT_RUN
NEW_NETWORK_UPDATES=0
HISTORICAL_PROTECTION=PASS

C/D commands are real CLI paths,not claims of execution. Do not proceed to D without all6 C cases PASS. E cannot arm with current qualification. No automatic training or physical simulation follows this delivery.
""")
 doc('G4C_P2_MANUAL_COMMANDS.md',"""
# Researcher manual C/D commands
NOT_RUN physical - Codex did not execute these. Both use equilibrium_hover,actual native feedback and the unchanged controller. No engineering model weights are loaded.
The existing process-provenance wrapper captures the real owner argv,cwd,environment,stdout/stderr,exit and resources in addition to all child records.
Port11349 must have no live listener;the runner does not kill unrelated ROS processes.

C:manually authorize the6-case campaign:
~~~bash
cd /home/lizheng/query_minco_mainline
C_RUN="research_outputs/g4cp2_manual_C_$(date +%Y%m%d_%H%M%S)_$$"
/home/lizheng/anaconda3/envs/yopo/bin/python -B -m query_minco.audit.run_provenance_v1 --output "$C_RUN/process" --cwd /home/lizheng/query_minco_mainline --timeout-s 600 -- /home/lizheng/anaconda3/envs/yopo/bin/python -B experiments/g4cp2_execution_v1/manual.py --stage C --allow-physics RESEARCHER_MANUAL_C_ONLY --output "$C_RUN/run"
~~~
Inspect "$C_RUN/run/RESULT.json":stage C_CAMPAIGN,status PASS,all6 episodes PASS,matching child/episode SHA,actual positive native steps,expected case detection,confirmed terminal physics process,and cleanup PASS. A failed campaign cannot authorize D.

D:ONLY after researcher review of C PASS,in the same shell or after setting C_RUN to the actual successful directory:
~~~bash
cd /home/lizheng/query_minco_mainline
D_RUN="research_outputs/g4cp2_manual_D_$(date +%Y%m%d_%H%M%S)_$$"
/home/lizheng/anaconda3/envs/yopo/bin/python -B -m query_minco.audit.run_provenance_v1 --output "$D_RUN/process" --cwd /home/lizheng/query_minco_mainline --timeout-s 180 -- /home/lizheng/anaconda3/envs/yopo/bin/python -B experiments/g4cp2_execution_v1/manual.py --stage D --allow-physics RESEARCHER_MANUAL_D_ONLY --prior "$C_RUN/run/RESULT.json" --output "$D_RUN/run"
~~~
Inspect "$D_RUN/run/RESULT.json" and none/RESULT.json:runtime_status/cleanup_status PASS,progress>=.25m,terminal error<=.10m,max tracking error<=.15m,terminal speed<=.10m/s,actual dynamics and child outcomes.
Do not reuse any occupied output directory or rerun a successful directory.

Optional SAFE C preflight(no ROS,no ARM,no output claim):
~~~bash
/home/lizheng/anaconda3/envs/yopo/bin/python -B experiments/g4cp2_execution_v1/manual.py --stage C --allow-physics RESEARCHER_MANUAL_C_ONLY --output research_outputs/NOT_CREATED_BY_CHECK_ONLY --check-only
~~~
PASS here means prerequisites/source identity only,not fault safety.

E source exists but current initial observed-route qualification FAIL blocks ARM. No E flight command is recommended or automatically chained. Repair/qualify observed startup space without GT completion,then obtain a new exact-source C/D authorization chain if source changes.
""")
 rows=[]
 for p in sorted((O/'logs').glob('*/PROCESS_RESULT.json')):
  d=json.loads(p.read_text());s=json.loads((p.parent/'PROCESS_START.json').read_text());rows.append(dict(log=str(p.parent.relative_to(O)),argv=s['argv'],cwd=s['cwd'],exit_code=d['exit_code'],elapsed_s=d['elapsed_s'],peak_rss=d['sampled_process_tree_peak_rss_bytes']))
 write(O/'COMMANDS_AND_RESULTS.json',rows)
 doc('G4C_P2_TEST_MATRIX.md',"# Actual tests\n\nRaw argv/stdout/stderr/exit/resource samples are in logs and COMMANDS_AND_RESULTS.json. Child exit,not outer wrapper exit,is authoritative.\n\n|log|exit|elapsed s|peak RSS bytes|\n|---|---:|---:|---:|\n"+'\n'.join('|'+x['log']+'|'+str(x['exit_code'])+'|'+str(round(x['elapsed_s'],3))+'|'+str(x['peak_rss'])+'|' for x in rows)+"""
 
PASS static - final47 tests include lifecycle/close races/late callbacks,duplicate and rollback semantics,stream tail freshness,full-window peaks,false PASS,guard failure fallback,launch failure,C/D evidence rejection,original MINCO/nonzero PVA/C4/rotated world X/lateral goals,atomic admission,unknown rejection,deterministic search,and20 old Hard/K1/K6/provenance contracts.
PASS compile - new fixed-seed native forest sensor;original P1 Plant binary is reused and SHA verified.
PASS replay - actual old B physical metrics;FAIL cleanup - actual P1 A exit1/B exit-9 retained.
FAIL qualification - real frozen forest sensor safe tube;original offline Hard for the same reference is PASS. This is an expected conservatism counterexample,not hidden as success.
NOT_RUN - new ROS master,ARM,C fault episodes,D movement,E forest execution or neural updates.
Intermediate43/45-test runs and all raw outputs are retained. Synthetic process/launch-error fixtures are labeled;they are not real C results.
""")
 write(O/'ASSEMBLY_IDENTITY.json',dict(new_sources={str(p):sha(p) for p in new},source_binding_sha256=sha(O/'SOURCE_BINDING.json'),network_updates=0,actual_physics_executions=0))
 print('P2_REPORTS_ASSEMBLED',len(new))
if __name__=='__main__':main()
