"""Evidence-based final delivery;no ROS launch or optimizer."""
from pathlib import Path
import json,hashlib,statistics,zipfile
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def put(name,text):
 with (O/name).open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')
def main():
 spec=json.loads((O/'K1_FOREST_TRAINING_RUNSPEC.json').read_text())
 sh=json.loads((O/'shadow_attempt003/planner/RESULT.json').read_text());rx=json.loads((O/'shadow_attempt003/receiver_data/RESULT.json').read_text())
 dyn=json.loads((O/'DYNAMIC_FRAME_AUDIT.json').read_text());pre=json.loads((O/'preflight001/RESULT.json').read_text())
 e1=json.loads((O/'engineering_step001/RESULT.json').read_text());e2=json.loads((O/'engineering_step002/RESULT.json').read_text())
 ledger=json.loads((O/'ENGINEERING_ATOMIC_LEDGER.json').read_text());hist=json.loads((O/'HISTORICAL_PROTECTION.json').read_text())
 assert all(x['status']=='PASS' for x in [sh,rx,dyn,pre,e1,e2,hist])
 assert ledger['attempted']==ledger['completed']==2 and ledger['pending']==[]
 native=[R.parent/'YOPO'/p for p in ['Simulator/src/src/test_simulator_cuda.cpp','Simulator/src/src/sensor_simulator.cu','Simulator/src/config/config.yaml',
 'Controller/src/so3_control/src/so3_control_nodelet.cpp','Controller/src/so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp',
 'Controller/src/so3_quadrotor_simulator/launch/simulator_position_control.launch','Controller/src/utils/quadrotor_msgs/msg/PositionCommand.msg','YOPO/test_yopo_ros.py','YOPO/control_msg/_PositionCommand.py']]
 with (O/'NATIVE_ROS_SOURCE_SHA.json').open('x') as f:json.dump({str(p):sha(p) for p in native},f,indent=2)
 put('CURRENT_METHOD_AND_SIM_INTEGRATION.md',"""
# G4-A current method and integration

Two independent lines completed:real native YOPO CUDA forest sensor + non-control dynamic shadow inference;new forest+mirror shared-network training lifecycle with exactly2 engineering updates. No formal training,aircraft execution,Controller launch,migration or Git mutation.

Original469821-parameter K1,CNN/State/QueryDecoder,Raw21,A_reasonable,actual initialPVA,three quintic MINCO pieces,T2/dt2/3,L0=C_viol+F+G_task+.1*C_shape+.025*P+.1*S,P0c unchanged. Hard v2:r.12,margin.03,v3/a8/jerk100,originalGoal/UNKNOWN unchanged. No failed G3-C nonlocal mean,B1,K2,K6 training,Anchor,imitation or newLoss.774 dead scalars protected by original guard.

New reusable contracts:query_minco/integration/yopo_shadow_v1. New thin lifecycle:query_minco/training/k1_forest_v1. ROS nodes:ros/g4a_k1_shadow_v1. Experiments/tests in dedicatedG4A folders. Original network,MINCO,Loss,initializer,guard/checkpoint helpers imported,not copied. Live mainline remains source authority;source ZIP is an audit copy only.

~~~mermaid
flowchart TD
 F[Scripted non-control odometry] --> Y[Native forest sensor]
 Y --> D[ROS Depth CameraInfo]
 F --> S[World velocity and acceleration estimate]
 D --> A[Dynamic B0 observation adapter]
 S --> A
 A --> N[Frozen K1 and deterministic references]
 N --> M[Original MINCO]
 M --> V[Shadow Path plus independent subscriber]
 M --> C[Dry command/safety contract only]
 T[TRAIN16 DEV12] --> L[Original shared K1 L0 P0c lifecycle]
 L --> U[Exactly2 engineering steps]
~~~

Engineering readiness does not imply learned visual routing or safe flight. Prior mirror/nonlocal-gradient failures remain. Controller watchdog/braking/tracking still blocks actual flight.128-step manual pilot prepared but not executed;512 requires a separate authorization/version.
""")
 put('YOPO_ROS_TOPIC_AND_FRAME_AUDIT.md',"""
# Actual ROS interfaces and frames

Installed ROS Noetic;actual roscore ros_comm1.17.4. Owned isolated master127.0.0.1:11341;all ROS_HOME/logs under unique output. No existing master/process or YOPO source/config/build was modified.

|Interface|Native live contract|Executed G4A interface|
|---|---|---|
|Odometry|/sim/odom nav_msgs/Odometry|/g4a/shadow/odom_input scripted30Hz|
|Depth|/depth_image sensor_msgs/Image32FC1|/g4a/shadow/depth native10Hz|
|CameraInfo|native camera_info_topic/default /camera_info|/g4a/shadow/camera_info|
|Goal|native /move_base_simple/goal PoseStamped|/g4a/shadow/goal PoseStamped world|
|Command|/so3_control/pos_cmd quadrotor_msgs/PositionCommand|NOT PUBLISHED|
|SO3|controller ->so3_cmd ->quadrotor simulator|NOT STARTED|
|Shadow|new nav_msgs/Path|/g4a/shadow/path/fixed_forward,state_goal_quintic,K1_step128|

Native YOPO uses local control_msg.PositionCommand;its actual type/MD5 matches generated quadrotor_msgs class4712f0609ca29a79af79a35ca3e3967a. Actual serialization tested. Receiver stores all actual message types/MD5 and graph.

Native odometry linear twist is WORLD velocity despite body child_frame_id. Native IMU contains world quad.getAcc(),not standard body specific force;it was not silently reinterpreted. New adapter finite-differences adjacent world velocities at exact timestamps. B0 is frozen instantaneous body axes:R.T rotates inertial velocity,acceleration,goal displacement and unit gravity. No fictitious rotating-frame derivatives.

Raw camera opticalZ meters,90x160,K80/80/80/45,max20,pitch0. Optical-to-body[[0,0,1],[-1,0,0],[0,-1,0]],zero physical offset. Valid .04<=Z<20;NaN/no-return invalid;pad19+19 to2x128x160. Full body quaternion used. Nonzero camera extrinsic pitch remains separately unsupported.

Native TF has a5cm display-only offset;physical renderer uses odometry origin. DisplayTF is not used for projection. Matching depth/camera/odom stamps within1e-5s;actual accepted differences0. Prior dt .001.. .2s;missing acceleration history rejects. Input age[-.01,.5]s;invalidpayload/frame/camera rejects. Latest-only queues may drop frames;raw timestamp gaps/counts preserved. No true physics odometry or tracking was tested.
""")
 rr=sh['records'];timing=['|Method|Median forward ms|Median MINCO ms|Median sensor-output age ms|Max age ms|','|---|---:|---:|---:|---:|']
 for m in rr[0]['methods']:
  vals={k:[r['methods'][m][k]*1000 for r in rr] for k in ['forward_s','minco_s','output_age_s']}
  timing.append(f"|{m}|{statistics.median(vals['forward_s']):.3f}|{statistics.median(vals['minco_s']):.3f}|{statistics.median(vals['output_age_s']):.3f}|{max(vals['output_age_s']):.3f}|")
 put('K1_ROS_SHADOW_RESULTS.md',"""
# Real non-control shadow results

PASS:30 dynamic-pose snapshots,10.000Hz accepted planning stream;independent receiver received30 Path messages for each of3 methods. Receiver10s counts:299 Odometry,90 Depth,91 CameraInfo,1 latchedGoal. Native sensor is genuinely running through ROS,not file replay. Odometry is explicitly a scripted kinematic fixture,not aircraft physics or execution.

Only nav_msgs/Path published. No PositionCommand,SO3,motor or Controller started. Model tensor SHA unchanged,no_grad,network updates0. All90 continuous trajectories and30 raw images recorded in SHADOW_ARRAYS.npz. Every method/frame initial worldPVA matches observed pose/velocity/estimated acceleration within2e-5. Shadow Path is visualization,not timed controller execution.

""" +'\n'.join(timing)+"""

Median input age14.412ms;full3-method callback43.554ms,max96.260ms. Methods run serially on the same frame;K1 output age includes previous baselines,not a deployment-only K1 latency estimate. CPU/FP32 planning;native CUDA sensor only GPU work. Component times include validation/path publication overhead.

Offline independent ray replay at predeclared first/middle/last dynamic poses:all nonidentity;max3.8147e-6m,median2.3842e-7m,all pixels within3cm. Preregistered acceptance99.5%<=3cm and median<=2e-5;no threshold adjusted. Full rotation/state/optical/world roundtrips separately tested;original identity-only camera module untouched.

Attempt001 failed:roscore lacked CMAKE_PREFIX_PATH,could not locate rosout,master died;true timeout/logs retained. Prefix corrected. Attempt002 passed30 frames. Attempt003 added independent subscriber delivery and passed30 frames. No map/model was tuned. All owned processes cleanly stopped,no background shadow remains.

No online GT/Hard selection or collision-free flight claim. Historical128-step process provenance remains UNVERIFIED;new logs do not reconstruct old execution.
""")
 put('CLOSED_LOOP_COMMAND_AND_SAFETY_CONTRACT.md',"""
# Controller contract and remaining gate

CONTROLLER_COMMAND_INTERFACE=PASS for source/type/frame/serialization and dry safety tests. LIVE_CONTROLLER_CONSUMPTION=NOT_RUN. CONTROLLED_CLOSED_LOOP_READY=NO.

Actual fields:Header(world,execution epoch),position/velocity/acceleration,yaw/yaw_dot,kx/kv[3],trajectory_id>=1,READY flag. Generated type/MD5 verified. New dry adapter samples original MINCO with translated position,rotated velocity/acceleration.

CommandGate validates finite coefficients/positive times,SO(3),origin,currentPVA,epoch/sensor/odom freshness<=.2s,low-speed1m/s,original a8/jerk100. Current-time target must match currentPVA within .05m/.1m/s/.5m/s2;previous-command mismatch rejects switches. Stale/future/NaN/invalid/discontinuous input latches ABORT. No implicit reset. This module has NO ROS Publisher. Tests cover actual message serialization and invalid states/frames/times/trajectories. The1m/s gate is a future execution restriction,not a changed Hard/Loss threshold. Frozen K1 can exceed it and would reject low-speed execution;no silent time scaling added.

Critical blocker:SO3ControlNodelet copies commandPVA without checking trajectory_flag or age;odom callback republishes the last command and contains TODO fallback-to-hover on command timeout. Stopping publication or sending ABORT is NOT certified braking. Need separate watchdog/braking supervisor and actual acknowledged transition under fault injection without modifying baseline. Not implemented or flight-tested.

Existing simulator_position_control.launch starts physics+controller at fixed0,0,2,nominal100Hz odom,physics default1000Hz. Those are source values,NOT measured controller rates. Launch was not run and must not be blindly paired with a forest pose lacking clearance.

Next manual gate:obstacle-free/sparse qualified initial pose;true dynamics state/accel alignment;measure controller/planner rates;low-speed tracking and switch continuity;lostdepth/odom,timeout,NaN and controllerfault injection;verify physical braking/termination under independent truth collision monitor. GT may stop/monitor,never choose model paths. Actual armed flight command withheld because this gate is not qualified;no invented execution CLI supplied. Manual SHADOW command is complete and non-control.
""")
 table=['|Role|Independent forest seeds|Forest snapshots|Required-channel tasks|Total|','|---|---|---:|---:|---:|']
 for role in ['train','dev']:
  rows=spec[role];seeds=sorted({r['map_seed'] for r in rows if r['kind']=='forest'})
  table.append(f"|{role}|{seeds}|{sum(r['kind']=='forest' for r in rows)}|{sum(r['kind']=='mirror' for r in rows)}|{len(rows)}|")
 put('FOREST_TRAIN_DEV_DATASET.md',"""
# Scientific data qualification

Seeds,states/pose ranges,split and proof criteria registered before generation/inference. Six new sparse/medium/dense native12m maps:24 inputs,22 certifiedfeasible,2 undetermined(dense4302 pose0 Near/Far),no infeasibility claim. Undetermined excluded,not adjusted. G3-D TEST3101/3102/3201/3202/3301/3302 never included.

""" +'\n'.join(table)+"""

New states:v[.6,.1,0],a[.05,0,0] orv[1,0,0],a[-.05,.05,0]. Fixed anchor fractions[.3,.35],[.45,.65],height1.8. Independent original continuous Hard witness bank and actualnative depth/count/camera qualification. Same tree asset,narrow motion/identity offline poses remain limitations.

Original G3-B TRAIN4/DEV2 required-channel pairs reused with unchanged sourceSHA/family roles. TRAIN staggered_double_aperture/blind_branch_splitter;DEV column_row_narrow_decoys. Both sides per family. DEV previously diagnostically observed,not a virgin TEST set. Mapseed/family disjoint;nearby frames not independent-map evidence.

Old generic forest exporter internally labels geometry forest_test. New versioned binding explicitly rebinds scientific role for NEW seeds after strict unchanged source loader validation;does not edit old manifest or relabel G3-D TEST. New geometry is the same original typed implementation with role identity. RunSpec checks bindingSHA/member rows,seed/family split and every source manifest. This semantic distinction is disclosed.

Network input onlyDepth/Mask+State12. Occupancy only original privileged offline L0/P0c+Hard;witness not label/model input/online selection. Channel metrics use existing explicit crossing sections. Forest reports original Near/Far tasks,not an invented crossing metric.
""")
 put('K1_FOREST_TRAINER_IMPLEMENTATION.md',"""
# Original-L0 thin forest trainer

FOREST_TRAINER_READY=YES for manual128-step engineering pilot,not scientific route-learning success.

Reuses original model/initializer/objective,numeric/group/metric checks,AdamW,DeadRowGuard,checkpoint/RNG helpers. Old TRAIN13/DEV8 trainer contract unchanged. One model across TRAIN16;preflight16 backwards checks identical parameter objects and unchanged weightSHA. DEV12 no_grad only;before/after weights and gradients identical.

Raw0:15 finite gradients/module norms saved;Raw15:21 zero checked.774 last-layer dead scalars guarded by original restore+moment sanitization,not zero-gradient assumptions. Atomic outer attempt token reserved before guard.step;old guard uses unique per-attempt private compatibility record,budget1. Atomic total controls actual updates;no abandoned-attempt refund. Existing real concurrency/duplicate-output tests reran.

Exactly2 engineering steps in separate processes. Fresh AdamW plus nonzeroWD.01/AMSGrad first;second resumes checkpoint on another TRAIN task,probes nonzero exp_avg/exp_avg_sq/max_exp_avg_sq in separate dirty checkpoint and memory. Model SHA continuity,cursor,liveoptimizer tensors and Python/NumPy/Torch RNG exact. Dead parameters byte-identical,all dead moments zero;livehead rows truly update. Atomic attempted=completed2,pending0,third reserve rejected. Checkpoints engineering_only/nondeployable.

Per-step full componentLoss,Raw/module gradients/updates,deadSHA,optimizer tensorSHA/norm,independent MINCO,poststep TRAIN/DEV Hard+Task/crossing/quality saved. After engineering tests TRAIN8/16,DEV8/12 Hard+Task;channel safe-crossing0 both. These are engineering diagnostics,not formal research findings.

CLI defaultpreflight;train requires manual approval bound to specSHA,seed,exact exclusive output and optional parentSHA. Every formal update recheckssource/data. Formal resume requires separate approval and new registered resume directory,restores fullstate/cursor/RNG and recordsparent;never overwrite oldrun. Actual engineering restart passed;formal128 lifecycle and formal fullresume NOT_RUN.

Canonical requested RunSpec is byte-identical to qualified_v2. Preliminary unexecuted bytes preserved as superseded_v1;VERSION_HISTORY/FINAL_INDEX disclose data-binding hardening and promotion. No expectedSHA override. Current CLI accepts128 only;512 remains separate design-only version/approval/qualification.
""")
 put('BENCHMARK_AND_BASELINE_PLAN.md',"""
# Baseline and scientific plan

Same live sensor frame:fixed-forward,state-goal,Step128;all3 shadow outputs delivered. No tracking/success flight comparison. Future trained checkpoint replacement must be explicitSHA/strictstate_dict plus repeated dynamic/safety tests,no automatic swap.

Pilot TRAIN16/DEV12,CPUFP32/B1,originalA/L0/P0c,128 updates,eval0/8/32/64/128,AdamWlr1e-4/WD0,clip10. Seeds20261104/05/06 registered;manual command runs just one. Report TRAIN/DEV components,Hard+Task,channel Hard+Task+crossing,mirror direction switches,UNKNOWN/collision/dynamics failures,continuous peaks,feasible quality,grad/update counts and resources. Loss decline is not success. OriginalL0 local route gradient remains insufficient;this unchanged-L0 pilot has no promised mirror success.

Future closed-loop evaluation needs independent collision monitor/task/abort/tracking/latency/rate/resource records. GT may terminate,not select. Same states/mapseeds/time/constraints for fair paired reference comparison. No cross-asset/high-speed/generalization claim from these local shared-tree data.512 independent authorization,not implicit continuation.

NativeYOPO policy NOT_RUN_NOT_COMPARABLE:15 anchors/2pieces/nominal10/6s/v6-a6-j60/1-channel96x160/different safety and learned selection vsK1. Sensor reuse alone does not align algorithms. Separate native policy bridge/constraint qualification required.
""")
 A="""#!/usr/bin/env bash
set -euo pipefail
BASE=/home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
cd "$BASE"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN="$BASE/research_outputs/g4a_manual_shadow_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$RUN/process" --cwd "$BASE" --timeout-s 120 -- "$PY" -B experiments/g4a_forest_and_shadow_v1/run_shadow.py --output "$RUN/session" --frames 100 --port 11341
"""
 B="""#!/usr/bin/env bash
# DRAFT:researcher intentionally reviews/authorizes before executing.
set -euo pipefail
BASE=/home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
OWNER="$BASE/research_outputs/g4a_forest_shadow_20261111_v1"
SPEC="$OWNER/K1_FOREST_TRAINING_RUNSPEC.json"
SEED=20261104
APPROVAL="$OWNER/approvals/A_reasonable_seed$SEED.json"
cd "$BASE"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B experiments/g4a_forest_and_shadow_v1/run.py --stage approve --spec "$SPEC" --seed "$SEED" --approval "$APPROVAL" --researcher-statement "I explicitly authorize this exact frozen K1-Forest-v1 RunSpec for seed20261104 and128 formal updates in its registered output."
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$OWNER/formal_process/A_reasonable_seed$SEED" --cwd "$BASE" --runspec "$SPEC" --approval "$APPROVAL" --timeout-s 3600 -- "$PY" -B experiments/g4a_forest_and_shadow_v1/run.py --stage train --spec "$SPEC" --approval "$APPROVAL" --seed "$SEED" --output-run "$OWNER/formal/A_reasonable_seed$SEED"
"""
 put('A_SHADOW_MANUAL.sh',A);put('B_TRAIN_MANUAL_DRAFT.sh',B)
 put('MANUAL_COMMANDS.md',"""
# Independent complete manual commands

A creates exclusive output,owned isolated ROS master,native sensor,scripted odometry,planner and independent receiver;visualization only,owned processes cleaned on exit. Port occupied rejects,never kill existing master. Prepared100-frame command NOT_RUN;actual30-frame tests authoritative.

~~~bash
bash /home/lizheng/query_minco_mainline/research_outputs/g4a_forest_shadow_20261111_v1/A_SHADOW_MANUAL.sh
~~~

B is a complete formal-training DRAFT. After intentional researcher review,execution first creates a REAL exactSHA-bound approval with an explicit statement,then starts only the registeredseed20261104/128-step run via actual process logger. No placeholder approval path. Approval creation and formal training NOT_RUN this round. Existing approval/run/process directories reject reuse;never delete to retry. Current source drift correctly rejects.

~~~bash
bash /home/lizheng/query_minco_mainline/research_outputs/g4a_forest_shadow_20261111_v1/B_TRAIN_MANUAL_DRAFT.sh
~~~

Canonical spec SHA equalsqualified_v2. Other registeredseeds need separate reviewed approvals;no automatic sweep.512 unsupported by currentCLI. Resume needs parentSHA,new registered resume output and independent approval,not overwrite/retry.

No armed-flight CLI is advertised because watchdog/braking/controller consumption remains unqualified. The real baseline launch exists but is not a safe autonomous-flight command.
""")
 logs=sorted((O/'logs').glob('*/PROCESS_RESULT.json'));table=['# True commands and test matrix','','|Attempt|Child exit|Seconds|Sampled summed RSS MiB|Result|','|---|---:|---:|---:|---|']
 for p in logs:
  d=json.loads(p.read_text());result='PASS' if d['exit_code']==0 else 'FAIL preserved'
  table.append(f"|{p.parent.name}|{d['exit_code']}|{d['elapsed_s']:.3f}|{d['sampled_process_tree_peak_rss_bytes']/1048576:.1f}|{result}|")
 table+=['','RSS sums process-tree mappings,may doublecount shared memory;not private/continuous peak.','',
 '|Check|Status|','|---|---|','|Native ROS sensor and independently received3 methods|PASS30 each|','|Dynamic geometry replay and input/frame tests|PASS|',
 '|New safety/split/approval/actual engineering contracts|PASS8 tests;no test optimizer calls|','|Old K1/K6/Hard/MINCO/UNKNOWN/crossing/Ledger|PASS20 tests|',
 '|Shared network qualification|PASS16 TRAINbackwards,DEV12 forward only,weightSHA unchanged|','|Engineering real updates|PASS2,WD,moments,AMSGrad,RNG,cross-process resume|',
 '|Actual live Controller consumption/tracking/frequency/braking/flight|NOT_RUN,NOT_READY|','|Formal128/512/full formalresume|NOT_RUN|',
 '|NativeYOPO policy comparison|NOT_RUN_NOT_COMPARABLE|','|Historical128 process provenance|UNVERIFIED preexisting|','|OldC1 registry|BLOCKED preexisting,not modified|',
 '|Historical preservation|PASS|','','## Actual logged argv','']
 for p in logs:
  d=json.loads(p.read_text());table+=['### '+p.parent.name,'','~~~json',json.dumps(d['argv']),'~~~','',f"cwd={d['cwd']};stdout={p.parent.relative_to(O)}/stdout.log;stderr={p.parent.relative_to(O)}/stderr.log;truechild={d['exit_code']}.",'']
 table+=['Nested nativecompile/link logs record truecompiler argv/exit. ROS service folders record each realargv,PID,stdout/stderr,actualexit and ownedSIGINT cleanup. All environment/resource samples in outer PROCESS_RESULT. First master/rosout failure retained,never counted as successful shadow.','',
 'SOURCE identity and Git reads:ENTRY_PROTECTION. Preliminary RunSpec preserved;current canonical copied exclusively from qualified_v2,not oldSHA bypass.']
 put('TEST_MATRIX.md','\n'.join(table))
 put('HISTORICAL_PROTECTION.md',f"""
# Historical protection PASS

ContentSHA checked{hist['counts']['old_source']} oldsource/config and{hist['counts']['old_outputs']} historicaloutputs;external1678 sourceSHA+298065 size/mtime;partialmigration80 fileSHA. No changes/missingoldfiles,Git branch/HEAD/index/diff drift or unexpectedsource. Whitelist only. Root.gitignore preserved as found,no restore/commit. No YOPO source/config/build/map/checkpoint,reach_planner,G3-DTEST,oldmethod/128run or migration modified.

External large tree coverage is metadata+selectedsource,not fullcontentSHA for every external asset. Explicit native source/config/lib/data/checkpoint inputs additionally hashed. All ownedROS services stopped.2 engineering steps,formal0. No push/reset/clean,migration or aircraft control.
""")
 status=dict(ROS_SENSOR_BRIDGE='PASS',K1_SHADOW_RUNNING='YES',DYNAMIC_FRAME_CONSISTENCY='PASS',CONTROLLER_COMMAND_INTERFACE='PASS',CONTROLLED_CLOSED_LOOP_READY='NO',
 FOREST_TRAIN_DATA_QUALIFIED='YES',FOREST_TRAINER_READY='YES',ENGINEERING_OPTIMIZER_STEPS=2,FORMAL_RESEARCH_TRAINING_STEPS=0,HISTORICAL_PROTECTION='PASS')
 with (O/'FINAL_STATUS.json').open('x') as f:json.dump(status,f,indent=2)
 put('README.md',"""
# G4-A audit delivery

Real native ROS forest->dynamic observation->frozen K1/reference->shadow paths PASS30 rounds. New TRAIN16/DEV12 shared-network lifecycle PASS,exactly2 engineering updates,formal0. All services stopped. K1_SHADOW_RUNNING=YES means actually executed,not still running in background.

Controlledclosedloop NOT_READY:dry message/frame/abort contracts pass,but livecontroller consumption/watchdog/braking acknowledgement/tracking/rates untested. No Controller/physics/flight launched.

Read8 requestedreports,TEST_MATRIX,MANUAL_COMMANDS,HISTORICAL_PROTECTION,FILES_AND_SHA andFINAL_STATUS. CanonicalRunSpec=qualified_v2;unexecutedv1 preserved. Manual A/B scripts are complete independent commands and NOT executed as delivery scripts. B creates real authorization only when researcher intentionally runs it.

SOURCE_SNAPSHOT.zip is audit-only,not a second implementation authority. No data/weights in sourceZIP;assets/checkpoints stayresearch_outputs. No GitHub push/migration. Engineeringweights nondeployable. Old scientific route-signal failures and legacyC1/process-provenance gaps preserved.
""")
 sources={}
 for base in ['query_minco','ros/g4a_k1_shadow_v1','experiments/g4a_forest_and_shadow_v1','tests/g4a_forest_and_shadow_v1']:
  for p in (R/base).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.cu','.cuh','.h','.hpp','.json','.md']:sources[str(p.relative_to(R))]=p
 for p in list((R.parent/'YOPO/Simulator/src').rglob('*')):
  if p.is_file() and p.suffix in ['.cpp','.cu','.cuh','.h','.hpp','.yaml']:
   sources['READONLY_YOPO/'+str(p.relative_to(R.parent/'YOPO'))]=p
 for p in native:sources['READONLY_YOPO/'+str(p.relative_to(R.parent/'YOPO'))]=p
 with zipfile.ZipFile(O/'SOURCE_SNAPSHOT.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name,p in sorted(sources.items()):z.write(p,name)
  z.writestr('SOURCE_IDENTITY.json',json.dumps({name:dict(original_path=str(p),sha256=sha(p)) for name,p in sources.items()},indent=2))
  for name in ['README.md','K1_FOREST_TRAINING_RUNSPEC.json','SCIENTIFIC_DATASET_BINDING.json']:z.write(O/name,'RUN_BINDING/'+name)
 print(json.dumps(status,indent=2));print('SOURCE_ARCHIVE_FILES',len(sources),sha(O/'SOURCE_SNAPSHOT.zip'))
if __name__=='__main__':main()
