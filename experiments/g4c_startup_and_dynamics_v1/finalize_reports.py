"""Assemble evidence reports;never launches training or physics."""
from pathlib import Path
import sys,json,csv,hashlib,subprocess,zipfile,math
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
def doc(name,s):
 with (O/name).open('x') as f:f.write(s.strip()+'\n')
def main():
 entry=json.loads((O/'ENTRY_PROTECTION.json').read_text())
 diag=json.loads((O/'DYNAMICS_ZERO_UPDATE_SUMMARY.json').read_text());records=diag['all_records']
 engineering=json.loads((O/'ENGINEERING_RESULT.json').read_text())
 startup=json.loads((O/'STARTUP_EVIDENCE_DIAGNOSIS.json').read_text())
 protection=json.loads((O/'HISTORICAL_PROTECTION.json').read_text())
 # Whitelisted generator itself was added after the first protection scan;check its membership and old identities again below.
 k6=json.loads((O/'K6_DIRECT_REGRESSION_CORRECT_FIXTURE.json').read_text())
 modules=[str(p) for base in entry['allowed_new'] if base.startswith(('query_minco/','experiments/','tests/','ros/')) for p in (R/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 bindings={p:sha(p) for p in modules}
 for p in [R/'ros/g4b_control_supervisor_v1/controller_only.launch',
  R/'query_minco/integration/yopo_control_supervisor_v1/lease.py',
  R.parent/'YOPO/Controller/src/so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp',
  R.parent/'YOPO/Controller/src/so3_control/config/gains_hummingbird.yaml',
  R.parent/'YOPO/Controller/src/so3_control/config/corrections_hummingbird.yaml',
  R.parent/'YOPO/Controller/devel/lib/libquadrotor_dynamics.so',
  R.parent/'YOPO/Controller/devel/lib/libso3_control_nodelet.so',
  O/'startup_build_v2/gated_quadrotor_simulator',O/'startup_build_v2/gated_quadrotor_simulator.cpp',
  R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/native_open_sensor',
  R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/config.yaml',
  R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build/floor_only.ply']:
  if not p.is_file():raise FileNotFoundError(str(p))
  bindings[str(p)]=sha(p)
 write(O/'MANUAL_STARTUP_SOURCE_BINDING.json',bindings)
 fields=['step','sample_id','split','F_speed','F_acceleration','F_jerk','v_sample','a_sample','j_sample','v_continuous','a_continuous','j_continuous','v_exceed_s','a_exceed_s','j_exceed_s','F_peak','continuous_J','old_L0','diagnostic_L0','gradient_status']
 with (O/'DYNAMICS_PER_SAMPLE.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for x in records:
   row={k:x[k] for k in ['step','sample_id','split']};row.update({k:x['F_parts'][k] for k in ['F_speed','F_acceleration','F_jerk']})
   row.update(dict(zip(['v_sample','a_sample','j_sample'],x['sample_peaks'])));row.update(dict(zip(['v_continuous','a_continuous','j_continuous'],x['continuous']['maxima'])));row.update(dict(zip(['v_exceed_s','a_exceed_s','j_exceed_s'],x['continuous']['exceed_duration_s'])))
   row.update(F_peak=sum(x['peak_terms'].values()),continuous_J=x['continuous']['continuous_violation'],old_L0=x['old_loss'],diagnostic_L0=x['diagnostic_loss'],gradient_status='AUDITED_NO_UPDATE' if x['split']=='train' else 'NOT_RUN_DEV_FORWARD_ONLY')
   w.writerow(row)
 with (O/'RAW_FD_AND_CONFLICTS.json').open('x') as f:json.dump([dict(step=x['step'],sample_id=x['sample_id'],gradients=x['gradients']) for x in records],f,indent=2)
 doc('REAL_PHYSICS_STARTUP_ROOT_CAUSE.md',"""
# Real physics startup: root cause and bounded repair
Current HEAD: 4a2d962f599550d0e48f1b9dbfb127565d2c1594. Both G4-B source snapshot and researcher physics source_binding match current bytes exactly. HEAD differs from the earlier report because of intervening researcher commits;this is not evidence of a source-byte regression. ENTRY_PROTECTION.json contains real status, staged diff, unstaged diff and SHA for all repositories. No reset,unstage,commit or baseline edit was performed.

## Historical observations
Source: research_outputs/g4b_researcher_physics_20261111_v1/safety/timeout.
Physics started at Unix1794374781.0520165,PID4639. The observer recorded6 odom,6 IMU,6 raw-controller events.
First odom stamp1794374781.1586134: z1.975312864,vz-0.694747756.
Last odom stamp1794374781.2086651: z1.928389570,vz-1.181945723.
Every IMU stamp was0;first measured acceleration z=-9.758552915 m/s^2.
Lease never became ACTIVE;final_gate_open was always false. It latched MEASURED_SPEED at monotonic12563.714916447. Owned physics ultimately exited-9. This was startup failure before trajectory tracking,not a failed K1 flight.

STARTUP_UNIFIED_TIMELINE.csv aligns process Unix times,ROS stamps and lease monotonic times. ROS-minus-monotonic alignment1794362217.4863298 comes from paired lease clocks;process-start alignment is approximate and does not identify the first integration timestamp.
Depth/request/final-SO3 samples and raw force vectors were NOT_RECORDED by the old observer. Lease evidence supports no depth/request receipts before abort and no final forwarding. Missing force measurements are not reconstructed as observed values.

## Causal chain
1. run_manual_physics.py used Topic registration plus WAIT heartbeat and a READY file to launch physics. Registered publishers do not establish live,finite,fresh content.
2. Native simulator integrates unconditionally from the first loop,without a valid-final-command handshake. Its static Command begins with zero force;initial motorRPM is0. This is a source inference,not a historical force measurement.
3. Native quadToImuMsg never sets header.stamp. physical_reference.py requires all odom/IMU/depth samples fresher than.2s;IMU stamp0 prevents Request publication.
4. WAIT correctly refuses final SO3 forwarding. Raw controller output alone is not execution. Removing WAIT gating would bypass authorization and would not repair sensor identity.
5. The plant therefore advances without established command readiness. The observed near-gravity acceleration and growing downward speed are consistent with this chain. Initial Control had no initializer,and NaN actuator fallback could reuse undefined prior values;a new host fails closed instead.

## Force,frames and unchanged physics
Native mass=.98kg,gravity=9.81m/s^2: nominal world hover force[0,0,9.6138]N. Native IMU acceleration is world inertial acceleration,not body-frame specific force. IMU axes and SO3 desired force use their actual native contracts.
Effective position gains are[5.7,5.7,6.2],velocity gains[3.4,3.4,4]. YAML gains/pos and gains/vel do not override the nodelet's gains/kx and gains/kv keys. No gains were retuned.
Integration1000Hz,odometry100Hz,motor time constant1/30s,initial RPM0 and the native dynamics library remain unchanged. Renderer intrinsic160x90,K=[80,0,80;0,80,45;0,0,1],Z-depth metres,20m range. No identity-rotation substitution is introduced into motion planning;this test is a stationary B0 HOLD only.

## Versioned repair
Original YOPO and G4-B sources are untouched.
query_minco/integration/yopo_startup_v2/readiness.py requires5 samples of odom,IMU,depth,HOLD Request,rawSO3 and held-plant state. WAIT never forwards a final command. A manual bootstrap-HOLD flag permits PREPARE_HOLD;5 distinct plant acknowledgements of actual final command stamps establish PREPARED_HOLD_ACK.
plant_gate.hpp prevents every integration step until5 finite,unit-quaternion,near-hover final commands and a matching explicit nonce permit. Missing command freshness(.2s),invalid force/clock or actuator failure latches a terminal abort.
build_gated_simulator.py applies a recorded overlay to the read-only native host,links the original dynamics library and stamps IMU at the same actual state publication epoch as odometry. Held-before-integration IMU acceleration0 is explicitly labelled constrained HELD kinematics;it is not integrated physical feedback. Native getAcc is uninitialized before integration and is not read in HELD.
ros/g4c_startup_v2/prepare_hold_node.py only requests stationary HOLD;it cannot execute a trajectory. manual_startup.py defaults to held-readiness,uses an isolated master11345,new output claim and source/binary SHA binding. A separate manual flag is required for the only arm-permit site.
The owner stops the physics process before stopping publishers. This is simulation fail-stop,not certified real-aircraft braking.

## Qualification and remaining gate
36 pure Python contract tests and a header-only C++ executable pass;the derived simulator compiles and links. ROS startup,actual first integration,motor spin-up dip,hover tracking and post-arm live fault injection are NOT_RUN.
The native motor lag has not been removed or initialized to artificial hover. Even a valid gravity-compensating command does not prove successful first hover. No real-physics PASS is claimed. Review MANUAL_STARTUP_COMMANDS.md,run held-readiness first,inspect its evidence,then separately decide whether to run the short hover. No trajectory/forest command is supplied.
""")
 doc('DYNAMICS_PROXY_ROOT_CAUSE.md',"""
# Dynamics proxy: frozen-checkpoint findings
All84 cases are frozen Step0/64/128 x TRAIN16+DEV12,using the original K1,absolute A decoder,fixedT2,three quintics,original L0/P0c and unchanged geometry/Hard. DEV is forward-only;DEV gradients are NOT_RUN. Existing checkpoints,RunSpec,manifests and original evaluation files are unmodified. Final84-case proxy replay matches the earlier recorded values exactly.

Original F_dyn is a normalized time average of squared relative exceedances:
F_speed=sum alpha*relu(norm(v)/3-1)^2;likewise a/8,j/100.
There are64 samples per segment with trapezoidal normalized weights,192 points including duplicated boundaries. F_domain is separate and unchanged. Hard uses independent continuous polynomial extrema;P0c is the collision/domain training proxy,not a different velocity evaluator.

| checkpoint | TRAIN v/a/j violating counts | DEV v/a/j violating counts | all-case maximum v/a/j |
|---|---|---|---|
|0|0/0/0|0/0/0|2.419027 /4.243675 /25.830817|
|64|0/0/0|0/0/0|2.652595 /5.324462 /33.151739|
|128|10/4/0|7/2/0|3.525597 /9.237077 /52.823258|

AtStep128 the largest absolute sampled-versus-continuous discrepancies are0.00043244m/s,0.00152217m/s^2,0.00179909m/s^3. All frozen violations are already visible in the original samples;none is solely a between-sample threshold miss.
Maximum exceed durations are0.338336s for speed and0.337237s for acceleration;jerk never exceeds100.
Step128 mean original F_speed/F_acc/F_jerk: TRAIN0.000723894/0.000336997/0;DEV0.000449873/0.000143217/0.
The main demonstrated mismatch is average-versus-maximum optimization,not sampling aliasing. A small dimensionless average penalty can coexist with a disqualifying continuous peak. This does not prove dynamics alone explains collision or route-learning failures.

## Gradient evidence
All48 TRAIN cases have raw0:15 per-component gradients;raw15:21 gradients are0. DEV gradient vectors remain NOT_RUN.
24 nonzero raw-directional finite differences pass the preregistered tolerance abs5e-5+3% relative at epsilon1e-3. Inactive terms are explicitly ZERO_LOCAL_GRADIENT,not fabricated active FD passes.
Among10 violating TRAIN checkpoint cases,cosine conflicts versus the new peak gradient occur in C_viol5/10,weighted shape5/10,weighted progress6/10,weighted smoothness0/10. G_task has zero local gradient in these10,so its cosine is undefined. Original total L0 conflicts in1/10;the diagnostic replacement total conflicts in0/10. These are local raw-coordinate comparisons,not an optimizer guarantee.
All10 normalized negative-peak-gradient counterfactuals at preregistered epsilon1e-3 decrease independent continuous squared violation;none increases. epsilon1e-4 is also retained. These are zero-update probes,not leaf-variable training or deployed candidate selection.

## Exactly two engineering network updates
A separate directory engineering_peak_probe loads frozenStep128 and one shared network. Only the first2 violating TRAIN rows in frozen order are used;AdamW lr1e-5,weight_decay0,clip10. DEV is absent from backward.
The two selected-case continuous J changes are-0.000458779 and-0.000563517. TRAIN sum J changes0.228415615 ->0.197980471. All CNN/State/Query/Head modules have finite gradients and real parameter updates. Raw15:21 and all774 dead-row scalars remain byte-identical;guard reports zero dead AdamW moments. Atomic Ledger attempted=completed=2,pending=0.
These weights are ENGINEERING_ONLY_NOT_DEPLOYABLE. No formal scientific training was run. Two successful updates establish local engineering feasibility,not robustness,Hard success,visual route learning or long-run convergence.
MICROSTEP_RESULTS.md and ENGINEERING_RESULT.json contain the actual checkpoint,optimizer,gradient,loss and solver records.

## Precision and unresolved limits
Original MINCO supports FP32 only. A full FP64 MINCO solve is NOT_RUN_UNSUPPORTED,not claimed as tested. The same frozen FP32 polynomial coefficients are promoted for independent extrema and FP64 proxy accumulation.
Maximum FP32-versus-FP64 upper-ratio differences over84 cases: speed1.500674e-7,acceleration7.838260e-7,jerk1.068834e-7.
NONSMOOTH_TIE_DIAGNOSIS.json records unequal central FD and autograd at a deliberate tied maximum;this is expected subgradient behavior and not concealed as a smooth FD pass.
The new proxy is versioned and diagnostic. Original L0 and its trainer remain unchanged. A separately authorized short pilot is reasonable after review and implementation of a bound entry;128-step training and physical execution remain NO-GO this turn.
""")
 doc('PEAK_PROXY_MATH.md',"""
# subdivided_bernstein_peak_upper_v1
Input polynomial c[B,K,3,3,6],K1 or6,duration[B,K,3] exactly the stored fixed2/3-second values;all values finite,CPU FP32/FP64. Polynomial coefficient axis6 is unrelated to candidate count. Differentiation of durations is rejected. Physical constraints remainv<=3m/s,a<=8m/s^2,j<=100m/s^3.

For derivative order r and degree n=5-r:
q_r(t)=sum_i c_(i+r)*(i+r)!/i! *t^i.
On one segment t=T*u,form power coefficients a_i=c_(i+r)*(i+r)!/i!*T^i.
Bernstein control vectors b_j=sum_(i<=j) a_i*choose(j,i)/choose(n,i).
Apply6 dyadic de Casteljau subdivisions,64 cells per segment.
For limit L_r define
 U_r=max_(segment,cell,j) sqrt(sum_axis(b_j/L_r)^2+epsilon_ratio^2),
 F_peak_r=relu(U_r-1)^2, F_peak=sum_r F_peak_r,
 epsilon_ratio=1e-9.
All ratios and penalties are dimensionless;derivative vectors keep their SI units before normalization.

## Bound and approximation
In exact arithmetic,Bezier convex-hull containment proves max_t norm(q_r(t))/L_r <= U_r.
An a-posteriori ratio gap bound is
 U_r - max_t norm(q_r(t))/L_r <= max_cell(h*D_cell/L_r)+epsilon_ratio,
where h=T/64 and D_cell is the largest norm of derivative Bezier controls n*(b_(j+1)-b_j)/h.
The implementation returns that conservative bound. It is deliberately loose;actual U-minus-exact-max is also inspectable in the per-sample arrays. Increasing subdivision depth cannot increase the unsmoothed control-hull bound in real arithmetic. The unit suite checks bounds and monotonicity at depths0/2/4/6 against independent stationary-root extrema.
This is NOT a floating-point interval certificate and cannot replace Hard. Rounding may invalidate an exact-arithmetic inequality by machine-scale amounts;FP64 accumulation reduces but does not certify this risk.

## Diagnostic total,normalization and gradient
 L_diag=L0 - mean_K(F_dyn) + mean_K(F_peak).
F_domain and every other original L0 term and scalar coefficient remain unchanged. The new function is not installed into the old trainer.
For identical K candidates,mean loss is unchanged and each independent-candidate gradient is1/K of K1;unit tests verify K1/K6 sum-gradient normalization. This artificial repeated-polynomial test checks reduction math only;it is not a new network implementation.

The norm smoothing has error<=epsilon_ratio in normalized units and finite derivative atzero. Maximum is differentiable almost everywhere;torch.amax uses a distributed subgradient at ties. Squared positive-part is zero below and at the boundary and has zero derivative exactly at the boundary. Thus it supplies no strict interior margin and does not guarantee finite-step Hard feasibility.
FP64 coefficient-direction finite differences pass at epsilon1e-5. SYNTHETIC_COMPONENT_FD.json additionally checks each active speed,acceleration and jerk penalty separately on declared synthetic constant-derivative fixtures. No real jerk-exceedance case exists in these84 tasks. A constructed three-piece tie explicitly reports both one-sided differences,central difference and the selected subgradient. Raw-to-MINCO directional FD remains FP32 and uses the separate tolerance declared before inference.
Original FP32 decoder/MINCO are unchanged;promoting their output coefficients to FP64 preserves backpropagation to the original FP32 network. Source configuration finite checks were tightened after the engineering run;all84 frozen proxy results were then replayed bit-exactly to verify normal-input math did not change.
""")
 doc('MICROSTEP_RESULTS.md',"""
# Engineering update qualification
Exactly2 shared-network AdamW steps were executed in engineering_peak_probe. FORMAL_RESEARCH_TRAINING_STEPS=0.
Starting identity is the original G4-A forestStep128,not a replacement of its checkpoint. The probe loads model weights only and creates fresh AdamW state;it does not claim formal-run optimizer resume.
Step1 forest_sparse_4101_p0_near_stop: independent continuous violation delta-0.000458778579.
Step2 forest_sparse_4101_p1_near_stop: delta-0.000563517275.
TRAIN aggregate J:0.228415615028 ->0.197980470911.
Both checkpoints and sidecars,private guard records,ATOMIC_LEDGER,full module gradient/update norms,Raw15 gradients,all original loss terms and solver/Hard diagnostics are saved. Output is explicitly engineering-only,not a deployable model or scientific training result.
Dead rows are restored with the existing DeadRowGuard and their AdamW state sanitized before/after each actual step. There were no additional optimizer steps in unit,regression or finite-difference tests.
No DEV backward,no GT online candidate selection,no new seed,no128-step run.
""")
 doc('K1_K6_NON_REGRESSION.md',"""
# Non-regression
PASS: original source bytes unchanged (historical protection),Hard v2 frozen identity unchanged.
PASS: existing G3-A20 contract tests,including K1/K6 full-result equality,illegal times,wrong shapes,UNKNOWN,independent MINCO/PVA/C4,witness crossing,provenance and Ledger.
PASS: currentK6 fixed G1 original synthetic probe:139 arrays bit-identical,including Raw21,coefficients,durations,loss,Raw gradients,module gradients,metrics,direct oldC1 checkpoint/postprocessor output and E0 output. Both old state_dicts strict-load without using the legacy registry. Legacy C1 registry remains BLOCKED,untouched.
PASS: K1 frozen models remain unchanged during84-case forward/audit;the final proxy replays all84 stored coefficient cases exactly.
PASS: new peak proxy K1/K6 mean loss and1/K gradient normalization.
NOT_RUN: historical pytest-only G1/BUG01 full suites;pytest is absent in the installed yopo environment. Equivalent requested duration and K1/K6/UNKNOWN cases were actually run in the existing unittest suite. This is not described as the full109-test BUG01 rerun.

## Failed attempts retained
Initial combined discovery reused colliding module names;subsequent unique import names fixed the harness. An attempted pytest-only suite could not import pytest;no dependencies were silently installed.
A new K1/K6 exact scalar equality assertion differed by7.1e-15 from FP64 mean reduction;it was corrected to12 decimal places while retaining tensor gradient checks. No algorithm tolerance/Hard threshold changed.
The first K6 replay mistakenly paired a frozen synthetic network input with real bound geometry/P0c. Raw and coefficients matched,but loss/gradients did not. Reading the original stored argv established the original SyntheticGeometry and original PhysicsV12Objective;the final replay uses exactly that frozen fixture. This was a harness contract mismatch,not an unexplained implementation drift. The failing JSON and logs remain.
The next reporting attempt could not subtract boolean NumPy arrays;casting only the diagnostic difference to float fixed it. All exact-equality predicates remain np.array_equal.
A malformed source-edit command failed at Python parse time and applied no edits. Attempts to rerun into occupied new test/build directories failed closed;unique final directories preserve those earlier outputs.
TEST_MATRIX lists every logged attempt and actual exit code. No failed log was overwritten.
""")
 doc('MANUAL_STARTUP_COMMANDS.md',"""
# Manual startup drafts - NOT_RUN
These commands exist and --help was executed;ROS launch/runtime behavior was not executed this turn.
Review source_binding and the derivative overlay before use. Keep every output unique. No forest trajectory and no K1 execution are enabled.

First: held-before-integration content/command handshake only. This launches a gated plant host but does not authorize its first integration step.
~~~bash
cd /home/lizheng/query_minco_mainline
PYTHONDONTWRITEBYTECODE=1 /home/lizheng/anaconda3/envs/yopo/bin/python -B \
  experiments/g4c_startup_and_dynamics_v1/manual_startup.py \
  --stage held-readiness \
  --allow-held-processes RESEARCHER_MANUAL_HELD_STARTUP_ONLY \
  --output "research_outputs/g4c_researcher_held_$(date +%Y%m%d_%H%M%S)_$$"
~~~
Expected label is HELD_READY_NOT_PHYSICS_VALIDATED,never physical tracking PASS. Inspect PREPARED_HOLD.json for0 integration steps,five actual final-command acknowledgements,all fresh content and a world hover force near[0,0,9.6138].

Only after researcher review,a separate optional short stationary hover draft:
~~~bash
cd /home/lizheng/query_minco_mainline
PYTHONDONTWRITEBYTECODE=1 /home/lizheng/anaconda3/envs/yopo/bin/python -B \
  experiments/g4c_startup_and_dynamics_v1/manual_startup.py \
  --stage armed-hover \
  --allow-held-processes RESEARCHER_MANUAL_HELD_STARTUP_ONLY \
  --allow-first-integration RESEARCHER_MANUAL_SHORT_HOVER_ONLY \
  --output "research_outputs/g4c_researcher_hover_$(date +%Y%m%d_%H%M%S)_$$"
~~~
This authorizes stationary HOLD only. It is NOT fault-certified and does not authorize State-Goal/K1/forest motion. Original1m/s execution cap and.2s freshness remain. Position drift>.25m or invalid/stale inputs latch abort. Simulator commands are stopped by exiting the owned plant;stopping a planner is never equated with braking. Existing motor lag may still cause startup dip and must be measured.
The preparation EVENTS.jsonl records actual raw/final forces,odom,IMU/depth stamps,HOLD requests and plant command acknowledgements. Parent records real child argv,environment,stdout/stderr and exit codes. Physical results remain NOT_RUN until the researcher executes and inspects them.
A six-case real fault-injection campaign and tracking test must precede any motion;they are not automatically chained or claimed completed by this driver.
""")
 # Draft intentionally has no nonexistent production training CLI.
 spec=json.loads((R/'research_outputs/g4a_forest_shadow_20261111_v1/K1_FOREST_TRAINING_RUNSPEC.json').read_text())
 pilot=dict(schema='g4c_peak_proxy_short_pilot_DESIGN_ONLY_v1',status='DRAFT_NOT_AUTHORIZED_NOT_RUN',method_id='k1_fixed2_phaseA_peak_dyn_replacement_pilot_v1',
  network='unchanged shared K1',decoder='unchanged A_reasonable absolute fixed2',objective='L0 - mean(F_dyn) + mean(F_peak)',proxy_component='subdivided_bernstein_peak_upper_v1',
  max_updates=16,evaluation_steps=[0,1,2,4,8,16],seed=20261104,batch_size=1,device='cpu',dtype='float32',proxy_accumulation='float64',depth=6,
  optimizer=dict(name='AdamW',lr=1e-5,weight_decay=0.,clip_norm=10),
  initialization=dict(path=str(R/'research_outputs/g4a_forest_shadow_20261111_v1/formal/A_reasonable_seed20261104/step128.pt'),sha256=sha(R/'research_outputs/g4a_forest_shadow_20261111_v1/formal/A_reasonable_seed20261104/step128.pt'),scope='model-only warmstart,freshoptimizer;not formal resume'),
  train=spec['train'],dev=spec['dev'],train_order=spec['train_order'],data_binding=spec['data_binding'],data_binding_sha256=spec['data_binding_sha256'],
  test_use='NOT_RUN,no TEST tuning',dev_policy='forward only,no parameter or hyperparameter selection',hard_thresholds=dict(v=3,a=8,jerk=100,T=2),
  stop=['nonfinite gradient/solver','data/source/checkpoint drift','dead-row drift','unexpected optimizer attempt','increased continuous TRAIN violation or safety-metric regression at evaluation'],
  required_before_execution=['researcher exact SHA approval,new unique output and AtomicLedger','implement versioned thin pilot entry;old trainer remains frozen','zero-update checkpoint/optimizer/RNG identity tests'],
  executable_entry=None,execution_command=None,source_sha256=bindings,formal_updates_this_turn=0,engineering_updates_this_turn=2)
 write(O/'SHORT_PILOT_RUNSPEC_DRAFT.json',pilot)
 doc('GO_NO_GO_MATRIX.md',"""
# Final decision
| Gate | Evidence | Decision |
|---|---|---|
| current identity vs G4-B/physical source_binding | source bytes match;HEAD/status/SHA preserved | PASS |
| historical startup root cause | zero IMU stamps,closed final gate,unconditional plant steps | DIAGNOSED;some raw signals NOT_RECORDED |
| versioned startup finite/fresh/authorization/ACK gate |36 unit contracts+C++ header test;native overlay compiled | PASS at static/unit level |
| actual ROS bootstrap,first integration,hover | forbidden this turn | NOT_RUN;physical GO withheld |
| actual live failure/stop/braking | not run on revised plant | NOT_RUN;motion/forest NO-GO |
| frozen dynamics audit |84 cases,TRAIN gradients only,24 active rawFD checks | PASS |
| peak directional qualification |10/10 violating TRAIN continuous counterfactuals improve | PASS |
| engineering shared-network update |exactly2,selected cases and TRAIN aggregate improve,dead rows protected | PASS limited engineering scope |
| currentK6 original numerical fixture |139 arrays bit-identical,old strict loads | PASS |
| scientific benefit/generalization/long training |no formal run | NOT_RUN |
| short peak-aware pilot |math and two-step evidence support a reviewed proposal | DESIGN ONLY;separate bound entry+approval needed |
| formal128 training/closed-loop forest |not authorized | NO-GO |
| historical protection |see HISTORICAL_PROTECTION.json | """+protection['status']+""" |

ENGINEERING_OPTIMIZER_STEPS=2
FORMAL_RESEARCH_TRAINING_STEPS=0
REAL_PHYSICS_EXECUTIONS=0
STARTUP_STATIC_QUALIFICATION=PASS
ACTUAL_PHYSICS_STARTUP=NOT_RUN
PEAK_PROXY_ZERO_UPDATE=PASS
PEAK_PROXY_ENGINEERING=PASS
K1_K6_NON_REGRESSION=PASS
READY_FOR_REVIEWED_SHORT_PILOT_PROPOSAL=YES
READY_FOR_FORMAL_TRAINING_NOW=NO
CONTROLLED_CLOSED_LOOP_READY=NO

No actual startup or braking PASS is inferred from static tests. No automatic training or physics follows this delivery.
""")
 doc('CODE_CHANGE_SUMMARY.md',"""
# Incremental source map
All changes are additions in the published whitelist;no original source file was edited.
- query_minco/integration/yopo_startup_v2/:finite/fresh content readiness,explicit HOLD authority,5 final ACKs,held-before-first-step C++ interlock.
- ros/g4c_startup_v2/prepare_hold_node.py:ROS HOLD-only bootstrap and complete signal/force telemetry.
- query_minco/training/peak_dynamics_proxy_v1/:Bernstein derivative bounds and diagnostic-only F_dyn replacement;no original objective edit.
- experiments/g4c_startup_and_dynamics_v1/:read-only timeline/frozen diagnostics,deterministic native-host overlay compiler,manual-only startup driver,exact2-step engineering probe,K6 replay,historical checker,report assembly.
- tests/g4c_startup_and_dynamics_v1/:startup fault contracts,peak math/FD/bound/normalization/dtype tests,header-only executable test.
- research_outputs/g4c_startup_and_dynamics_20261111_v1/:exclusive reports,logs,NPZ,generated build overlay and engineering-only checkpoints.

Original network,MINCO,K1 forest trainer,L0/P0c,Hard,all old manifests/checkpoints and YOPO/reach_planner/migration repositories are preserved. Native host overlay is generated from read-only source with a patch and SHA;the unchanged native dynamics library is linked instead of duplicating its algorithm. The generated source is build evidence,not a second authoritative planner.
Remaining:revised ROS adapter runtime,first hover motor lag,post-arm full fault campaign,physical tracking,full FP64 MINCO unsupported,and an exact-authorized production pilot entry. LegacyC1 registry issue remains BLOCKED.
""")
 # Every real tested process including failures has its original records.
 lines=['# Actual commands and outcomes','','Each directory contains PROCESS_START.json/PROCESS_RESULT.json,stdout.log and stderr.log. The real child exit code,not the outer wrapper exit,is authoritative. No failed attempt is removed.','','| log directory | real exit | elapsed s | peak sampled RSS bytes |','|---|---:|---:|---:|']
 process_rows=[]
 for file in sorted((O/'logs').glob('*/PROCESS_RESULT.json')):
  data=json.loads(file.read_text());start=json.loads((file.parent/'PROCESS_START.json').read_text())
  row=dict(directory=str(file.parent.relative_to(O)),argv=start.get('argv'),cwd=start.get('cwd'),exit_code=data['exit_code'],elapsed_s=data['elapsed_s'],rss=data.get('sampled_process_tree_peak_rss_bytes'))
  process_rows.append(row);lines.append('|'+row['directory']+'|'+str(row['exit_code'])+'|'+str(round(row['elapsed_s'],4))+'|'+str(row['rss'])+'|')
 lines+=['','Final qualified checks:36 Python tests+C++ header executable,139 exactK6 arrays,84 frozen proxy replays,24 active rawFD checks,10/10 continuous directional decreases,and exactly2 engineering updates.','',
 'NOT_RUN:actual ROS bootstrap/integration/hover/tracking/live faults;formal training;DEV gradients;TEST evaluation;full109-test historical pytest suite(pytest missing);FP64 MINCO(original adapter rejects).',
 'Earlier red attempts are test harness/fixture/build-output-claim failures documented in K1_K6_NON_REGRESSION.md;final PASS is bounded to the corrected,explicitly identified contracts.']
 doc('TEST_MATRIX.md','\n'.join(lines));write(O/'COMMANDS_AND_RESULTS.json',process_rows)
 # Preserve SHA identities of all preexisting protected files in ENTRY_PROTECTION,then hash final delivery files.
 write(O/'ASSEMBLY_SOURCE_IDENTITY.json',dict(schema='g4c_files_sha_v1',head=startup['current_head'],new_source_sha256={p:sha(p) for p in modules},
  current_baseline_sha256={p:h for p,h in entry['source_before'].items()},protected_history_manifest=str(O/'ENTRY_PROTECTION.json'),protected_history_manifest_sha256=sha(O/'ENTRY_PROTECTION.json'),
  output_sha256={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='FILES_AND_SHA.json'},source_binding_sha256=sha(O/'MANUAL_STARTUP_SOURCE_BINDING.json'),
  original_checkpoint_identities=diag['model_identities'],engineering_optimizer_steps=2,formal_research_steps=0,physics_execution='NOT_RUN'))
 print('G4C_REPORTS_ASSEMBLED_HISTORY',protection['status'])
if __name__=='__main__':main()
