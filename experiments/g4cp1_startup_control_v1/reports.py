"""Reports only;no ROS,physics,network training or optimizer."""
from pathlib import Path
import json,sys,hashlib,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp1_startup_control_20261111_v1'
def doc(name,s):
 with (O/name).open('x') as f:f.write(s.strip()+'\n')
def main():
 e=json.loads((O/'ENTRY.json').read_text());protect=json.loads((O/'HISTORICAL_PROTECTION.json').read_text())
 assert protect['status']=='PASS'
 new=[p for base in e['allowed_new'][:-1] for p in (R/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 binding={str(p):sha(p) for p in new}
 files=[O/'build_equilibrium_v3/gated_host',O/'build_equilibrium_v3/gated_host.cpp',O/'build_equilibrium_v3/plant_gate.hpp',
 R/'ros/g4b_control_supervisor_v1/controller_only.launch',R/'query_minco/integration/yopo_control_supervisor_v1/lease.py',
 R.parent/'YOPO/Controller/devel/lib/libquadrotor_dynamics.so',R.parent/'YOPO/Controller/devel/lib/libso3_control_nodelet.so',
 R.parent/'YOPO/Controller/src/so3_control/src/so3_control_nodelet.cpp',R.parent/'YOPO/Controller/src/so3_quadrotor_simulator/src/dynamics/Quadrotor.cpp',
 R.parent/'YOPO/Controller/src/so3_control/config/gains_hummingbird.yaml',R.parent/'YOPO/Controller/src/so3_control/config/corrections_hummingbird.yaml',
 R.parent/'YOPO/Simulator/src/src/test_simulator_cuda.cpp']
 sensor=R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build'
 files += [sensor/n for n in ['native_open_sensor','config.yaml','floor_only.ply','BUILD_IDENTITY.json']]
 binding.update({str(p):sha(p) for p in files});write(O/'MANUAL_SOURCE_BINDING.json',binding)
 doc('G4C_P1_CODE_FIX_AUDIT.md',"""
# G4C-P1 code-fix audit
PASS - read all six requested G4C reports in research_outputs/g4c_startup_and_dynamics_20261111_v1,plus current v2 readiness/header,ROS node,manual runner and generated host. Current HEAD4a2d962f599550d0e48f1b9dbfb127565d2c1594 is unchanged. ENTRY.json records real staged/unstaged status and hashes,including untracked sources. No Git reset/unstage/commit or migration was performed.

| Finding | Frozen failure evidence | P1 change | Verification |
|---|---|---|---|
| callback counts count repeated samples | v2 Python enters PREPARE after5 repeats of one timestamp | v3 strictly increasing source stamps;duplicates ignored without freshness/count refresh;regressions latch | PASS |
| PlantGate counts duplicate commands and permits direct ARM without PREPARE | old_gate_repro executable prints count5 and permit=true for one stamp | explicit HELD/PREPARE/ACK/ARMED/ABORT,prepare nonce,5 unique commands,ACK-only ARM | PASS |
| final ACK can name a command never sent by the supervisor | old v2 ACK set trusts any valid plant stamp | actual forwarded-stamp membership,5 distinct ACKs,strict plant phase progression | PASS |
| invalid/replayed command is copied into the command structure even when gate rejects | frozen generated callback does not branch on acceptance | bool accept and immediate return before command assignment;invalid gains/aux/frame/motors fail closed | PASS compile/header contracts |
| producer/consumer frame contract unchecked | v2 payload drops ROS frame IDs | explicit world/optical/native SO3 frame checks;exact depth-to-native-odom epoch match | PASS serialization/contracts;live NOT_RUN |
| hover can ARM without previous held qualification | v2 stages have separate flag but no previous-evidence binding | B requires actual A PASS,source/asset SHA,plant PID/result SHA,and identical motor initial condition | PASS implementation/static CLI checks;actual predecessor replay NOT_RUN |
| motorRPM0 with airborne p/v/a=hover is not equilibrium | native model gives initial thrust0 and acceleration about-g | zero cold-start retained;explicit separately accepted pre-spun equilibrium initial state | DESIGN_ONLY physical condition;compile/unit PASS |
| terminating publishers is not braking | frozen controller can reuse old PositionCommand | owned plant SIGSTOP then SIGKILL;native stale final-command interlock retained | PASS plain-process test;actual plant stop NOT_RUN |

The new authority is query_minco/integration/yopo_startup_v3/,ros/g4cp1_startup_v3/,experiments/g4cp1_startup_control_v1/. Original v2/G4B/YOPO code remains frozen. No network,MINCO,Hard,Loss,checkpoint or dataset is changed. A generated host is an auditable overlay linked to the original dynamics library,not a copied planner. New source and output whitelist is in ENTRY.json.
The optional equilibrium initial state changes only initial native rotor state;mass/gravity/inertia,motor lag,thrust/moment conversion and integration remain native. It is explicitly NOT ground spin-up. The build addendum corrects an intermediate unchanged-initialRPM metadata phrase;zero remains the default control condition.
PASS - old source/history protection;NOT_RUN - actual ROS and physical success. Zero optimizer/network updates this turn.
""")
 doc('STARTUP_HANDSHAKE_TESTS.md',"""
# Startup handshake qualification
PASS - final36 Python tests(16 new+20 existing G3A contracts),C++ header contracts,frozen-v2 duplicate reproducer,installed ROS message serialization and real SIGSTOP/SIGKILL on a plain Python sleeper.
PASS - native v3 host compiled/linked;no host executable was run.
NOT_RUN - ROS master,live message consumption,native first integration,actual suspension/termination of a physics process,hover or tracking. A synthetic protocol transcript/header counter is never called real physics feedback.

## Protocol
HELD: native state is explicitly HELD_NOT_PHYSICAL;no integration,command forward or ARM.
PREPARE: entered only after manual HOLD authority and5 distinct fresh observations per stream. Owner sends a prepare nonce;forwarding starts only after actual plant PREPARE.
ACK: native plant received5 strictly increasing valid final commands;Python independently associates5 observed ACKs with commands it actually published. All integration counters remain0.
ARMED: only a separately authorized hover owner sends an ARM nonce from ACK. Native integration starts and the supervisor requires ARMED telemetry with positive step count and DYNAMICS_INTEGRATED source.
ABORT: latched,never returns to a running mode. Invalid command cannot replace the stored command;owner stops the actual plant process. Duplicate samples do not count or refresh watchdogs. Source-stamp rollback,gap>.2s,zero stamp,nonfinite data,wrong frames or invalid physics identity reject.

## Actual Frame/time contract
| stream | ROS frame | meaning/timing |
|---|---|---|
|Odom|world;child /quadrotor|native integrated position/world velocity after ARM;unique native state epoch|
|IMU|world|native world inertial acceleration,not body specific force;stamp is assigned by producer at same odom epoch|
|Depth|camera_link|optical Z metres,160x90,valid in(0,20];Mask from finite positive returns;no-return remains unknown|
|CameraInfo|camera_link|K80/80/80/45,160x90;native calibration checked|
|Request|world|new timed stationary HOLD request from a fixed actual initial anchor|
|Raw SO3|/quadrotor|native message label;force numerically WORLD,orientation body-to-world;explicit source-derived exception,not TF rotation|
|Final SO3|same /quadrotor|unaltered accepted raw message;unique stamp,never receipt-time restamped|
|Plant ACK|world JSON|native final command stamp/count,phase,step count,mass,g,kf,tau,rpm input/actual,actual thrust|

Depth is accepted only if its exact native odom header epoch exists in the pose buffer;unmatched startup frames wait rather than invent a pose. No image-channel flip or identity rotation is used to substitute real geometry.
The native display TF camera translation(+.05m z) differs from renderer physical offset0 in the earlier audit;no TF-derived point cloud is used for this stationary HOLD. A later moving occupancy bridge must retain physical calibrated extrinsics and independently test dynamic projection.

Frame/time content,not Topic registration,is the prestart gate. Full raw/final force values and latency clocks are logged in EVENTS.jsonl. Native getControl retains projection onto R.col(2),Psi<1 thrust logic,rotor squared-speed allocation and clipping. Nominal world hover force[0,0,9.6138]N,nominal native RPM=sqrt(.98*9.81/(4*8.98132e-9));no rad/s or RPM unit conversion is inserted. Effective kx5.7/5.7/6.2,kv3.4/3.4/4 unchanged.
""")
 doc('PHYSICS_EXECUTION_GO_NO_GO.md',"""
# Physical execution gate matrix
| Item | Status | Consequence |
|---|---|---|
| frozen evidence/source identity | PASS | current SHA binding,untracked sources included |
| strict unique-stamp/state-machine contracts | PASS | allows researcher review of A command |
| compiled host/installed message types | PASS | executable/interface exists,not runtime proof |
| A actual HELD-before-integration | NOT_RUN | no physical startup success claimed |
| B actual static hover | NOT_RUN | requires A PASS for exact same sources AND initial motor condition |
| cold-start hover with zero RPM and actual a<=8 | FAIL analytical compatibility | default cold condition retained as control;hover runner refuses this combination |
| explicit equilibrium-rotor initial condition | DESIGN_ONLY physical qualification | manually accepted initial state,not a ground-spin-up result |
| C real fault injection and plant termination | NOT_RUN | no movement stage may proceed |
| D0.4m MINCO tracking | DESIGN_ONLY | blocked by B/C results;no D CLI claimed |
| E sparse-forest multicycle replanning/tracking | DESIGN_ONLY | blocked by D and qualified sensor-only occupancy/collision monitor |
| F frozen K1 shadow/execution comparison | DESIGN_ONLY | requires separate safe admission,no engineering weights |
| old K1/K6 Hard/UNKNOWN contracts | PASS | thresholds unchanged |
| historical protection | PASS |655 source files,50563 historical outputs,1678 external source hashes,298065 external metadata checked |
| new optimizer updates | PASS |0 |

Manual A/B driver implements only A and B. It cannot chain into C/D/E/F and cannot execute network trajectories. The final user decision to execute the command is the authorization;Codex has not launched it.
Actual state/trajectory limits remain speed<=1m/s for execution,acceleration<=8m/s^2,planned jerk<=100m/s^3,fixedT2. No startup acceleration exemption is applied. Frozen Hard limits3/8/100 remain intact;the1m/s execution admission is stricter.
If any manual stage fails,retain that unique output and stop. Do not tune gains,hide frames,increase limits or proceed to later stages.
""")
 doc('HOVER_AND_TRACKING_PLAN.md',"""
# Hover,fault and0.4m tracking plan
PASS - current implementation can prepare a manually authorized stationary HOLD and can accept a separate manual B invocation only after source-bound A evidence.
NOT_RUN - actual A/B execution. DESIGN_ONLY - C/D implementation/test campaign.

## Initial condition distinction
Native cold state:position[0,0,2],velocity0,motorRPM0. The same original physics yields acceleration near-9.81 initially. Even with immediate hover command,motor lag1/30s requires time to create thrust. Under constant hover input,no feedback/drag,the asymptotic velocity loss is1.5*g*tau=.4905m/s;this analytical estimate is not a physical run.
P1 optional equilibrium_hover sets only the initial native motor state to sqrt(m*g/(4*kf));all force/moment equations and motor lag remain native. This represents an already airborne,spun-up vehicle. It does not claim ground takeoff or motor spin-up validation. Both A and B require the explicit acceptance token and matching initial condition metadata. Zero mode remains available for HELD-only diagnosis;B zero mode is rejected rather than exempted from a<=8.

## A acceptance
Actual native sensor,controller and gated-host processes;five distinct Odom/IMU/Depth/Request/Raw/Plant samples;five matched final-command ACKs;native phase ACK;all integration counters0. ACCEPTANCE_A.json names the actual plant PID/process result SHA,source/asset SHA and initial condition. HELD state is not mislabeled integrated feedback.
No ARM token is sent in stage held.

## B acceptance
New exclusive output and fresh processes. Verify A evidence first;repeat HELD/PREPARE/ACK in this run;only then send explicit manual ARM. Require positive native integration count,DYNAMICS_INTEGRATED provenance,fresh real feedback for5s and>=200 distinct Odom samples.
Keep measured speed<=1,acceleration<=8,position drift<=.25;end position error<=.10,velocity<=.15. Log actual rpm/input/thrust and all p/v/a norms,source stamps and receipt times. Any violation immediately fails and stops the owned simulation. No fake Odom,Path-only success or prewritten kinematic stream can satisfy B.

## C fault campaign
Six new independent runs after a qualified B:Request timeout,Depth loss,Odom delay/rollback,NaN request/SO3,planner termination,controller death. Additional replayed final command and stopped guardian cases are recommended. Every run must establish real ARMED feedback first;fault onset is timestamped separately. Preserve valid pre-fault data,supervisor reason,last actual final-command consumption,native integration-stop time,SIGSTOP observation and actual process exit.
Keep.2s watchdog;report total fault-to-freeze latency and owner polling/scheduling overhead separately. Continuing old PositionCommand,continuing native integration or no observed terminal process state is FAIL. A plain-process unit stop is not a C PASS. Runtime injection adapter/campaign is still DESIGN_ONLY and must be versioned and tested before launch.

## D0.4m State-Goal MINCO
Only after B and every C case pass. Use actual settled Odom/IMU initial PVA and world goal current+[.4,0,0];fixedT2,three MINCO pieces,original math. Reuse state-goal reference/decoder/MincoAdapter rather than implementing a second solver.
For the ideal zero-PVA single quintic,the0.4m/T2 reference has peakv=.375m/s,peaka=.57735m/s^2,peakj=3m/s^3. These are analytical checks;the actual reconstructed MINCO and nonzero measured initial PVA must be checked separately.
Continuous extrema and start/end/C4 conditions must pass before publishing any command. Track p/v/a at controller rate100Hz,carry the original state/trajectory epoch,record stale rejection and switching discrepancies. Do not reset the epoch to conceal planning latency.
Proposed acceptance:progress>=.25m,terminal position error<=.10m,peak position tracking error<=.15m,terminal speed<=.10m/s,all dynamic/admission limits and safe terminal stop. No D launch command exists yet;none is fabricated in this delivery.
""")
 doc('FOREST_LOW_SPEED_CLOSED_LOOP_PLAN.md',"""
# Deterministic sparse-forest closed-loop plan
DESIGN_ONLY - not implemented or run. No neural method is required for first forest qualification. A/B must pass,then real C faults,then D tracking. New scene/source/config/output identities are mandatory;old YOPO data/config/checkpoints remain read-only.

1. Freeze one previously documented sparse-map asset/seed for engineering integration and label its previous exposure. Record cloud SHA,renderer calibration,world/body/optical transforms and obstacle volume. Do not choose a map using current planner success.
2. Native integrated Odom + native Depth/Mask feed the existing frame-contract/ForestAdapter interfaces. Buffer exact source poses,rotate with measured quaternion,validate dynamic projection;never use the display TF offset as physical camera translation.
3. Build conservative occupancy solely from valid observed rays;unobserved/occluded/padded/no-return space stays UNKNOWN. Use robot radius.12+.03 safety margin and conservative voxel/interpolation bounds. A whole MINCO curve plus tracking envelope must lie in known free space. If no qualified curve exists,fail-stop the simulation rather than force progress.
4. Start with a deterministic,goal/state-conditioned reference. At each5-10Hz planning cycle,reconstruct three-piece MINCO from actual PVA,T2,and apply execution speed<=1,acceleration<=8,jerk<=100. Online clearance/admission uses sensor-derived occupancy only,not GT Hard or witness route direction.
5. Reuse the same authoritative MINCO and sampling modules. Coordinate planning/state epochs,compensate measured latency transparently,check old/new p/v/a continuity before a switch. Initial proposed bounds from existing supervisor:.05m position,.1m/s velocity,.5m/s^2 acceleration discontinuity. These control-contract bounds do not change Hard thresholds.
6. Execute at measured controller frequency nominal100Hz with time-indexed p/v/a;at least20 successive replans and>=2m real physical progress are proposed engineering acceptance,not achieved outcomes. Record aborted cycles and all output-rejection reasons,not just accepted plans.
7. An independent observer uses frozen true map geometry ONLY for collision monitoring/emergency termination and offline metrics. It cannot select/rank/reference online paths. Include continuous swept-volume collision checks between feedback samples with explicit interpolation/uncertainty assumptions;report UNKNOWN when certification is unavailable.
8. Fail on lost/stale sensors,NaN,unknown occupancy,unsafe MINCO,late plan,controller death,excess tracking error or collision. Freeze/terminate the actual owned physics process;stopping Path/publication/setting ABORT is insufficient.
9. Log raw sensor stamps,source pose,plan start/end,MINCO and admission time,first/last executed command,state/control cadence,queue delay,trajectory-switch deltas,collision-clearance and terminal process identity. No latency/frequency value is claimed measured this turn.

F remains a separate later gate. First shadow fixed/State-Goal/frozen old/new K1 on the identical native input stream. Freeze original checkpoint path+SHA;engineering_peak_probe weights are prohibited. K1 candidates exceeding1m/s or any other execution check must be rejected unchanged;no time scaling or threshold relaxation. Offline GT Hard may evaluate them after the run but cannot select the deployed path. Current entrypoints cannot publish K1 motion commands.
""")
 doc('K2_K6_LEARNED_MULTIMODAL_DESIGN.md',"""
# K2/K6 future mechanism comparison
DESIGN_ONLY - no model,Head,Query,Anchor or Loss change and no training implemented.
Guidance read directly from Query-MINCO research-route V2 root documents00/01 in the local ZIP. Its four-module combination is lightweight vision,reliable candidate generation,kinematics-conditioned MINCO parameterization,and no-expert physical quality learning. The latter two have priority. Query/Transformer andK6 are not assumed final contributions.

## Prerequisites
K1 basic learning/continuous feasibility/quality must be qualified first. Existing128-step forest regression and failed mirror routing are not sufficient proof. Demonstrate that a well-trained K1 under equal capacity/compute cannot cover the actual decision ambiguity,then request a separate K2 implementation authorization. This study must not block physical integration.

## First candidate architecture
Shared CNN + StateEncoder -> shared visual/state feature z.
A motion/goal-conditioned reference Theta_ref(s,g) defines PVA-consistent MINCO parameters.
For interchangeable head k:
 Theta_k = Theta_ref(s,g) + D(s,g)*tanh(h_k(z,s,g)).
MINCO reconstructs each continuous trajectory from the same initial PVA,three pieces,T2.
Start with simpleK2 heads;K6 and learned-query decoders are later controlled comparators. No head is designated left/right,no fixed+-0.5m offset,no fixed aimpoint and no YOPO anchor clone is called a new method. Test parameter support independently of initialization;avoid applying the old absolute Raw21 tanh decoder a second time to physical residuals.
Preserve original six-query,YOPO anchor and historical left/right-reference experiments with their SHA as baselines.

## Collapse is a central open problem
Independent heads do not imply different safe modes. Identical zero residual heads give identical initial trajectories and often identical physical gradients. Shared features,one narrow valid corridor and the observed local collision-cost flat region all favor collapse.
Mean candidate cost can send every head to the same easiest mode. Min/softmin cost may starve losing heads. A distance repulsion can create unsafe or unknown trajectories and is not sufficient evidence of multimodality.
Possible later hypotheses,not selected implementations:small IID head-parameter perturbations without semantic direction;training-only unbiased stochastic exploration evaluated by original physical costs;alternating per-head gradient accounting;validity-conditioned coverage signals. Each requires preregistered seeds/amplitudes,TRAIN-only selection,zero-update gradient tests and comparison against equal-budget K1. Do not silently reinstate the G3-C failed symmetric-average Loss or fixed lateral initialization.
Report exact/near-duplicate rate,per-head gradient norm/update share,valid candidates per sample,strict feasible mode coverage,and whether candidate removal changes deployed success. A single narrow channel may legitimately yield one safe mode.

## Training privilege,Oracle and deployable Top-1
Training:all generated candidates may be evaluated with qualified GT geometry for physical costs/quality gradients. No GT witness is a trajectory target.
Offline Oracle@K:any candidate passing strict GT Hard+Task+crossing is a coverage upper bound,never online selection.
Deployment:score(feat(depth,mask,state),candidate parameters) only. A separately trained ranking/feasibility estimator may use stop-gradient physical cost ordering and ternary PASS/FAIL/UNKNOWN evaluation labels during training. The candidate must be explicitly represented to the scorer;an untrained Raw19:21 score is unusable.
Calibrate ranking regret,Top-1 precision,UNKNOWN handling and abstention on independent maps. A visual score is not a safety certificate. Sensor-only execution admission/fallback is still required. GT map must never enter online argmin or head assignment.

## Local YOPO/YOPO-MINCO distinction
Local native YOPO-MINCO uses anchor/primitive-conditioned candidates;the earlier interface audit records15 candidates and2 MINCO segments with different camera,time,dynamics and radius contracts. CurrentK1 uses one learned output and three fixed-time segments. These are local-code facts,not a fair performance comparison or a claim about every YOPO version.
The proposed interchangeable residual heads remove fixed directional identities but retain the general prior+residual/physics-learning ideas already present in YOPO-family methods. Newness would need demonstrated parameterization/quality/selection gains,not merely replacing anchors with learned embeddings.

## Controlled ablations and cost
Hold CNN,StateEncoder,reference support,MINCO,T2,data splits,family/map isolation,seeds,optimizer and total budget fixed.
Compare K1 vs simpleK2;thenK6 heads;then historical/free-query/anchor mechanisms. Separate support-range,initialization,loss aggregation and scorer changes into different factors.
Shared CNN cost is roughly paid once;head output,MINCO/physical checking grow approximately withK. Query self-attention adds aK^2 interaction term,while image cross-attention grows withK*80;actual CPU latency/memory must be measured. K2 is not guaranteed2x andK6 not guaranteed6x total runtime.
Compare both equal update counts and equal total candidate-evaluations/CPU wall budget. Report Oracle@K,visualTop-1,mirror switch,safe crossing,quality within valid trajectories,continuous dynamics,gradient distribution,CPU planning/control latency,and closed-loop failure modes. None of these experiments ran this turn.
""")
 doc('FINAL_DECISION.md',"""
# G4C-P1 final decision
PASS - frozen-v2 duplicate/authorization defects reproduced and new v3 contracts tested;new native host compiled;old source/history protected.
PASS -36 Python tests,header-only executables,installed ROS message serialization,plain-process stop and CLI authorization rejection.
NOT_RUN - any ROS master or native physics execution by Codex.
NOT_RUN - actual A HELD and B hover outcomes;manual commands are supplied with exact source/binary/asset binding and prior-stage evidence enforcement.
DESIGN_ONLY - C live fault campaign,D0.4m MINCO tracking,E sparse forest multicycle execution,F K1 shadow/execution;these gates cannot be marked passed from unit tests.
DESIGN_ONLY - explicit equilibrium-rotor initial condition as a physical experiment. It avoids the analytical cold-RPM mismatch without changing native dynamics;it is not a completed ground-startup or hovering result.
DESIGN_ONLY - K2/K6 architecture/anti-collapse/ranking plan;K1 training qualification and necessity evidence still gate future implementation.
PASS - no new network/optimizer updates,no formal training,no engineering weights used,no baseline/dataset/Hard changes.
PASS - no automatic progression to motion or training.

CODE_FIX=PASS
STATIC_HANDSHAKE=PASS
HOST_COMPILE=PASS
ACTUAL_HELD=NOT_RUN
ACTUAL_HOVER=NOT_RUN
ACTUAL_FAULT_INJECTION=NOT_RUN
ACTUAL_04M_TRACKING=NOT_RUN
ACTUAL_FOREST_CLOSED_LOOP=NOT_RUN
K2_K6_IMPLEMENTATION=NOT_RUN
NEW_NETWORK_UPDATES=0
HISTORICAL_PROTECTION=PASS

Next action is researcher review and manual A only. B is a separate optional manual command after A PASS using the same explicit motor initial condition. If either fails,stop and preserve its directory;do not execute C/D/E/F.
""")
 commands="""# Manual A/B commands
NOT_RUN - these processes have not been launched by Codex. The CLI and rejection paths were executed without ROS launch.
The default zero-RPM mode remains available for cold HELD diagnosis. It cannot enter B under the acceleration contract.
The commands below explicitly choose an already-spun equilibrium motor initial state. This is NOT ground takeoff/spin-up and must be accepted as a distinct experiment. Original dynamics parameters,Hard and measured limits remain unchanged.

A:prepare and acknowledge,without any integration or ARM:
~~~bash
cd /home/lizheng/query_minco_mainline
HELD="research_outputs/g4cp1_manual_held_$(date +%Y%m%d_%H%M%S)_$$"
PYTHONDONTWRITEBYTECODE=1 /home/lizheng/anaconda3/envs/yopo/bin/python -B experiments/g4cp1_startup_control_v1/manual_startup.py --stage held --allow-held-processes RESEARCHER_MANUAL_HELD_ONLY --motor-initial-state equilibrium_hover --accept-equilibrium-initial-condition RESEARCHER_ACCEPT_PRESPUN_INITIAL_CONDITION --output "$HELD"
~~~
Inspect "$HELD/ACCEPTANCE_A.json":status PASS,stage A_HELD,source SHA binding,motor_initial_state equilibrium_hover,at least5 distinct ack_stamps,max_integration_steps0 and real owned plant PID/terminal record. The expected state source is HELD_NOT_PHYSICAL.

B:separate manual invocation ONLY after inspecting A PASS. In a new shell,set HELD to the actual successful A directory first;do not type a placeholder path.
~~~bash
cd /home/lizheng/query_minco_mainline
HOVER="research_outputs/g4cp1_manual_hover_$(date +%Y%m%d_%H%M%S)_$$"
PYTHONDONTWRITEBYTECODE=1 /home/lizheng/anaconda3/envs/yopo/bin/python -B experiments/g4cp1_startup_control_v1/manual_startup.py --stage hover --allow-held-processes RESEARCHER_MANUAL_HELD_ONLY --motor-initial-state equilibrium_hover --accept-equilibrium-initial-condition RESEARCHER_ACCEPT_PRESPUN_INITIAL_CONDITION --allow-first-integration RESEARCHER_MANUAL_HOVER_ONLY --prior-held "$HELD/ACCEPTANCE_A.json" --output "$HOVER"
~~~
B requires actual integrated feedback,5s hover,measured v<=1,a<=8,and tracking bounds. It fails closed on any error. No C/D/E/F command is emitted or automatically chained. Runtime ROS/CUDA/renderer behavior remains NOT_RUN until these manual tests occur.
"""
 doc('MANUAL_EXECUTION_COMMANDS.md',commands)
 rows=[]
 for p in sorted((O/'logs').glob('*/PROCESS_RESULT.json')):
  d=json.loads(p.read_text());s=json.loads((p.parent/'PROCESS_START.json').read_text())
  rows.append(dict(directory=str(p.parent.relative_to(O)),argv=s['argv'],cwd=s['cwd'],exit_code=d['exit_code'],elapsed_s=d['elapsed_s'],peak_rss=d['sampled_process_tree_peak_rss_bytes']))
 write(O/'COMMANDS_AND_RESULTS.json',rows)
 doc('TEST_MATRIX.md',"# Actual test commands\n\nPASS - raw argv,cwd,stdout,stderr,true child exit and process-tree resources are in COMMANDS_AND_RESULTS.json and logs/.\n\n|log|exit|elapsed s|peak RSS bytes|\n|---|---:|---:|---:|\n"+'\n'.join('|'+x['directory']+'|'+str(x['exit_code'])+'|'+str(round(x['elapsed_s'],4))+'|'+str(x['peak_rss'])+'|' for x in rows)+"""
 
PASS - final_tests/STATIC_TEST_RESULT.json and header_tests/*.json define the exact final test scope. Earlier35-test pre-initial-condition run is retained.
FAIL - frozen v2 defect reproductions are expected historical failures;they are not new physical test failures.
NOT_RUN - actual ROS startup,ARM,hover,faults,tracking,forest,K1 execution,training. The only real process terminated in integration tests was a plain Python sleeper.
DESIGN_ONLY - motor cold-response prediction/equilibrium initial condition physical qualification. No native dynamics step was executed.
""")
 # Do not write final output hashes while this report process is still writing its own stdout.
 write(O/'ASSEMBLY_IDENTITY.json',dict(source_binding_sha256=sha(O/'MANUAL_SOURCE_BINDING.json'),new_sources={str(p):sha(p) for p in new},network_updates=0,physics_executions=0))
 print('P1_REPORTS_COMPLETE',len(new),'SOURCE_FILES')
if __name__=='__main__':main()
