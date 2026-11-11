"""Generate formal acceptance/science/control reports from immutable recorded evidence."""
from pathlib import Path
import sys,json,csv,hashlib,statistics,io
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import Counter
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4b_formal128_and_control_20261111_v1';A=R/'research_outputs/g4a_forest_shadow_20261111_v1';RUN=A/'formal/A_reasonable_seed20261104'
def put(name,text):
 data=(text.strip()+'\n').encode('utf-8');path=O/name
 if path.exists():
  if path.read_bytes()!=data:raise ValueError('PARTIAL_REPORT_DRIFT '+name)
 else:
  with path.open('xb') as f:f.write(data)
def csvwrite(name,rows):
 buffer=io.StringIO(newline='');w=csv.DictWriter(buffer,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);data=buffer.getvalue().encode();path=O/name
 if path.exists():
  if path.read_bytes()!=data:raise ValueError('PARTIAL_CSV_DRIFT '+name)
 else:
  with path.open('xb') as f:f.write(data)
def main():
 import numpy as np
 identity=json.loads((O/'FORMAL128_IDENTITY_ACCEPTANCE.json').read_text());paired=json.loads((O/'PAIRED_OFFLINE_RESULTS.json').read_text())
 ctrl=json.loads((O/'controller_consumption004/RESULT.json').read_text());history=json.loads((O/'HISTORICAL_PROTECTION.json').read_text())
 assert identity['status']==ctrl['status']==history['status']=='PASS'
 assert json.loads((O/'DELIVERY_VALIDATION002.json').read_text())['status']=='PASS'
 spec=json.loads((A/'K1_FOREST_TRAINING_RUNSPEC.json').read_text());info={x['sample_id']:x for x in spec['train']+spec['dev']}
 groups=[];samples=[];pairs=[]
 table=['|Step|Split|Loss|Hard+Task|Forest|Channel safe crossing|Direction-switch pairs|Max v/a/jerk|','|---|---|---:|---:|---:|---:|---:|---|']
 terms=['C_viol','F','G_task_train','C_shape','P','S']
 for step in [0,8,32,64,128]:
  evaluation=json.loads((RUN/('evaluation%03d.json'%step)).read_text())
  for split in ['train','dev']:
   rows=[x for x in evaluation['records'] if x['split']==split];forest=[x for x in rows if x['kind']=='forest'];mirror=[x for x in rows if x['kind']=='mirror']
   family={info[x['sample_id']]['family_id'] for x in mirror};direction_pairs=0;safe_pairs=0
   for f in sorted(family):
    members=[x for x in mirror if info[x['sample_id']]['family_id']==f]
    directions=[bool(x['crossings'] and all(c['correct_direction'] for c in x['crossings'])) for x in members]
    bothsafe=all(bool(x['hard_task_crossing']) for x in members)
    pairs.append(dict(step=step,split=split,family=f,samples=[x['sample_id'] for x in members],direction_correct=directions,
      direction_switched_both_sides=all(directions),both_safe=bothsafe,
      crossing_y=[[c['center_crossing_y_m'] for c in x['crossings']] for x in members]))
    direction_pairs+=int(all(directions));safe_pairs+=int(bothsafe)
   means={t:float(np.mean([np.array(x['terms'][t]).item() for x in rows])) for t in terms}
   reconstruct=means['C_viol']+means['F']+means['G_task_train']+.1*means['C_shape']+.025*means['P']+.1*means['S']
   loss=float(np.mean([x['loss'] for x in rows]));assert abs(reconstruct-loss)<1e-6
   maxima=[max(x['hard']['dynamic_polynomial_maxima'][k] for x in rows) for k in range(3)]
   failures=dict(Counter(reason for x in rows for reason in x['hard']['reasons']))
   row=dict(step=step,split=split,n=len(rows),loss=loss,**means,hard_task=sum(x['hard']['passed'] for x in rows),
    forest_pass=sum(x['hard']['passed'] for x in forest),forest_n=len(forest),channel_safe=sum(bool(x['hard_task_crossing']) for x in mirror),channel_n=len(mirror),
    direction_switch_pairs=direction_pairs,pair_n=len(family),safe_both_pairs=safe_pairs,speed_peak=maxima[0],acceleration_peak=maxima[1],jerk_peak=maxima[2],
    reasons=json.dumps(failures),loss_reconstruction_error=abs(reconstruct-loss))
   groups.append(row)
   table.append(f"|{step}|{split}|{loss:.8f}|{row['hard_task']}/{len(rows)}|{row['forest_pass']}/{len(forest)}|{row['channel_safe']}/{len(mirror)}|{direction_pairs}/{len(family)}|"+'/'.join(f'{x:.4f}' for x in maxima)+'|')
   for x in rows:
    h=x['hard'];samples.append(dict(step=step,sample_id=x['sample_id'],split=split,kind=x['kind'],loss=x['loss'],**{t:float(np.array(x['terms'][t]).item()) for t in terms},
     hard_task=h['passed'],safe_crossing=x['hard_task_crossing'],reasons='|'.join(h['reasons']),speed=h['dynamic_polynomial_maxima'][0],acceleration=h['dynamic_polynomial_maxima'][1],jerk=h['dynamic_polynomial_maxima'][2],
     progress=h['goal']['projected_progress_m'],goal_error=h['goal']['endpoint_goal_error_m'],terminal_speed=h['goal']['terminal_speed_mps']))
 csvwrite('FORMAL128_POINT_METRICS.csv',groups);csvwrite('FORMAL128_SAMPLE_METRICS.csv',samples)
 metrics=dict(groups=groups,mirror_pairs=pairs,loss_identity_verified=True)
 if (O/'FORMAL128_METRICS.json').exists():assert json.loads((O/'FORMAL128_METRICS.json').read_text())==metrics
 else:
  with (O/'FORMAL128_METRICS.json').open('x') as f:json.dump(metrics,f,indent=2)
 put('FORMAL128_FINAL_ACCEPTANCE.md',f"""
# Formal128 final evidence acceptance

FORMAL128_COMPLETED=YES;ACTUAL_NETWORK_UPDATES=128 for this forest Run;NEW_NETWORK_UPDATES_THIS_AUDIT=0.

Original run:{RUN}
True process:exit0,timed_out=false,elapsed340.416085706s,stdoutCOMPLETED. Terminal status COMPLETED_EXECUTION_NOT_SCIENTIFIC_SUCCESS. This is evidence of completed execution,not scientific success.

Exactly128 STEPS rows,continuous1..128;atomic attempted/completed/budget128,pending0,128 unique completed tokens matched per-step ledgers. All16 TRAIN tasks updated8 times in one shared model.128 before/after parameterSHA link continuously;all steps change live parameters. Checkpoints0/8/32/64/128 independently strict-loaded and matched sidecar/model/cursor and the stepSHA chain. Every saved optimizer tensorSHA/step counter matches its recorded update;options/seed/sidecar metadata matched frozen spec. All five full28-row evaluations replayed with exact JSON equality,including original Hard/Goal/MINCO checks. No optimizer was constructed during replay.

Raw15:21 zero gradients recorded all128 steps;774 dead parameter scalars have constantSHA and per-step zero deadmoments. Five actual checkpoints directly verify exact dead tensors and optimizer slices;intermediate claims rely on per-step digests and the frozen guard source,not unsaved full tensor dumps. Train-only membership and source control flow,DEVno_grad/gradient guard and evaluation dev_backward_calls0 support noDEVupdates;no independent historical autograd trace is invented.

Process-copied RunSpec SHA:{identity['runspec_sha256']}
Process-copied approval SHA:{identity['approval_sha256']}
Both match original files,checkpoint metadata and real process argv/cwd/stdout/stderr hashes. Frozen source/data identities pass current checks. Manual approval is separate from the frozen spec's initial informational formal_authorized=false field;authorization code requires the explicit exact approval and was not bypassed.

G4-A pretraining reports stating formal0 remain unchanged historical records. THIS report supersedes that old status for the completed forest run. G2 oldK1 historical process provenance remains a separate UNVERIFIED gap;the new G4 forest process has genuine recorded provenance.
""")
 termtable=['|Step|Split|C_viol|F|G_task|C_shape|P|S|','|---|---|---:|---:|---:|---:|---:|---:|']
 for x in groups:termtable.append(f"|{x['step']}|{x['split']}|"+'|'.join(f"{x[t]:.8f}" for t in terms)+'|')
 failtable=['|Step|Split|Rejection reason counts|','|---|---|---|']+[f"|{x['step']}|{x['split']}|{x['reasons']}|" for x in groups]
 put('FOREST128_TRAIN_DEV_COMPARISON.md',"""
# Scientific result:REGRESSED

Final128 compared with fixed preregisteredStep0:TRAIN Hard+Task8/16->4/16,DEV8/12->4/12. Forest TRAIN8/12->4/12,DEV8/10->4/10. Required-channel safe crossing remains0/4TRAIN and0/2DEV. MeanLoss improves .08129324->.05438915TRAIN and .03340808->.02320461DEV;it does not imply planning success. Primary safety/task metrics therefore REGRESSED rather than improved. Do not substitute Step8 despite its temporarily higher TRAIN success.

""" +'\n'.join(table)+"""

## Exact original component means

These are unweighted means. L0 coefficients remain1,1,1,.1,.025,.1. Reconstruction agrees to<1e-6;no weight or MINCO/Hard threshold changed.

""" +'\n'.join(termtable)+'\n\n'+'\n'.join(failtable)+"""

Reason counts may overlap per trajectory. CSV has every sample's loss/components,continuous polynomial speed/acceleration/jerk maxima,task progress/terminal errors and rejection reasons. All complete original evaluationJSONs and exact replays are preserved. Mirror-pair JSON distinguishes direction-sign response from both-side safe passage;no pair is certified learned routing merely from a small output difference.

Quality inside Hard-passing trajectories is available in completeJSON asS/P/C_shape. Feasible subsets differ across steps,so their conditional quality is not a paired ranking and cannot offset lost safety. Mapseed/family split matches frozen TRAIN16/DEV12;noDEVgradient or TEST-based parameter/Checkpoint selection. Shared tree asset,correlated same-pose Near/Far and prior diagnostic exposure limit scientific claims.
""")
 comp=['|Method|G3-D certified feasible Hard+Task|G3-D undetermined|G3-A safe crossing|','|---|---:|---:|---:|']
 records=paired['records'];csvrows=[]
 for method in ['fixed_forward','state_goal_quintic','old_K1_step128','forest_K1_final128']:
  f=paired['summary']['G3D_exposed_test'][method];m=paired['summary']['G3A_exposed_mechanism'][method]
  comp.append(f"|{method}|{f['CERTIFIED_FEASIBLE']['hard_task']}/20|{f['FEASIBILITY_UNDETERMINED']['hard_task']}/2|{m['CERTIFIED_FEASIBLE']['safe_crossing']}/4|")
 for x in records:
  h=x['hard'];csvrows.append(dict(group=x['group'],sample_id=x['sample_id'],classification=x['classification'],method=x['method'],loss=x['loss'],**{t:float(np.array(x['terms'][t]).item()) for t in terms},
   hard_task=h['passed'],safe_crossing=x['hard_task_safe_crossing'],speed=h['dynamic_polynomial_maxima'][0],acceleration=h['dynamic_polynomial_maxima'][1],jerk=h['dynamic_polynomial_maxima'][2],
   reasons='|'.join(h['reasons']),endpoint_y=x['endpoint'][1],forward_s=x['forward_s'],objective_minco_s=x['objective_with_minco_s'],hard_reference_s=x['hard_reference_s']))
 csvwrite('PAIRED_OFFLINE_RESULTS.csv',csvrows)
 newfails=Counter(reason for x in records if x['method']=='forest_K1_final128' and x['group']=='G3D_exposed_test' and x['classification']=='CERTIFIED_FEASIBLE' for reason in x['hard']['reasons'])
 put('NEW_VS_OLD_K1_OFFLINE.md',"""
# Paired final-only offline comparison

All four methods used the same frozen inputs,original A/MINCO/T2/L0/P0c/Hard,no optimizer or leaf optimization. Only the actual final128 was used for the new model;no bestCheckpoint search.

""" +'\n'.join(comp)+f"""

New model loses10 successes relative to oldK1 on20 preregistered certified-feasible G3-D snapshots. New failure reasons:{dict(newfails)}. Continuous speed failures dominate;original v3 threshold is not relaxed. Two undetermined snapshots remain a separate denominator,not certified impossible or mixed into20.

Both network models have unchanged tensorSHA/empty gradients after104 evaluations. FullRaw21/coefficients/durations NPZ and fullJSON preserve exact trajectories,task/crossing/dynamics,quality,timings and continuous position differences. G3-D raw numerical outputs for both fixed references and oldK1 reproduce prior results. New0/4 G3-A safe crossing does not support learned route switching. All mirrored directions/raw outputs are retained.

G3-D maps are held out by training mapseed but have prior diagnostic exposure and a shared tree asset;not virgin generalization data. G3-A is exposed mechanism diagnosis only. Do not claim new-map/asset/generalization success or tune Loss/initialization based on these results.
""")
 # Independent nominal short-reference numerical contract,not actual physics.
 import torch
 from query_minco.trajectory.state_goal_reference_v1 import reference_raw
 from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
 from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
 from query_minco.trajectory.minco_adapter import MincoAdapter
 from query_minco.evaluation.validation import norm_max
 torch.set_num_threads(1)
 with torch.no_grad():
  state=torch.tensor([[0,0,0,0,0,0,.4,0,0,0,0,-1]],dtype=torch.float32)
  p=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1)(reference_raw(state,'state_goal_quintic'),state)
  tr=MincoAdapter(num_candidates=1)(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
 maxima=[max(norm_max(c,t,k) for c,t in zip(tr.coeffs[0,0].numpy(),tr.durations[0,0].numpy())) for k in [1,2,3]]
 assert tr.solve_valid.all() and all(x<=limit for x,limit in zip(maxima,[1,8,100]))
 with (O/'NOMINAL_SHORT_REFERENCE_MATH.json').open('x') as f:json.dump(dict(status='PASS',actual_physics=False,T=2.,goal=[.4,0,0],maxima=maxima,network_updates=0),f,indent=2)
 cases=ctrl['cases'];ct=['|Fault|Observed delay s|Healthy controller Fx|Post-abort Fx|Final SO3 after abort|','|---|---:|---:|---:|---:|']
 for x in cases:
  d=x['result'];ct.append(f"|{x['case']}|{d['detection_delay_s']:.6f}|{d['raw_max_healthy_fx']}|{d['raw_max_post_abort_fx']}|{d['final_so3_after_abort']}|")
 put('ACTUAL_CONTROLLER_CONSUMPTION.md',"""
# LIVE_CONTROLLER_CONSUMPTION=PASS

Actual frozen SO3ControlNodelet was loaded through ROS Noetic nodelet and consumed real generated quadrotor_msgs/PositionCommand. Isolated master11342,synthetic stationary odometry/one-pixel depth liveness and a .02m reference were explicitly MESSAGE TEST ONLY. No quadrotor plant or flight process launched.

Six faults:timeout,depthloss,delayedodom,NaN,real request-process kill,real controller-process shutdown. Native rawSO3 had nonzeroFx for the desired reference;for the five live-controller cases,latched hold replaced oldPVA and actualFx became0 while odometry/controller continued. Controller-death correctly has no post-fault raw outputs. Independent subscriber observed final actuator topic locked with0 messages beyond abort+.05s.

""" +'\n'.join(ct)+"""

This is real controller consumption/command replacement evidence,not physical braking/tracking. Supervisor uses explicit --message-test-profile only for synthetic1x1 heartbeat;manual physical mode requires native90x160 validdepth and no test flag.

Baseline parameter audit found YAML gains/pos and gains/vel do not match onInit's gains/kx,gains/kv keys. Actual defaults are5.7/5.7/6.2 and3.4/3.4/4,consistent withFx=.02*5.7=.114. Gain overrides in forwarded messages arezero,so effective baseline gains,mass.98 and other physical params are preserved,not retuned. Baseline source/config/library SHA unchanged. This mismatch is disclosed,not silently repaired.

Attempt001's probe failed a list-minus-list assertion expression after the actual controller had run;all raw logs retained. Corrected harness002 passed;003 added continuity/deadline checks;final004 captures exact source/library binding and passes all6 cases. Final9 contract tests pass. Real exits and rawstdout/stderr remain separate from assertion conclusions.
""")
 put('CONTROL_WATCHDOG_AND_BRAKING_AUDIT.md',"""
# Watchdog and physical safety scope

MESSAGE_LEASE_FAULT_TESTS=PASS. OWNED_OS_CHILD_STOP_AND_GUARDIAN_DEATH=PASS. WATCHDOG_BRAKING=NOT_RUN for actual quadrotor physics.

New versioned Lease uses monotonic receipt deadlines plus ROSheader age/order. Timeout,invaliddepth,staleodom,NaN,plannerstop/controllerloss,PV discontinuity or planned/measured speed>1m/s latch ABORT. Late packets cannot revive expired leases. Fresh hold replaces oldPVA;internal rawSO3 is isolated from the final /g4b/so3_cmd gate. Closing publication alone is NOT called braking:the simulator also reuses lastSO3,so the plant must be independently stopped.

OwnedProcessStop verifies an exclusively owned process group,observes SIGSTOP state,then SIGKILL/wait proves terminal execution. Linux PR_SET_PDEATHSIG additionally binds direct physics child to guardian death. Both actions were tested on actual disposable sleeping OS children,NOT a quadrotor. Terminating a simulation is not hovering or real-aircraft braking.

Manual guardian independently checks supervisor heartbeat/process/controller/planner exits and owns the actual physics process. All services/camera/reference must be ready before plant launch to avoid dropping an uncommanded vehicle during Python startup. Fault schedule is written before launching physics. Guardian pauses/terminates on fault and records nativePID/exit/feedback. Simulator physical params remain frozen.

Manual safety requires all6 actual physics fault cases,realnative odom/controller feedback,observed plant STOP and terminalPID exit with fault-specific reason/detection deadline. Tracking validates those source-bound episodes and process records;message-only or fabricated-schema results reject. No manual physics stage was run this round. Thus physical stop/braking and tracking metrics remain NOT_RUN and controlled forest readinessNO.

Native nodelet ignores ABORT/age and simulator retains commands;no source modification pretends otherwise. The final implementation enforces1m/s execution independently of unchanged original Hardv3;no horizon scaling,threshold relaxation or checkpoint substitution. K1 execution not prepared until reference tracking passes.
""")
 put('FIRST_PHYSICS_TRACKING_RESULTS.md',f"""
# FIRST_PHYSICS_TRACKING=NOT_RUN

No real quadrotor physical process launched by this audit. No p/v/a tracking error,actual feedback/controller frequency,latency,switch discontinuity,aborted physical motion or collision outcome is reported as measured. Message-only controller tests do not fill those fields.

Prepared manual test:empty native floor map,initial0,0,2,original physical/controller params,single geometry-free State-Goal reference to+.4m with actual initialPVA,T2 and original3-pieceMINCO. Nominal rest-state reference mathematics passed:v/a/jerk maxima={maxima}. This is numerical feasibility only. Real starting states are recomputed and reject if limits fail,without clipping/time scaling.

The manual reference uses real native odom/world inertial acceleration,depth/camera liveness and fullbody transform. Images are not used for geometry-based selection;this reference does not use a neural network. Tracking logs actualPVA/targeterrors,plan+MINCOcost,sourceages;independent observer records nativeodom/IMU,request/rawcontroller/finalSO3 rates. Single reference construction is not a measured high-rate replanner.

Declared future track acceptance:full2sec trajectory plus settling,maxposition error<.15m,terminal<.1m,terminalspeed<.1m/s,actual+x progress>.25m;strict1m/s execution/measurement and independent abort gates. No K1 physicaltest CLI exists here. Existing K1 offline candidates may exceed1m/s;reject rather than scale them.

Real physical safety must PASS before the tracking command accepts evidence. Both manual stages remain unexecuted and require researcher review.
""")
 put('NEXT_MANUAL_CLOSED_LOOP_COMMANDS.md',"""
# Manual stages only - neither executed by this audit

First read CONTROL_WATCHDOG_AND_BRAKING_AUDIT and source binding. These commands really start a quadrotor simulation,so only researcher execution is authorized. Isolated master11343,emptyfloor,ownedprocess groups;no complex forest or K1 execution. Existing output/port rejects rather than cleanup/overwrite.

1. Six actual physical hover/fault cases. Emergency policy pauses+terminates owned simulation,not an assumed brake from stopping publication:

~~~bash
bash /home/lizheng/query_minco_mainline/research_outputs/g4b_formal128_and_control_20261111_v1/MANUAL_PHYSICS_SAFETY.sh
~~~

2. After reviewing a genuine PASS from step1,single actual-State-Goal tracking. Script/runner checks exact real safety episodes,sourceSHA,feedback and nativePID exits before plant launch:

~~~bash
bash /home/lizheng/query_minco_mainline/research_outputs/g4b_formal128_and_control_20261111_v1/MANUAL_STATE_GOAL_TRACKING.sh
~~~

Both use the explicit new research_outputs/g4b_researcher_physics_20261111_v1 directory. No /path placeholder. Do not delete/reuse failed output;register a new unique ID for any retry. Full environment/argv are inside scripts and real process logger wraps each. Bash syntax,CLIhelp,explicit-authorization refusal,source bindings and Python syntax passed;actual physical runtime remains NOT_RUN.

Only after reference tracking PASS may a separateK1 same-scene test be prepared. This audit provides none and never selects a model byTEST performance. Do not retrain,newseed sweep,modify L0/Hard or run complex forest autonomy.
""")
 logs=sorted((O/'logs').glob('*/PROCESS_RESULT.json'))
 tm=['# Test matrix and genuine argv','','|Run|True child exit|Elapsed s|Sampled tree RSS MiB|Assertion|','|---|---:|---:|---:|---|']
 for p in logs:
  d=json.loads(p.read_text());tm.append(f"|{p.parent.name}|{d['exit_code']}|{d['elapsed_s']:.3f}|{d['sampled_process_tree_peak_rss_bytes']/1048576:.1f}|"+('PASS' if d['exit_code']==0 else 'FAIL retained')+'|')
 tm+=['','|Item|Result|','|---|---|','|128 rowSHA/ledger/deadrow and5 full checkpoint/eval replays|PASS|','|Paired frozen inference104 trajectories|PASS,newmodel science REGRESSED|',
 '|Final six real nodelet consumption/fault cases|PASS,NO PLANT|','|New contract suite9 cases|PASS,no optimizer call|','|Owned processpause/kill and guardian-death OS tests|PASS,NOT PHYSICS|',
 '|Manual CLI/source/authorization/bash syntax|PASS|','|Actual quadrotor stop/braking|NOT_RUN|','|Actual physical tracking/rates/errors|NOT_RUN|','|K1 controlled forest/complex flight|NOT_RUN,NOT_READY|',
 '|New network optimizer updates|0|','|Old K6 regression|NOT_RUN this round,oldsource unchanged|','|Historical protection|PASS|','',
 'Native open sensor compilation allowed;its execution with real physics is NOT_RUN. No original RunSpec/approval/Checkpoint/Run changed. Sampled summedRSS can doublecount shared mappings and is not a continuous/private peak.','',
 '## Actual logged child argv','']
 for p in logs:
  d=json.loads(p.read_text());tm+=['### '+p.parent.name,'','~~~json',json.dumps(d['argv']),'~~~','',f"cwd={d['cwd']};exit={d['exit_code']};stdout={p.parent.relative_to(O)}/stdout.log;stderr={p.parent.relative_to(O)}/stderr.log.",'']
 tm+=['Original formal process record is copied into the evidence ZIP directly from G4A,not reconstructed. Per-service ROS folders contain real PID/argv/stdout/stderr/exit. Final004 SOURCE_BINDING binds actual message-test code and native library. Intentional plannerSIGKILL/controllerSIGINT are fault injections,not failed network training.','',
 'The initial failed assertion harness is retained. New validation002 and finalcontract suite apply to hardened final source;initial manual binding/validation preserved as development evidence. No tested physics PASS is invented from static validation.']
 put('TEST_MATRIX.md','\n'.join(tm))
 hc=history['counts']
 put('HISTORICAL_PROTECTION.md',f"""
# Historical protection PASS

Entry contentSHA:oldsource{hc['old_source']},oldoutputs{hc['old_outputs']};external1678 sourceSHA+298065 size/mtime;partialmigration80 fileSHA. No old file/branch/HEAD/index/diff changed. Only published whitelist added. Existing .gitignore preserved;no clean/reset/push,migration,YOPOsource/config/physicalparameter or oldRun overwrite.

All49538 old outputs include the now-completed formal128Run/process/approval. They remained immutable during this audit. Old G4A formal0 reports are pretraining history,not edited;new report records actual128. Full external trees use metadata+selectedsource,not a false claim of all-content hashing.
""")
 status=dict(FORMAL128_COMPLETED='YES',ACTUAL_NETWORK_UPDATES=128,TRAIN_DEV_SCIENTIFIC_RESULT='REGRESSED',MIRROR_ROUTE_LEARNING='NOT_SUPPORTED',
 LIVE_CONTROLLER_CONSUMPTION='PASS',WATCHDOG_BRAKING='NOT_RUN',FIRST_PHYSICS_TRACKING='NOT_RUN',CONTROLLED_FOREST_CLOSED_LOOP_READY='NO',HISTORICAL_PROTECTION='PASS')
 with (O/'FINAL_STATUS.json').open('x') as f:json.dump(dict(**status,NEW_NETWORK_UPDATES_THIS_AUDIT=0),f,indent=2)
 put('README.md',"""
# G4-B formal128 acceptance and controller audit

Actual formal forest128 completed and identity/math replay PASS. Scientific safety result REGRESSED:TRAIN8/16->4/16,DEV8/12->4/12;exposed G3D finalnew8/20 vsold18/20,refs16/20;mirror safe crossing0. No new network update this audit,no bestCheckpoint substitution.

Actual native nodelet consumed supervised commands and6 message-level faults passed. Physics stop/braking/tracking NOT_RUN;no quadrotor plant launched. Controlled forest closedloop NOT_READY. Manual safety then shortreference tracking commands prepared,review required. Parent-death and process-stop tests use disposable OS children,not a physics simulator.

Read requested7 reports,TEST_MATRIX,FILES_AND_SHA,rawlogs/CSV/JSON/NPZ and the independent audit ZIP. ZIP contains original5 checkpoints/evaluations/Ledger/STEPS/realprocess+approval/RunSpec and current source/evidence. Historical G4A formal0 report preserved,now superseded by this acceptance.

Large prior training/test geometry assets are referenced with source paths/SHA rather than silently rewritten or declared standalone-reproduced. Package is sufficient for artifact/source audit;independent full data replay additionally requires the original bound data assets/ROS dependencies. No GitHub push or actual forest flight.
""")
 print(json.dumps(status,indent=2))
if __name__=='__main__':main()
