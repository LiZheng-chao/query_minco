"""Generate reports from recorded evidence only; no network or optimizer."""
from pathlib import Path
import json,hashlib,statistics,csv
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def put(name,text):
 with (O/name).open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')
def main():
 q=json.loads((O/'dataset/FOREST_QUALIFICATION.json').read_text())
 idx=json.loads((O/'dataset/FOREST_INDEX.json').read_text());maps=json.loads((O/'dataset/MAP_INDEX.json').read_text())
 result=json.loads((O/'FOREST_RESULTS.json').read_text());metric=json.loads((O/'FOREST_DESCRIPTIVE_METRICS.json').read_text())
 protection=json.loads((O/'HISTORICAL_PROTECTION.json').read_text());entry=json.loads((O/'ENTRY_PROTECTION.json').read_text())
 assert protection['status']=='PASS'
 for name in ['forest_evaluation_process','forest_prepare_process','forest_contract_tests_process_v3','hard_k1_k6_regression_process','forest_metrics_process_v2']:
  assert json.loads((O/'logs'/name/'PROCESS_RESULT.json').read_text())['exit_code']==0
 extra={}
 for name in ['config/config.py','policy/yopo_dataset.py','policy/yopo_network.py','policy/poly_solver.py']:
  p=R.parent/'YOPO/YOPO'/name;extra[str(p)]=sha(p)
 with (O/'YOPO_ADDITIONAL_READONLY_SOURCE_SHA.json').open('x') as f:json.dump(extra,f,indent=2)
 repos=[]
 for path,commands in entry['repositories_before'].items():
  repos.append('- '+path+': '+json.dumps({k:v['stdout'].strip() for k,v in commands.items() if 'HEAD' in k or 'branch' in k}))
 put('YOPO_FOREST_SOURCE_AND_INTERFACE_AUDIT.md',"""
# Live YOPO forest source and interface audit

No migration, policy training, optimizer construction/step, leaf optimization, ROS node, controller, Git mutation or push was executed. Live source, including important untracked physics code, is authoritative.

## Actual repositories

""" +'\n'.join(repos)+"""

Exact Git argv, exits, status, staged/unstaged diffs are in ENTRY_PROTECTION and HISTORICAL_PROTECTION. Existing staged files were preserved. The partial query_minco_planner migration was protected and not resumed. YOPO_SOURCE_IDENTITY and YOPO_ADDITIONAL_READONLY_SOURCE_SHA bind read-only sources; FILES_AND_SHA indexes all new source and dependencies.

## Assets and side effects

Live Simulator/src/config/config.yaml:maze_type5,seed3,random_map=true,x/y60m,z15m,resolution.1m,tree_dist4m. The existing native pointcloud asset tree.ply has227969 points. No reusable forest.ply or dataset was found at the explicit audited native dataset/pointcloud locations (GENERATOR_SIDE_EFFECT_AUDIT.json). Existing mainline1001/1102/2102 native exports are training-exposed single-tree maps, not independent multi-density holdout forests.

Maps::generate(5)->forest->generatePoissonPoints->loadPLYFile->in-memory transformed trees->generateGround has no save/delete/collector call. Despite its name, the sampler is a seeded jittered grid. Tree roll/pitch0..10deg,yaw0..360deg,scale.5..1.0. Native collector save_path=../dataset is not invoked.

New forest_generator_host_v1.cpp compiles current read-only YOPO maps.cpp/perlinnoise.cpp into a new output binary. It does not call ros::init or build inside YOPO. It refuses a nonempty destination before generation; output uses O_CREAT|O_EXCL and fsync with a six-million-point cap. Occupied-output sentinel regression verifies rejection and unchanged bytes/mtime. No native config or generated asset was touched.

## Preregistered benchmark scope

Six TEST-only map seeds:3101/3102 sparse spacing6m,3201/3202 medium4m,3301/3302 dense3m,nominal tree counts4/9/16. In-memory generation width12m differs explicitly from native60m. Actual bounds include transformed canopy extents. This is a bounded local forest benchmark, not the complete60m native benchmark or an unseen-tree-asset test.

Two anchors/map,two tasks/pose. First initial-clearance-qualified position among128 fixed nearest grid candidates,z1.8m,clearance>=.6m,chosen without policy/witness success. Assigned offline initial v/a are [.8,0,0]/[.1,0,0] or[1.2,0,0]/[-.1,0,0]; not measured simulator odometry. Near goal[2.4,0,0],Far[6,0,0],gravity[0,0,-1].

## Entry call graph

~~~mermaid
flowchart TD
 S[Read-only native generator and tree] --> H[Exclusive host]
 H --> C[Raw points and exact occupancy compaction]
 C --> D[Frozen native CUDA depth renderer]
 D --> A[ForestAdapter Depth Mask State]
 A --> N[Frozen K1 Step0 or Step128]
 A --> B[Geometry-free fixed/state-goal reference]
 N --> M[Original FixedTimeDecoder and MINCO]
 B --> M
 C --> G[Original typed geometry and finite KnownROI]
 M --> L[Original L0 and P0c evaluation]
 M --> V[Hard v2 through generic K1 K6 helper]
 G --> L
 G --> V
~~~

Networks and deterministic references receive only observation/state. Geometry and witnesses never enter network input or online selection. The original128-step artifacts are SHA-verified; their historical process provenance remains UNVERIFIED. This new recorded process cannot retroactively repair it.
""")
 put('FOREST_ADAPTER_VALIDATION.md',"""
# ForestAdapter input/physics validation

FOREST_ADAPTER=PASS for the supported identity-body,zero-camera-pitch profile. Arbitrary pose rotations,full60m maps and high-speed native states are NOT_RUN/unsupported, not implied by PASS.

Component strict_yopo_forest_voxel_equivalent_adapter_v1. Frozen native renderer SHA7cb7ead45a858da5f56fc0d9ebc3b9c35a6274ede690f54c58564fcae5b0069b. Dense raw clouds exceed its3-million-point limit. Keep one actual point per occupied native voxel plus all extrema: original/compact grid bounds,origin,shape,count>0 exactly equal for all6 maps. Multiplicities intentionally differ; equivalence applies only to current threshold0. Full raw points,FULL_SOURCE_COUNTS,COMPACT_COUNTS,native GPU counts and COMPACTION_PROOF remain available.

Native index=floor((point+.001-origin)/.1),XY symmetricIndex period2*n-2,z<=0 solid floor. KnownROI is only finite base bounds inward1e-5,never extended by mirror/no-return rays. Outside/top is UNKNOWN. Occluded source geometry is privileged offline truth, not inferred free from depth.

Optical coordinates right/down/forward ->B0FLU via[[0,0,1],[-1,0,0],[0,-1,0]]. Camera/body origins coincide,R_WB0=I; unsupported rotation rejects. Raw opticalZ is meters,90x160,fx=fy80,cx80,cy45,max20m,pitch0. Independent NumPy native-ray replay <=2e-5m per camera. Pad19 rows top/bottom,shift cy64; input[1,2,128,160] is Z/20 and validMask. Valid iff .04<=Z<20; max-range/no-return and padding have mask0/depth0. A valid pixel does not certify a free trajectory.

Strict manifests bind raw source,map/config,camera/pose,raw/processed depth,mask,state,counts,qualification,adapter SHA. Full source chain checked at load; operational hashes and memory fingerprints revalidated during physics. No historical manifest rewritten or SHA bypassed. Original BoundBatch and typed AABB SDF feed original RoiSafeTrainingObjective/L0 and independent Hard. UNKNOWN outside ROI raises.

Original A position support x(0,10),y(-4,4),z(-2,2),actual-state initialPVA,Raw21,three quintic MINCO pieces,T2,dt2/3,v3/a8/jerk100,radius.12+margin.03 unchanged. Far means progress,not arrival6m.

Geometry-free references: fixed-forward samples nominal rest-to-rest2.4m knots with actual initialPVA in MINCO. State-goal samples the actual-initial-PVA quintic to min(goal length,2.4m),terminal v/a0. Exact inverse A tanh maps physical knots into Raw21. Out-of-support/backward/invalid states reject,not clip. No map-dependent bias or optimization.

New suite9 tests PASS: native/source occupancy,synthetic extrema,hash/schema rejection,depth/mask/camera/UNKNOWN,prior witness PVA/C4,original MINCO/illegal duration,denominator/frozen identities and native occupied-output guard. Existing G3-A suite20 PASS includes K1/K6 full-result equality,illegal shape/time/nonfinite,independent MINCO,UNKNOWN,crossing,provenance/Ledger.

Preserved harness failures: first metrics import lacked project sys.path (exit1); first occupied-output test lacked native LD_LIBRARY_PATH (binary127,suite1). Only new harness paths were corrected; successful new-directory reruns pass. No historical/physics source changed.
""")
 table=['|Map|Spacing m|Trees|Source / compact points|Occupied voxel fraction|','|---|---:|---:|---:|---:|']
 for m in maps:
  mm=json.loads(Path(m['map_manifest']).read_text())
  table.append(f"|{m['map_id']}|{mm['map']['tree_dist_m']}|{mm['tree_count']}|{m['source_points']} / {m['compact_points']}|{m['occupied_fraction']:.5f}|")
 snapshots=['|Snapshot|Class|Initial clearance m|Valid depth fraction|Finite feasible directions|','|---|---|---:|---:|---|'];csvrows=[]
 for row in idx:
  m=json.loads(Path(row['manifest']).read_text());s=json.loads((Path(m['folder'])/'PROBLEM_QUALIFICATION.json').read_text())
  ys=sorted({w['y'] for w in s['feasible_directions']});zs=sorted({w['z'] for w in s['feasible_directions']})
  snapshots.append(f"|{row['snapshot_id']}|{row['classification']}|{s['initial_clearance_lower_bound_m']:.4f}|{s['depth_valid_fraction']:.4f}|{len(s['feasible_directions'])}/15;y={ys};z={zs}|")
  csvrows.append(dict(**row,initial_clearance_m=s['initial_clearance_lower_bound_m'],nearest_obstacle_m=s['nearest_initial_obstacle_m'],depth_valid_fraction=s['depth_valid_fraction'],no_return_fraction=s['no_return_fraction'],finite_feasible_count=len(s['feasible_directions']),y_offsets=json.dumps(ys),z_offsets=json.dumps(zs)))
 with (O/'FOREST_SCENE_QUALIFICATION.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 put('FOREST_SCENE_QUALIFICATION.md',"""
# Independent forest qualification

6 maps,planned24 snapshots,input-qualified22,independently CERTIFIED_FEASIBLE20,CERTIFIED_INFEASIBLE0,FEASIBILITY_UNDETERMINED2,excluded2. FOREST_QUALIFIED_SCENES=20 means pre-certified feasible snapshots,not22 inputs or6 independent maps.

Model construction was blocked until qualification completed. Fixed15 witness bank:y={-.8,-.4,0,.4,.8}m,z={-.3,0,.3}m inner offsets,actual headPVA,original independent numerical MINCO,T2 and unchanged Hard. This finite proof is not leaf optimization,label fitting,model input or online selection. Witness pass certifies feasibility. No witness found means UNDETERMINED; no infeasibility proof was attempted.

Excluded medium3202 pose1 Near/Far:NO_CLEARANCE_POSE_AMONG_REGISTERED128. No density/map was adjusted based on policy output. Dense3302 pose1 Near/Far remain UNDETERMINED:all15 failed. Four methods subsequently failed there; denominator remains fixed.

Density is nominal spacing,not monotone local difficulty. Actual bounds/voxel fractions vary with seeded tree transforms. Finite inner-offset directions reach the same local endpoint; not exhaustive topology/reachability or GT policy. All maps TEST-only,seed is isolation unit.11 poses and shared-depth Near/Far tasks are correlated; same tree asset limits generalization.
""" +'\n\n'+'\n'.join(table)+'\n\n'+'\n'.join(snapshots))
 summary=metric['summary'];rows=metric['records']
 comparison=['|Method|Hard+Task|Sparse|Medium|Dense|Near|Far|','|---|---:|---:|---:|---:|---:|---:|']
 timing=['|Method|Forward/reference median ms|MINCO median ms|Hard+reference median ms|Max v/a/jerk|','|---|---:|---:|---:|---|']
 quality=['|Method|Feasible mean length m|Feasible mean integral jerk squared|Mean sampled surface clearance m|Mean continuous lower bound m|','|---|---:|---:|---:|---:|']
 for method in ['fixed_forward','state_goal_quintic','K1_step0','K1_step128']:
  s=summary[method];d=s['by_density'];t=s['by_task'];tm=s['timing_ms'];qual=s['feasible_quality_mean']
  comparison.append(f"|{method}|{s['hard_task']}/{s['n']}|"+'|'.join(f"{d[k]['hard_task']}/{d[k]['n']}" for k in ['sparse','medium','dense'])+'|'+ '|'.join(f"{t[k]['hard_task']}/{t[k]['n']}" for k in ['near_stop','far_progress'])+'|')
  timing.append(f"|{method}|{tm['network_or_reference_s']['median']:.3f}|{tm['minco_decode_s']['median']:.3f}|{tm['independent_hard_and_reference_s']['median']:.3f}|"+'/'.join(f'{v:.3f}' for v in s['peak_dynamics'])+'|')
  quality.append(f"|{method}|{qual['path_length_quadrature_m']:.4f}|{qual['jerk_squared_integral_quadrature']:.3f}|{qual['surface_clearance_sampled_m']:.4f}|{qual['surface_clearance_continuous_lower_bound_m']:.4f}|")
 failures=[]
 for method in summary:
  for f in summary[method]['failures']:failures.append(f"- {method}: {f['snapshot_id']};{f['reasons']};center clearance sampled={f['clearance_sampled']:.5f}m.")
 ys=[]
 for method in ['K1_step0','K1_step128']:
  vals=[r['endpoint_y_m'] for r in rows if r['method']==method and r['classification']=='CERTIFIED_FEASIBLE']
  ys.append(f"{method} endpointY mean/min/max={statistics.mean(vals):.5f}/{min(vals):.5f}/{max(vals):.5f}m.")
 put('K1_FOREST_OFFLINE_RESULTS.md',"""
# Frozen K1 forest results

Step128 Hard+Task18/20 on pre-certified feasible snapshots. References/Step0 each16/20. All methods0/2 on separately UNDETERMINED inputs. torch.no_grad,optimizer construction/step guards active; both model tensor SHA and empty grad buffers unchanged. No leaf optimization or GT/Hard online selection. Every method outputs one Raw21 trajectory.

Original Hard passed already includes Goal/Task. CSV also reports physical_hard_without_task by excluding only GOAL_TASK from reasons; here counts coincide. No criterion changed.

""" +'\n'.join(comparison)+"""

Step128 gains4 snapshots at medium3201 but loses dense3302 pose0 Near/Far compared with the other methods. Net+2 is not uniform improvement. All failures are collision/required-margin violations,none dynamics or UNKNOWN. Its two known-feasible failures have center clearance .074/.078m,below .12+.03. All88 trajectories have0 Hard UNKNOWN pieces and0 P0c outside-center fraction in this selected finite set. Out-of-ROI UNKNOWN rejection was separately tested.

""" +'\n'.join(failures)+'\n\n'+' '.join(ys)+"""

NegativeY bias remains. All goals are+X and low-speed states vary little; these results do not establish visual route switching,full60m capability or generalization. Prior mirror0/4 findings are not overturned. Six seeds share one tree asset; only11 poses and correlated Near/Far tasks.

## Clearance and feasible quality

Existing Hard remains authoritative. Additional global metric evaluates frozen polynomials at201 points/piece using exact original AABB/floor/ROI SDF. Continuous center lower bound=min sampled distance - polynomial vmax*max sample gap/2 -1e-6,from1-Lipschitz distance. Surface subtracts radius.12. All passing trajectories also satisfy this additional .15 center bound. Negative bound alone is not collision proof; sampled violations separately identify collisions/margin failure.

Hard broadphase clearance fields are not interpreted as global nearest-obstacle minima. No validator changed. Path/derivative integrals below are quadrature,not replacement acceptance. Quality means condition on each method's own passing subset,not a paired ranking.

""" +'\n'.join(quality)+"""

## Timings and resources

CPU FP32,single-thread,CUDA hidden for networks. Native depth rendering alone uses CUDA. No timing warmup/repetition; cold maxima are retained. Values include integrity checks. Process logs record sampled summed process-tree RSS,not continuous/private memory peaks.

""" +'\n'.join(timing)+"""

Hard includes independent numerical MINCO reference and repeated manifest checks. Objective timing includes another MINCO decode and is not additive with isolated decode for a deployed planner. Do not sum component medians into a median total. FOREST_DESCRIPTIVE_METRICS.json/CSV preserve per-sample progress,goal error,terminal speed,quality,peaks,clearance,UNKNOWN and timings. FOREST_TRAJECTORIES.npz retains Raw21,coefficients,durations. No backward or new update was executed.
""")
 put('YOPO_MINCO_BASELINE_FAIRNESS.md',"""
# Native YOPO-MINCO comparability

YOPO_BASELINE_COMPARABLE=NO;native policy NOT_RUN. Reusing native generator/rendering does not make policy contracts equivalent.

|Contract|Live native YOPO-MINCO|K1|
|---|---|---|
|v/a/jerk|6/6/60|3/8/100|
|MINCO|2 pieces|3 quintic pieces|
|Horizon|nominal10/6s;native decode timing|fixed2s|
|Candidates|5x3=15 anchored primitives|true1|
|Selection|learned score/corridor|Raw19:21 unused,no selector|
|Depth|1 channel96x160,16bitPNG /65535,nearest resize|2 channels128x160,Z/20+mask,pad|
|Frame|native level-yaw transform and sampler|explicit identityB0FLU12D|
|Safety|native r.6,d0=1.2,d_safe2|Hard radius.12+margin.03|
|Map|native expanded bounds/top|finite base ROI and UNKNOWN|

Sources:live traj_opt.yaml,config.py,primitive.py,state_transform.py,yopo_dataset.py,yopo_network.py,poly_solver.py,all SHA recorded. No policy module imported or native checkpoint/config changed.

A future native offline bridge needs explicit PNG/mask/state and frame qualification,native two-piece timing/PVA checks,and common independent geometry/task evaluation. Native-constraint performance can be a separate setting. Forcing limits/horizon/radius to K1 changes method identity and requires independent qualification,not silent equivalence. Not implemented this round.
""")
 put('K2_MULTIMODAL_HYPOTHESIS.md',"""
# K2 physical double-candidate hypothesis - design only

READY_FOR_K2_PHYSICAL_FEASIBILITY=YES means readiness for a separately authorized physical feasibility study,not a network/training/selector authorization. K2_MODEL_IMPLEMENTED=NO;network updates0.

G3-C found local flatness and insufficient stable direction under symmetric nonlocal mean. Independent two-mode physical initialization is a separate representation/initialization hypothesis. Preserve backbone,Raw21 per candidate,A support,actual headPVA,terminal rules,3-piece MINCO,T2,L0/P0c,Hard and family/seed identities.

Future physical references:original A_reasonable inner points P_i plus s*delta*e_y,i=1,2,s=-1,+1,terminal unchanged. Use exact inverse A tanh; reject out-of-support/nonfinite. Do not add meter-valued offsets to logits or call duplicated K1 outputs a trainedK2. Preregister delta=.5m from old TRAIN diagnosis only; no DEV/forest pass-rate sweep. B1 and A support expansion remain separate,unimplemented hypotheses.

## Prior physical coverage evidence

K2_PRIOR_REFERENCE_COVERAGE.json verifies old G3-B LATERAL_COST_HARD_SCAN SHA. Fixed-forward +/-.5m references fit A and MINCO; in each of4 old mirror worlds,one passes original Hard and the other fails:offlineOracle@2=4/4,both safe simultaneously0/4. This is old certificate data,not a freshK2 model experiment. G3-A crossing must be freshly replayed for both references in next qualification; this summary alone does not newly certify crossing. Forest15-witness banks show finite multidirectional feasibility but no new2-candidate forest policy was run.

## Distinct training mechanisms

J_minus,J_plus are unchanged original per-candidateL0:
- min(J_minus,J_plus):winner gradient,neglected branch and tie switching.
- mean:each gradient has1/2 scale; may punish legitimate blocked-side candidate or collapse modes.
- independent candidate gradients:dJ_k/dRaw_k;shared backbone gradients can still conflict.
Compare finite differences,norms,dot products,actual correct-side progress and nominal Hard+Task+crossing. Softmin introduces temperature/newmechanism and needs separate version/preregistration.

Oracle@2 is offline coverage only. GT/Hard cannot select deployableTop1; untrained scoring rows cannot select. No-GT deployable selection with calibration/UNKNOWN is an independent unsolved problem,not implemented.

## Next zero-update gate

Before anyK2 network:two physical references,actual headPVA,A inverse roundtrip,shape/candidate axes,independent MINCO PVA/C4,T2 and original limits. Replay memberwise Hard+crossing on frozen mirror scenes,then held-out forest seeds. Report both/one/zero pass,UNKNOWN,Oracle@2 and per-candidate quality. No onlineOracle.
If network work is later authorized,use independent candidate parameters,check normalization/factor2,shared-gradient conflicts,dead-row/optimizer protection,K1/K6 default/state_dict/numeric preservation and source/data identities. Separate approved RunSpec and zero-update gate required before training.

## Relation to native YOPO anchors

Native YOPO already uses physical direction lattice primitives5x3. Two lateral physical modes overlap that principle and alone are not novelty. Current hypothesized2-candidate A/3-piece/T2 differs from native spherical anchors15/2-piece/native timing and learned score/corridor. Isolate candidate coverage,initialization,aggregation and selection in any method comparison. No Anchor or newLoss implemented this round.
""")
 lines=['# True commands,exits and test matrix','',
 'Every directory below has PROCESS_START/RESULT,stdout.log,stderr.log and copied+hashed PREREGISTRATION. Child exit,not wrapper exit,is authoritative. CWD/environment/20ms process-tree resource samples are recorded.','',
 '|Run|Child exit|Elapsed s|Sampled tree RSS MiB|Conclusion|','|---|---:|---:|---:|---|']
 logs=sorted((O/'logs').glob('*/PROCESS_RESULT.json'))
 for p in logs:
  d=json.loads(p.read_text());c='PASS' if d['exit_code']==0 else 'FAIL preserved;new-directory corrected-harness rerun passed'
  lines.append(f"|{p.parent.name}|{d['exit_code']}|{d['elapsed_s']:.3f}|{d['sampled_process_tree_peak_rss_bytes']/1048576:.1f}|{c}|")
 lines+=['','## Recorded child argv and log locations','']
 for p in logs:
  d=json.loads(p.read_text());lines+=['### '+p.parent.name,'','~~~json',json.dumps(d['argv']),'~~~','',f"CWD:{d['cwd']};stdout:{p.parent.relative_to(O)}/stdout.log;stderr:{p.parent.relative_to(O)}/stderr.log;child exit={d['exit_code']}.",'']
 lines+=['## Assertions and NOT_RUN','',
 '|Check|Status|Evidence|','|---|---|---|',
 '|Forest bridge suite|PASS9 tests|forest_contract_tests_process_v3|',
 '|Old K1/K6/Hard/MINCO/UNKNOWN/crossing/provenance|PASS20 tests|hard_k1_k6_regression_process|',
 '|Six native generations/eleven native depth renders|PASS all exit0|dataset/*/generator_process,pose*/renderer_process|',
 '|Prior feasible denominator|PASS20 feasible,2 unknown,2 excluded|qualification/prepare|',
 '|Four-method zero-update evaluation|PASS88 records|FOREST_RESULTS,NPZ|',
 '|Source/operational SHA and exact occupancy|PASS|manifests,proofs,tests|',
 '|Original checkpoint/model unchanged|PASS|strict model_at loader,checkpoint_identity,no-grad assertion|',
 '|Native occupied output|PASS intentional rejection exit1|sentinel test|',
 '|Additional global clearance|PASS|forest_metrics_process_v2|',
 '|Native YOPO policy|NOT_RUN_NOT_COMPARABLE|fairness report|',
 '|Arbitrary rotations/full60m/high-speed|NOT_RUN_UNSUPPORTED_PROFILE|adapter report|',
 '|Backward,optimizer construction/step,leaf optimization|NOT_RUN_PROHIBITED|zero-update guards|',
 '|K2 model,selector,newLoss|NOT_RUN_DESIGN_ONLY|K2 design|',
 '|Old C1 registry|BLOCKED preexisting,NOT_RUN|historical gap kept separate|',
 '|Historical/source protection|PASS|HISTORICAL_PROTECTION|','',
 'First metrics failed project import;first native sentinel test exited127 missing shared library before guard. Only new harness paths fixed;original failed logs retained. No physics/data/history rewrite. Inner occupied-output rejection is expected PASS evidence;missing-library failure was not accepted as successful rejection.']
 put('TEST_MATRIX.md','\n'.join(lines))
 put('HISTORICAL_PROTECTION.md',"""
# Historical protection

PASS against entry:593 old source/config SHA,48480 old output SHA,1678 external source SHA,298065 external size/mtime records,80 partial-migration file SHA. No old files changed/missing,no unexpected new source at check. Branch/HEAD/index/unstaged tracked diffs preserved. Mainline status may add only whitelisted new untracked paths. Existing staged files not reset,unstaged or committed.

YOPO/reach sources/config/assets/checkpoints,formal128-step run,old manifests,.gitignore and partial migration untouched. Whitelist:ENTRY_PROTECTION.json. New source only adapter/nativehost/reference,thin experiments/tests;all assets/logs/reports under the unique output. FILES_AND_SHA records final additions.

External full-tree protection uses metadata plus1678 source hashes,not a content SHA for all298065 files. Mainline old outputs and explicitly used native assets are content-hashed. This is the actual protection scope. No optimizer update/ROS/controller/Git operation or migration occurred. Native CUDA depth rendering is offline sensing only.

New logs are genuine current child executions. Historical128-step process provenance remains UNVERIFIED;no retrospective logs fabricated.
""")
 statuses=dict(FOREST_ADAPTER='PASS',FOREST_QUALIFIED_SCENES=q['independently_feasible'],K1_FOREST_HARD_TASK=f"{summary['K1_step128']['hard_task']}/{summary['K1_step128']['n']}",YOPO_BASELINE_COMPARABLE='NO',FOREST_DATA_IDENTITY='PASS',K1_NETWORK_UPDATES=0,HISTORICAL_PROTECTION='PASS',READY_FOR_K2_PHYSICAL_FEASIBILITY='YES')
 with (O/'FINAL_STATUS.json').open('x') as f:json.dump(statuses,f,indent=2)
 put('README.md',"""
# G3-D forest audit package

Step12818/20 pre-certified-feasible Hard+Task;fixed/state-goal/Step0 each16/20. Six newTEST-only mapseeds,22 input snapshots,20 feasible,2 unknown,2 planned inputs excluded. No network update,leaf optimization,ROS/control,migration or Git mutation. Native YOPO policy NOT_RUN_NOT_COMPARABLE.

Read the six requested audit/design reports,TEST_MATRIX,HISTORICAL_PROTECTION,FINAL_STATUS and FILES_AND_SHA. CSVs support independent paired analysis;NPZ has actual coefficients/Raw21. Dataset manifests bind raw points,occupancy,camera,pose,depth/mask,state,source andqualification. All data stay in research_outputs;do not commit large assets to GitHub.

This local12m,same-asset,identity-pose,low-speed test does not establish visual routing,full60m/high-speed/closed-loop/generalization. Prior mirror failures remain. K2 is design only and next-study readiness is not training authorization. Existing output directories are exclusive:do not overwrite/rerun preparation into this result.
""")
 print(json.dumps(statuses,indent=2))
if __name__=='__main__':main()
