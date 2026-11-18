from pathlib import Path
import sys,json,subprocess,time,hashlib,shutil
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/b0_sensor_observability_20261118_v1';D=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 logs=O/'refinement_final';logs.mkdir();commands=json.loads((O/'EXECUTION_COMMANDS.json').read_text())
 for name in ['subcell_reference.py','test_subcell.py']:
  argv=[sys.executable,'-B',str(D/name)];begin=time.time()
  with (logs/(name+'.stdout.log')).open('x') as so,(logs/(name+'.stderr.log')).open('x') as se:proc=subprocess.run(argv,cwd=R,stdout=so,stderr=se,timeout=120)
  commands.append(dict(argv=argv,cwd=str(R),exit_code=proc.returncode,elapsed_s=time.time()-begin,stdout=str(logs/(name+'.stdout.log')),stderr=str(logs/(name+'.stderr.log'))));assert proc.returncode==0,name
 write('EXECUTION_COMMANDS.json',commands)
 preserve=O/'pre_refinement_reports';preserve.mkdir()
 for name in ['B0_UNKNOWN_VOXEL_ROOT_CAUSE.md','B0_EXACT_RAY_REFERENCE.json','B0_SENSOR_OBSERVABILITY_RESULT.json','B0_GEOMETRY_BEFORE_AFTER.json','B0_NEXT_GO_NO_GO.md','BOUND_IDENTITY.json']:shutil.copyfile(O/name,preserve/name)
 sub=json.loads((O/'SUBCELL_EXACT_SET_COVER_REFERENCE.json').read_text());bound=json.loads((O/'BOUND_IDENTITY.json').read_text());bound['sha256'].update({str(f):sha(f) for f in D.glob('*.py')});bound['sha256'][str(O/'EXECUTION_COMMANDS.json')]=sha(O/'EXECUTION_COMMANDS.json');write('BOUND_IDENTITY.json',bound);identity=sha(O/'BOUND_IDENTITY.json')
 for name in ['B0_EXACT_RAY_REFERENCE.json','B0_SENSOR_OBSERVABILITY_RESULT.json','B0_GEOMETRY_BEFORE_AFTER.json']:
  data=json.loads((O/name).read_text());data['provenance'].update(binding_sha256=identity,execution_commands_sha256=sha(O/'EXECUTION_COMMANDS.json'),tests_passed=43)
  data['refined_tube_intersection_reference']='SUBCELL_EXACT_SET_COVER_REFERENCE.json';data['refined_conclusion']='Full-cell FoV failure does not prove actual tube invisibility. Conservative subcell cover and sharper birth-intersection proof still fail. Actual outside-birth inside-tube witnesses have nominal depth20. No valid-return evidence under unchanged contract; no production patch justified.'
  if name=='B0_GEOMETRY_BEFORE_AFTER.json':
   data.update(tests_passed=43,all_logged_commands_exit_zero=False,retained_failed_diagnostic_attempts=1,final_test_suites_all_pass=True,failure_resolution='Initial witness test rejected a birth-boundary AABB false blocker. Exact collinear sphere-box maximum removed that artifact; actual uncovered witnesses and all six final subcell tests pass.',subcell_reference=sub)
  if name=='B0_EXACT_RAY_REFERENCE.json':data['subcell_reference']=sub
  write(name,data)
 root='''# B0 UNKNOWN voxel root cause

B0_PHYSICAL=NO_GO. No production code or safety constants changed.
The observed blocker is insufficient valid Depth evidence under the existing contract. There is additional full-voxel proof conservatism, but removing computation caps and refining that conservatism did not certify a complete trajectory. Do not equate this conclusion with a theorem that the physical lens cannot ever support B0.

## Native single-view reference
All 69 saved frames were reused and hash-checked; there is only ONE distinct Depth content. The baseline orientation was not collected again. All complete voxel coordinates, world AABBs, optical coordinates, projection intervals, windows and per-mode outcomes are in B0_EXACT_RAY_REFERENCE.json:voxel_diagnosis.voxels.
For query center (0.55,-0.25,2.0), radius0.33, all52 unauthorized UNKNOWN voxels remain unresolved after removing the radius24 cap. The46 cap-exceeding windows overlap other causes; none of these52 is recovered by merely removing the cap.
Uncapped causes:38 full uncertainty windows outside FoV,14 windows with invalid/no usable return,0 near-plane,0 occlusion/depth-margin-only cases. Query OCCUPIED=0. Invalid saved pixels are at/above20m, not zeros or NaNs. They cannot become FREE under the unchanged finite-depth<20 contract.
Ten full voxels have nominal corners outside FoV even before error margins. Example center(0.75,-0.45,1.75), bounds x[0.70,0.80],y[-0.50,-0.40],z[1.70,1.80], includes nominal v=98.33 beyond the90-row image. This is a FULL-VOXEL witness, not proof that the actual uncovered portion of the trajectory is invisible.

## Conservative ray calculation
Uncapped square uses the exact algebra, rounding, full-cell uncertainty sphere and min-depth test of Frame.carve, omitting only the compute admission radius24. Real native and synthetic per-cell equivalence tests pass.
For optical point(x,y,z), original h=0.1 model uses r=h*sqrt(3)/2+epsilon_p+norm(optical)*sin(epsilon_theta). Projection extrema of (x+-r,y+-r,z+-r) enclose that sphere. Keep z-r>=0.04, guard1.5+pixel_guard, all finite0.04<=Depth<20, and minDepth>z+r+epsilon_d. The uncapped square never clips an out-of-FoV window to invent free evidence.
The additional rectangular reference separately outward-rounds conservative u/v intervals of the same box. It is a tighter conservative box envelope, not a new sensor assumption. Both references reject the full0.6m and0.8m MINCO in all69 frames. Their maxima are about[0.5625,0.8660,4.5] and[0.7500,1.1547,6.0], within1/8/100. Frozen Depth-A* also finds no route on each single-view reference.
RAY_SUPPORT_DETAILS.json records every blocking cell/window, first invalid pixel/value, valid/range-limited pixel counts and optical rays. No GT was used.

## Refined birth-plus-Depth set cover
A full voxel can straddle already-authorized birth space. To avoid blaming its irrelevant FoV-outside corner, an independent offline reference recursively subdivides unresolved cells to1.5625mm. Each query intersection must be outside the query, rigorously birth-authorized, or covered by a complete conservative Depth projection. Any original OCCUPIED cell rejects before refinement; no map UNKNOWN is relabeled FREE.
Pose displacement for each subcell uses epsilon_p+2*sin(epsilon_theta/2)*max_corner_norm, bounding all rotations by the exact chord bound with unchanged angle constant. Project the transformed subcell plus that error box, outward-round with unchanged guard; require ALL valid ray returns and minDepth greater than maximum opticalZ+epsilon_d. No-return remains UNKNOWN.
For x in query ball centered p, birth center b: |x-b|^2 <= r^2+2(p-b).(x-p)+|p-b|^2. A box support bound is conservative. For this experiment's positive-X collinear centers, x>=0,d>=0, use the exact sphere-box reduction: perpendicular squared distance s lies in[smin,smax]; largest feasible x*=min(xhi,d+sqrt(r^2-smin)); maximum birth distance squared=x*^2+min(smax,r^2-(x*-d)^2). Both objective branches are monotone, so the maximum feasible x suffices. This is an OFFLINE fixed-geometry proof, not a general production patch.
An early witness test correctly exposed a remaining AABB boundary false blocker. That failed run is retained. The tighter sphere-box calculation removes it; all six final refinement tests pass.

Decisive ACTUAL witnesses:
-0.6m MINCO: point(0.845625,-0.3025,1.95), birth distance0.600021m, query distance0.304385m<radius0.304395m. It is genuinely outside birth and inside the swept query. Nominal pixel(87,52) returns20.0.
-0.8m MINCO: point approximately(0.848203,-0.289297,1.975), birth distance0.600014m, query distance0.305857m<radius0.305859m. Nominal pixel(85,48) also returns20.0.
These are not voxel corners outside the query and not FoV-exterior pixels. Under the current rule these required rays have no usable free-space evidence. Unlimited calculation cannot supply missing measurements; finite subdivision itself is not a global infeasibility theorem.
For comparison, the nominal query-ball portion outside the birth sphere is within a forward cone of about18.65deg, inside the camera's approximately29.36deg vertical half-FoV. Therefore widening FoV alone is not demonstrated to solve the decisive missing-return witnesses.

## Preregistered virtual views
Eight fixed yaw/pitch pairs: (0,-30),(45,-30),(45,0),(45,30),(0,30),(-45,30),(-45,0),(-45,-30). Fixed spawn and K; no scene/start/view search. PREREGISTERED_VIEWS.json predates capture. Native source span is0.223029s, within2s history; old baseline frames were NOT fused with this new acquisition.
Current query UNKNOWN decreases52->48; uncapped references end at44. Current full MINCO still UNKNOWN28; uncapped references UNKNOWN26. No full trajectory passes. Fusion applies latest-visible evidence revocation, occupancy dominance and original TTL; a fresh final view after>2s gap retains only1 frame. No Depth after0.2s yields NO_FRESH_DEPTH.
These are virtual pose teleports, requiring13.1-28.3rad/s average attitude changes. No Controller/SO3/Plant chain established them. Current Supervisor fixes yaw0; pitch/roll follow translational force, not independent stationary camera control. Real feasible sensing motion remains UNKNOWN. Full-voxel yaw witnesses do NOT prove that every possible moving viewpoint is impossible.

## Tests, protection and identity
Final43 nonphysical tests pass:37 existing/reference tests plus6 refinement tests. One intermediate failed diagnostic witness test is preserved and resolved; do not count its initial attempt as PASS. Raw commands/exit codes/logs: EXECUTION_COMMANDS.json, final_runs/, refinement_runs/, refinement_final/.
YOPO1928 protected source/build hashes unchanged. Previous B0 binding1037 files unchanged. No Controller, Plant, ARM, physical episode, training, or Push. Source HEAD5a17778f625ae03506ed53d0e5bf944a4ead5815; prior dirty changes preserved.
All results bind source, scene, original camera and69+8 native input identities via BOUND_IDENTITY.json. Final binding SHA: BINDING_SHA
'''
 (O/'B0_UNKNOWN_VOXEL_ROOT_CAUSE.md').write_text(root.replace('BINDING_SHA',identity))
 next_report='''# B0_PHYSICAL=NO_GO

1. Root cause: insufficient valid sensor evidence under the existing safety contract, with extra full-voxel conservatism. No implementation defect sufficient to release B0 was found. Neither a compute-cap change nor a claim of intrinsic lens impossibility is supported.
2. Native single-view0.6m/0.8m MINCO: no certified result. Removing radius24, tightening rectangular projections and refining birth/Depth intersections did not pass. Actual inside-tube/outside-birth points have nominalDepth20.0; all69 saved Depth frames have the same content. No-return cannot become FREE.
3. Real feasible orientation/viewpoint changes remain UNKNOWN. Eight virtual views did not certify complete tubes. Current yaw0 and force-derived pitch/roll cannot be replaced by virtual attitude teleports. The full-voxel yaw FoV argument does not rule out all possible translating sensing motions.
4. Most useful minimal next experiment: preregister ONE bounded sensing motion inside the unchanged birth sphere, keeping the original camera and birth location. FIRST nonphysically verify feasible PVA/attitude dynamics, full safety tube, actual predicted viewpoint and2s history/0.2s freshness support for the missing rays. Do not search for favorable spawns or treat virtual poses as physical execution. Such a motion has NOT been solved or validated this round; do not fly it now.
An independent wider-FoV sensor experiment is an optional new contract. It may improve boundary coverage but cannot automatically fix the demonstrated central20m pixels. Any such experiment must bind new Renderer K and map projection/uncertainty f_min together, and revalidate all usable-return and tube proofs. It cannot count as original-camera B0 qualification. No change to20m/UNKNOWN semantics is authorized.
5. Production changes this round: NONE. Frame.carve radius24 is not the decisive blocker; isolated refinement of Snapshot.ball/certify and birth_covers_intersection conservatism still fails, so integrating a more complicated production certificate is not justified. Engine and Controller stay unchanged.

Required order before motion: demonstrate complete safe plans and real-time continuity under physically attainable observations or a separately approved sensor contract -> freeze binding -> explicitly authorize same-version S -> C7 -> one manual B0. Old S/C7/F0 do not transfer automatically. No physical command is issued.

Final43 nonphysical tests PASS, not physical qualification. One resolved intermediate failed diagnostic attempt is retained. All five deliverables share BOUND_IDENTITY.json SHA=BINDING_SHA. Native acquisition commands: virtual_views/ACQUISITION_COMMANDS.json; calculation/test commands: EXECUTION_COMMANDS.json. See SUBCELL_EXACT_SET_COVER_REFERENCE.json for actual missing-ray witnesses.
'''
 (O/'B0_NEXT_GO_NO_GO.md').write_text(next_report.replace('BINDING_SHA',identity))
 drift=[f for f,h in bound['sha256'].items() if not Path(f).is_file() or sha(f)!=h];assert not drift
 required=['B0_UNKNOWN_VOXEL_ROOT_CAUSE.md','B0_EXACT_RAY_REFERENCE.json','B0_SENSOR_OBSERVABILITY_RESULT.json','B0_GEOMETRY_BEFORE_AFTER.json','B0_NEXT_GO_NO_GO.md']
 write('FINAL_VALIDATION.json',dict(reports={n:sha(O/n) for n in required},binding_sha256=identity,source_drift=drift,nonphysical_tests_passed=43,retained_failed_diagnostic_test_runs=1,physical='NO_GO',production_modified=False,exit_code=0))
 print('FINAL43 tests PASS; binding drift0; physical NO_GO')
if __name__=='__main__':main()
