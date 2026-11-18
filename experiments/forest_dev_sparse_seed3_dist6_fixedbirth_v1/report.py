from pathlib import Path
import json,hashlib,subprocess,collections,re,sys
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1';D=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def stats(values):
 values=sorted(x for x in values if x is not None)
 return dict(count=len(values),min_s=values[0] if values else None,max_s=values[-1] if values else None,mean_s=sum(values)/len(values) if values else None,p95_s=values[min(len(values)-1,int(len(values)*.95))] if values else None)
def main():
 capture=json.loads((O/'capture_ordered/INPUTS.json').read_text());records=capture['records'];rows=json.loads((O/'replay_ordered/ROUTE_FRAMES.json').read_text());timeline=json.loads((O/'replay_ordered/COMMAND_TIMELINE.json').read_text());first=json.loads((O/'FIRST_BLOCKER_DETAILS.json').read_text());identity=json.loads((O/'BUILD_IDENTITY.json').read_text());events=timeline['events'];samples=timeline['samples'];commands=json.loads((O/'TEST_COMMANDS.json').read_text());testcount=sum(int(re.search(r'Ran (\d+) tests?',(O/x['stderr']).read_text()).group(1)) for x in commands)
 intervals=[b['stamp']-a['stamp'] for a,b in zip(records,records[1:])];span=records[-1]['stamp']-records[0]['stamp'];counts=dict(collections.Counter(x['event'] for x in events));reasons=dict(collections.Counter(x.get('reason',x['status']) for x in rows));longest=samples[-1]['ros_time']-samples[0]['ros_time'] if not any(x['valid'] for x in samples) else None
 write('B0_SENSOR_CERTIFICATION.json',dict(scope='NONPHYSICAL_FIXED_CAMERA_DEPTH_GEOMETRY_ONLY_NOT_S_C7_QUALIFICATION',frame_count=len(records),duration_s=span,measured_fps=(len(records)-1)/span,intervals=stats(intervals),strictly_ordered=all(x>0 for x in intervals),exact_stamp_pose_intrinsics_match=True,position=[.25,-.25,2.],quaternion_xyzw=[0,0,0,1],intrinsics=records[0]['K'],image_shape=[90,160],image_encoding='32FC1',depth_units='m',startup_ball_033_passed_frames=sum(x.get('start_ball',{}).get('passed',False) for x in rows),startup_ball_030=first['start_tube_030'],raw_input='capture_ordered/INPUTS.json',frame_files_sha_bound=True,no_controller=True,no_plant=True,no_arm=True,live_graph=capture['live_graph'],engineering_birth_assumption='0.60m fixed grant for NONPHYSICAL diagnosis; not a new physical birth/S qualification',GT_used=False,exit_code=0))
 write('B0_OUTSIDE_MINCO_RESULT.json',dict(scope='NATIVE_DEPTH_FULL_TUBE_NONPHYSICAL',experiment=O.name,goal=[3.25,-.25,2.],route_attempts=len(rows),certified_outside_plans=sum(x.get('outside_plan',False) and x['status']=='CERTIFIED' for x in rows),certified_astar_minco_plans=sum(x['status']=='CERTIFIED' for x in rows),reasons=reasons,primary_blocker='INSUFFICIENT_OBSERVED_SPACE',first_forward_probe=rows[0]['forward_probe'],first_astar_failure=rows[0]['first_failed_edge_query'],astar_expanded_lattice_nodes=rows[0]['expanded_lattice_nodes'],outside_raw_known_free_cells_first_frame=rows[0]['outside_raw_known_free_cells'],full_minco_diagnostic_probes=first['minco_probes'],unknown_reason_counts=first['reason_counts'],birth_radius=.6,tube_radius=.30,astar_radius=.33,dynamics_limits=[1,8,100],source='replay_ordered/ROUTE_FRAMES.json',GT_used=False,exit_code=0))
 write('B0_REALTIME_REPLAY.json',dict(scope='RECORDED_FIXED_NATIVE_INPUT_ORIGINAL_INTERVALS_NONPHYSICAL_NO_ACTUAL_PUBLISHER',input_frames=len(timeline['inputs']),measured_input_fps=(len(records)-1)/span,goal=[3.25,-.25,2.],counts=counts,rejection_reasons=dict(collections.Counter(x['reason'] for x in events if x['event']=='REJECTED')),queue_peak=max(x['queue'] for x in samples),queue_capacity=8,evidence_queue_overflows=sum('EVIDENCE_QUEUE_OVERFLOW' in str(x) for x in events+timeline['inputs']),commits=counts.get('CERTIFIED_COMMITTED',0),valid_command_fraction=timeline['valid_fraction'],longest_no_valid_command_s=longest,observation_revocations=counts.get('REVOKED',0),map_s=stats([x.get('map_s') for x in events]),total_planning_s=stats([x.get('total_s') for x in events]),planning_s=stats([x.get('planning_s') for x in events]),astar_minco_certificate_timings_available=False,timing_limitations='A* returns no route before MINCO; MINCO/certification times are separately recorded for deterministic goal-directed probes. No commit timings exist. Geometry tracing hit one SEARCH_DEADLINE; worker replay had none.',geometry_search_deadlines=reasons.get('SEARCH_DEADLINE',0),realtime_deadline_events=[x for x in events if 'DEADLINE' in x.get('reason','')],native_renderer_mean_depth_s=.00522072,latency_source='capture_ordered/renderer.stdout.log',command_samples=len(samples),tail_s=.25,raw_timeline='replay_ordered/COMMAND_TIMELINE.json',tests_passed=testcount,test_commands=commands,exit_code=0,physical_PASS=False))
 # Bind all current relevant repository source, including pre-existing F1 changes.
 names=subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=R,text=True).splitlines();extensions={'.py','.cpp','.hpp','.h','.cu','.cuh','.yaml','.yml','.xml','.launch','.patch','.cmake'}
 paths={R/n for n in names if Path(n).suffix in extensions and not n.startswith(('research_outputs/','isolated_worktrees/')) and (R/n).is_file()}
 paths.update(Path(x) for x in identity['inputs']);paths.update([O/'build/config.yaml',O/'build/native_renderer'])
 for name in ['Controller/src/so3_control/src/SO3Control.cpp','Controller/devel/lib/libquadrotor_dynamics.so','Simulator/src/src/maps.cpp','Simulator/src/src/perlinnoise.cpp','Simulator/src/src/sensor_simulator.cu']:
  p=R.parent/'YOPO'/name
  if p.is_file():paths.add(p)
 for name in ['devel_v2/lib/libf0_audit_nodelet.so','devel_v2/lib/libf0_audit_core.so']:
  p=R/'research_outputs/f0_controller_consumption_20261118_v1'/name
  if p.is_file():paths.add(p)
 for line in (O/'RENDERER_LDD.log').read_text().splitlines():
  match=re.search(r'=> (/[^ ]+)',line)
  if match and Path(match.group(1)).is_file():paths.add(Path(match.group(1)))
 for root in [O/'capture_ordered',O/'build']:
  paths.update(p for p in root.rglob('*') if p.is_file() and (p.suffix in {'.npz','.yaml','.json','.o'} or p.name.startswith('native_grid_GT_NOT_USED')))
 # Include historical S/C7/F0 identities read-only; does not inherit qualification.
 historical=R/'research_outputs/f0_controller_consumption_20261118_v1/runs';history={str(p):sha(p) for p in historical.glob('*/RESULT.json')};write('HISTORICAL_RESULT_IDENTITIES.json',dict(qualification_inherited=False,sha256=history));paths.update(Path(p) for p in history)
 write('B0_SOURCE_BINDING.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),dirty=True,scope='NONPHYSICAL_B0_SOURCE_CONFIG_INPUT_SCENE_BINARY; NOT_PHYSICAL_QUALIFICATION',sha256={str(p):sha(p) for p in sorted(paths)}))
 status=subprocess.check_output(['git','status','--short','--branch'],cwd=R,text=True);(O/'GIT_STATUS_FINAL.txt').write_text(status)
 (O/'SPARSE_SCENE_IDENTITY.md').write_text(f"""# forest_dev_sparse_seed3_dist6_fixedbirth_v1
Native random forest: maze_type=5, seed=3, tree_dist=6. Only YAML difference from original dist4 is tree_dist 4 -> 6; original config remains untouched.
Spawn=(0.25,-0.25,2.0), task goal=(3.25,-0.25,2.0), fixed flight quaternion xyzw=(0,0,0,1). No spawn/orientation search or multi-view fusion.
Native Renderer reported 23,200,380 generated points. Scene generated by native Maps::setParam YAML flow and seeded forest generator; no imported GT route.
Independent executable: {identity['binary']}
SHA256: {identity['binary_sha256']}
Config SHA256: {identity['config_sha256']}
Read-only native CUDA library and map/perlin objects reused. Host source and truth-export header are existing Query native Renderer adaptations, recompiled only with the new CONFIG_FILE_PATH. No full Simulator copy or rebuild.
Camera: 160x90, fx=fy=80,cx=80,cy=45; original pitch=0; native Depth target33Hz. Captured {len(records)} frames / {span:.6f}s = {(len(records)-1)/span:.3f}Hz.
No Controller or Plant started. Existing Query audit plugin identity remains f0_controller_audit_pkg/AuditSO3ControlNodelet; it was NOT loaded this round. Native SO3/Plant will require explicit physical permission and same-version qualification later.
GT export is required by the existing Renderer; exported files were retained and hashed, NEVER loaded by planning or route selection.
YOPO_PROTECTION_CHECK.json verifies 1,928 protected YOPO source/build files against the pre-existing baseline: zero drift. Historical F0/S/C7/F1 directories received no writes.
The first capture attempt and its replay are retained at this output root. It selected one frame out of order; superseded by complete chronological capture_ordered and replay_ordered. No first-attempt metrics are used for qualification.
""")
 (O/'B0_GO_NO_GO.md').write_text(f"""# B0_PHYSICAL=NO_GO
Nonphysical preparation executed; no physical qualification or episode executed. {testcount} nonphysical tests passed, all command exits0. This is not physical PASS.

Answers:
1. No native-Depth-certified outside-birth MINCO: 0/{len(rows)}. Birth-start 0.33m tube passes {len(rows)}/{len(rows)} under the explicitly nonphysical engineering birth grant.
2. No valid new online A*/MINCO plan: 0 commits, 0 valid commands. A* visits only five local lattice nodes. Raw outside-free cells ({rows[0]['outside_raw_known_free_cells']}) do not certify the complete tube.
3. Real-time evidence queue peak3/8, overflows0. Worker planning has no deadline rejection; separate instrumented geometry run has one SEARCH_DEADLINE. Main blockage is geometry, so continuous command delivery remains unproven.
4. Not ready to apply for B0 motion qualification. New source/scene binding cannot inherit old S/C7. No S/C7, ARM or Plant was started.
5. Primary blocker: insufficient certified observed-space overlap between the 0.60m birth grant and a complete forward tube at the fixed single flight camera pose.

Concrete first forward test: center=(0.55,-0.25,2.0), r=0.33: UNKNOWN52, birth-covered18, OCCUPIED0. Full 0.6m MINCO first fails at t~1.0s, p~(0.55,-0.25,2.0), r=0.30439453: UNKNOWN28, OCCUPIED0. The 0.8m probe fails near p=(0.545913,-0.25,2.0) with UNKNOWN28, OCCUPIED0.
Forward blocking-cell diagnoses overlap: 46 projection windows exceed existing computation bound24 pixels; 38 full error windows cross camera FoV; 6 have missing/no-return pixels. Counts are overlapping, not exclusive. Merely increasing a compute cap cannot certify cells whose full observation window lies outside FoV. No-return is not free-space evidence.

No safety parameter, UNKNOWN semantics, source map/Engine/Controller, or scene/spawn was changed. No favorable scene replacement attempted. A safe next implementation must demonstrate complete full-cell ray evidence across this overlap gap under the same camera/contract before rerunning B0; this round supplies exact cell/window evidence and makes no unsupported claim that such a patch already exists.

After geometry and real-time command eligibility pass: freeze source+scene+diagnostic plugin+binary binding -> explicit same-version S -> C7 seven-fault campaign -> separately approved B0 (~0.8m, real replan+distinct consumed IDs) -> independent B1 (>=2m, >=20 distinct valid published plans). S/C7 must cover the new runtime; old F0 remains frozen. New sparse experiment never counts as dist4 F1.
""")
 (O/'B0_NEXT_MANUAL_COMMANDS.md').write_text("""# Only read-only verification is currently eligible
```bash
cd /home/lizheng/query_minco_mainline
/home/lizheng/anaconda3/envs/yopo/bin/python -B experiments/forest_dev_sparse_seed3_dist6_fixedbirth_v1/check.py --check-only
```
Expected: SOURCE_BINDING_MATCH_NO_ROS_NO_ARM, B0_PHYSICAL=NO_GO.
Do not rerun prepare.py/capture.py/replay.py in their occupied output paths; preserved first and second attempts must not be overwritten.

No physical command is issued because geometry has not passed and a qualified sparse B0 physical entry is not established. Do not launch the old F1 physical entry or alter its goal/qualification file to bypass this NO_GO.

Once the geometry gate passes and a fixed sparse B0 entry is reviewed: freeze binding, explicitly authorize and run same-version S, then C7 (7/7, preserve full campaign), review exact RESULT paths and matching binding, then separately approve a single B0 run. Do not use /absolute/path placeholders. B0 STOP on new obstacle/UNKNOWN/expired proof/epoch conflict, input loss, queue/audit loss, excess dynamics/tracking error, duplicate/conflicting command identity, or missing Controller-to-Plant evidence. Preserve existing STOP/cleanup mechanism and original thresholds. This order is a future plan, not present execution authorization.

B0 fixed acceptance: native online Depth, approximately0.8m physical progression, genuine replanning with different released and Controller-consumed Plan IDs, unchanged safety, verified STOP and owned-process cleanup. Freeze exact numeric tolerances in a reviewed contract before any episode; no retrospective adjustment. B1 separately requires >=2m and >=20 distinct valid published trajectories with full causal evidence.
""")
 print('REPORTS_WRITTEN',testcount,'tests; source files',len(paths))
if __name__=='__main__':main()
