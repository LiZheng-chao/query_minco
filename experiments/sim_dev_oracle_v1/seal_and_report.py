# Identity seal and five focused reports. Never launches ROS/Plant/ARM.
from pathlib import Path
import sys,json,hashlib,subprocess,re,argparse,math
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';B=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1';D=Path(__file__).parent
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def semantics():
 lib=R.parent/'YOPO/Simulator/devel/lib/libraycast_cuda.so';binary=B/'build/native_renderer';probe=O/'renderer_identity';lines=[x for x in (probe/'renderer.stderr.log').read_text().splitlines() if 'renderDepthImage' in x];cuda=json.loads((O/'CUDA_FIXTURE_RESULT_v4.json').read_text());assert str(lib) in (probe/'RENDERER_PROCESS_MAPS.txt').read_text() and lines and str(lib) in lines[0] and cuda['loaded_library']==str(lib)
 meta=json.loads((O/'THIN_COUNTEREXAMPLE_GRID.json').read_text());data=np.memmap(O/'THIN_COUNTEREXAMPLE_GRID.u8',mode='r',dtype=np.uint8,shape=tuple(meta['shape']));ids=np.argwhere(data>0);assert len(ids)==1
 c=next(x for x in cuda['cases'] if 'thin_grazing' in x['name']);lo=np.array(meta['origin'])+ids[0]*meta['resolution'];hi=lo+meta['resolution'];origin=np.array(c['camera']);direction=np.array([1.,-.9875,0.]);interval=[0.,20.]
 for j in range(3):
  if direction[j]==0:assert lo[j]<=origin[j]<=hi[j];continue
  a,b=sorted([(lo[j]-origin[j])/direction[j],(hi[j]-origin[j])/direction[j]]);interval=[max(interval[0],a),min(interval[1],b)]
 assert interval[0]<interval[1] and c['depth']==20
 point=np.array(origin,dtype=np.float32);step=np.array([.05,-.9875*.05,0],dtype=np.float32);inside=0
 for _ in range(402):point+=step;inside+=int(((point>=lo)&(point<hi)).all())
 assert inside==0
 assert all(sha(Path(str(B/'capture_ordered/native_grid_GT_NOT_USED')+ext))==sha(Path(str(probe/'native_grid_GT_NOT_USED')+ext)) for ext in ['.json','.u8']),'SCENE_REGENERATION_DRIFT'
 d=dict(renderer=str(binary),renderer_sha256=sha(binary),loaded_cuda_library=str(lib),cuda_library_sha256=sha(lib),actual_symbol=cuda['resolved_symbol'],binding_lines=lines,cuda_source_sha256=sha(R.parent/'YOPO/Simulator/src/src/sensor_simulator.cu'),cases=cuda['cases'],thin_voxel=dict(bounds=[lo.tolist(),hi.tolist()],continuous_ray_parameter_interval=interval,samples_inside=inside,native_occupied_cells=1,returned_depth=20),scene_regeneration_sha_match=True,max_range_clear_ray_enabled=False)
 write('CUDA_RUNTIME_IDENTITY_AND_WITNESS.json',d)
def seal():
 semantics();old=json.loads((R/'research_outputs/f0_controller_consumption_20261118_v1/runtime/SOURCE_BINDING.json').read_text());paths={Path(p) for p in old};exts={'.py','.cpp','.hpp','.h','.cu','.cuh','.yaml','.yml','.launch','.patch','.xml'}
 paths.update(R/p for p in subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines() if Path(p).suffix in exts and not p.startswith(('research_outputs/','isolated_worktrees/')))
 for root in [D,R/'tests/sim_dev_oracle_v1',O/'generated_v2',O/'runtime_v2']:
  paths.update(p for p in root.rglob('*') if p.is_file() and p.name not in ['SOURCE_BINDING.json','SIM_DEV_RUNTIME_QUALIFICATION.json'] and '__pycache__' not in p.parts)
 for name in ['build/native_renderer','build/config.yaml','capture_ordered/INPUTS.json','capture_ordered/native_grid_GT_NOT_USED.json','capture_ordered/native_grid_GT_NOT_USED.u8']:paths.add(B/name)
 names=['build_v4/native_fixture','CUDA_FIXTURE_RESULT_v4.json','CUDA_RUNTIME_IDENTITY_AND_WITNESS.json','SIM_REPLAY.json','NONDRIVING_RUNTIME_REPLAY.json','SYNTHETIC_FOLLOW_RESULT.json','CHECK_COMMANDS.json','controller_probe_v3/RESULT.json','controller_probe_v3/INPUT_COMMANDS.json','controller_probe_v3/OUTPUT_SO3.json','controller_probe_v3/CONSUMPTION.jsonl','controller_probe_v3/COMMANDS.json']
 paths.update(O/n for n in names);paths.update(Path(x['path']) for x in json.loads((B/'capture_ordered/INPUTS.json').read_text())['records'])
 for row in json.loads((O/'CHECK_COMMANDS.json').read_text()):paths.update([Path(row['stdout']),Path(row['stderr'])])
 for name in ['renderer_identity/RENDERER_PROCESS_MAPS.txt','controller_probe_v3/PROCESS_MAPS.txt']:
  for line in (O/name).read_text().splitlines():
   p=Path(line.split()[-1])
   if p.is_absolute() and '.so' in p.name and p.is_file():paths.add(p)
 missing=[str(p) for p in paths if not p.is_file()];assert not missing,missing
 protected=json.loads((R/'research_outputs/f0_controller_consumption_20261118_v1/audit/PROTECTED_BEFORE.json').read_text());upstream={p:h for p,h in protected.items() if p.startswith('/home/lizheng/YOPO/')};assert all(sha(p)==h for p,h in upstream.items())
 previous=json.loads((B/'B0_SOURCE_BINDING.json').read_text())['sha256'];assert all(sha(p)==h for p,h in previous.items())
 mapping={str(p):sha(p) for p in sorted(paths)};head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();record=dict(seal_revision=3,mode='SIM_DEV_ORACLE',HEAD=head,verified_remote='68ec920b10087cdc8a10551dd368fe1326368fb5',branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),sha256=mapping,upstream_files_checked=len(upstream),previous_B0_files_checked=len(previous),protected_drift=0,strict_depth_modified=False,scene=dict(scene_kind='native_forest_seed3_maze5_tree_dist6_sim_dev_oracle',maze_type=5,seed=3,tree_dist=6,spawn=[.25,-.25,2.],goal=[3.25,-.25,2.]),camera=dict(width=160,height=90,fx=80,fy=80,cx=80,cy=45,max_depth=20),physical_steps=0)
 write('SIM_SOURCE_BINDING.json',record);(O/'runtime_v2/SOURCE_BINDING.json').write_text(json.dumps(mapping,indent=2));print('SEALED',len(mapping),head)
def report():
 binding=json.loads((O/'SIM_SOURCE_BINDING.json').read_text());assert all(sha(p)==h for p,h in binding['sha256'].items());qual=json.loads((O/'runtime_v2/SIM_DEV_RUNTIME_QUALIFICATION.json').read_text());assert qual['physical_qualified'] is False
 identity=json.loads((O/'CUDA_RUNTIME_IDENTITY_AND_WITNESS.json').read_text());native=json.loads((O/'SIM_REPLAY.json').read_text());rate=json.loads((O/'NONDRIVING_RUNTIME_REPLAY.json').read_text());synth=json.loads((O/'SYNTHETIC_FOLLOW_RESULT.json').read_text());ctrl=json.loads((O/'controller_probe_v3/RESULT.json').read_text());prefix=json.loads((O/'PREFIX_STUDY_RESULT.json').read_text());commands=json.loads((O/'CHECK_COMMANDS.json').read_text());assert all(c['exit_code']==0 for c in commands)
 tests=sum(int(m.group(1)) for c in commands if (m:=re.search(r'Ran (\d+) tests?',Path(c['stderr']).read_text())));assert tests>=90,tests
 boundsha=sha(O/'SIM_SOURCE_BINDING.json');common=dict(binding='SIM_SOURCE_BINDING.json',binding_sha256=boundsha,HEAD=binding['HEAD'],scene=binding['scene'],camera=binding['camera'],commands='CHECK_COMMANDS.json',tests=tests,physical_steps=0)
 replay=json.loads((O/'MAX_RANGE_REPLAY_RESULT.json').read_text());replay.update(provenance=common,cuda_runtime=identity,partial_information='Only discrete below-threshold queries before the ambiguous terminal region; not continuous uncertainty-tube FREE.',max_range_clear_ray_enabled=False);write('MAX_RANGE_REPLAY_RESULT.json',replay)
 rows='\n'.join('| '+x['name']+' | '+str(x['depth'])+' |' for x in identity['cases'])
 text=f'''# Depth20 semantics: actual binary, not source-only inference
Renderer SHA: {identity['renderer_sha256']}
Loaded library: {identity['loaded_cuda_library']}
CUDA SHA: {identity['cuda_library_sha256']}
Resolved function: {identity['actual_symbol']}
Native process maps and LD_DEBUG bindings prove this resolution; the compiled fixture calls the same existing library. YOPO was not rebuilt or modified. Source SHA is recorded separately; no stronger source-to-binary provenance is invented.

| Actual GPU case | Optical-X depth m |
|---|---:|
{rows}

20 is many-to-one: no sampled hit up to range; an occupied voxel CENTER exactly at20; or upper-Z map exit treated as empty. Near20 float rounding is also possible. There is no hit/miss validity bit. XY indices mirror; Z>=size is empty; Z<=0 is occupied. Point counts use strict >threshold, so a single point at threshold1 is not a native occupied voxel.

The camera-X step is0.5*resolution=0.05m; world step rotates(.05,-(u-cx)/fx*.05,-(v-cy)/fy*.05). Euclidean steps grow off-axis. The exported GPU grid has one truly occupied voxel; continuous ray parameter intersection={identity['thin_voxel']['continuous_ray_parameter_interval']}, but actual sample points inside=0 and GPU depth=20. This is an actual near thin-voxel miss, not a hypothetical source interpretation. Oblique hit, near hit, exact20 hit, upper/lowerZ and mirroredX boundary cases were exercised independently.

MAX_RANGE_CLEAR_RAY is NOT enabled. A future renderer-specific type would require the exact library/kernel/config/scene identity, matched pose/K/time, a terminal-range exclusion, and proof that sampled empty native voxels cover every required geometric/error cell without boundary aliasing. Discrete query information alone is insufficient; removing a terminal band does not fix grazed skipped voxels. No20->FREE or UNKNOWN->FREE rule was added.

All original69 frames were actually replayed. Strict0.6/0.8m MINCO certificates remain0/69 each. The separate SIM diagnostic generates{native['runtime_diagnostic_candidates']}/69 candidates and{native['runtime_diagnostic_prefix_accepts']}/69 GT prefix/brake acceptances. This is not original strict Depth-A* success.
Binding SHA={boundsha}. Commands: build_v4/COMMANDS.json, renderer_identity/ACQUISITION_COMMANDS.json, CHECK_COMMANDS.json. Detailed witness: CUDA_RUNTIME_IDENTITY_AND_WITNESS.json and THIN_COUNTEREXAMPLE_GRID.*. GT fixture data never enters the planner.
''';(O/'DEPTH_20M_SEMANTICS.md').write_text(text)
 text=f'''# SIM_DEV_ORACLE design and tests
Original STRICT_DEPTH map, Engine, F1/B0 validators, MINCO, Controller and Plant source remain unchanged. Only the isolated experiment/tests and generated_v2 adapters are new.

## Candidate generation
The original strict Depth-A* still certifies0 candidates on the69 frames. Live SIM mode explicitly uses SIM_LATEST_DEPTH_HIT_ASTAR_DIAGNOSTIC_NOT_STRICT_DEPTH: same frozen A* lattice and original2s/three-segment MINCO; finite measured-hit uncertainty envelopes veto occupied space. UNKNOWN may occur in a CANDIDATE, never a FREE map or execution grant. Fixed development local lookahead0.4m; fixed task goal(3.25,-0.25,2). Planner gets only Depth/Pose/IMU/Goal; no GT handle, waypoint, network input or GT-driven route search.
All sensor frames are saved before ACK to a bounded8-entry queue. Planning5Hz and guardian renewal50Hz are separate. GT is read only in a dedicated process. Exact AABB exclusion, query caching and original batch sampling remove duplicate work. Search-selected goals are rebuilt from newest measured PVA before admission. Full Plan SHA and run-unique trajectory_id remain distinct.

## Prefix, brake and simulation fault stop
A100ms lease checks nominal prefix plus350ms reaction/STOP delay (existing C7 stop bound), the original0.30m vehicle+tracking tube and speed-Lipschitz sample padding. A separate0.6s quartic braking family preserves initial PVA and reaches zero velocity/acceleration. Original polynomial extrema validate1/8/100; [0.15m,0.1m/s,0.5m/s^2] PVA uncertainty propagates into reaction and brake radii/dynamic extrema.
An independent GT ball covers arbitrary ADMISSIBLE speed<=1 directions until lease+STOP deadline: radius0.30+1*(0.10+0.35)=0.75m. It is checked separately. This fallback is explicitly native SIMULATOR INTEGRATION FREEZE, not real-UAV braking. Upon lease denial/expiry the existing owner aborts and stops Plant; no brake polynomial was physically flown. Bounds/latency must still be confirmed by NEW same-binding S/C7; excursions abort and are not qualified deployment behavior.
Map-version changes invalidate cached proofs; stale/invalid Depth, near measured occupancy, tracking/dynamic errors, nonce/Plan conflicts, overflow and guardian loss cannot grant execution. No GT result is written into observed-space FREE. Formal strict validators reject the new certificate type.

## Native reuse and actual results
Fixed maze5/seed3/dist6 and spawn(0.25,-0.25,2). Existing read-only CUDA Renderer, original SO3 computation via verified diagnostic Nodelet and existing native gated Plant/dynamics are reused. Original STOP, nonce/PID ownership, fresh ROS masters, publisher identity and cleanup remain. Only SIM stage selects the new Reference; S/C keep their original reference. Runtime_v2 is separately bound and never inherits old S/C7.
Native69: {native['runtime_diagnostic_candidates']} diagnostic candidates, {native['runtime_diagnostic_prefix_accepts']} oracle prefix/brake acceptances.
Native-rate non-driving bridge: {rate['distinct_eligible_plan_ids']} eligible Plan IDs, {rate['steady_valid_fraction']:.1%} steady eligibility, readiness after{rate['first_ready_after_s']:.3f}s, longest steady gap{rate['steady_longest_invalid_s']:.6f}s, queue peak{rate['queue_peak']}/8, no overflow. It does not fake PositionCommand publications.
Synthetic ANALYTIC Depth ideal follower: {synth['progress_m']:.6f}m kinematic progression, {synth['distinct_simulated_command_ids']} simulated Plan IDs, {synth['steady_valid_fraction']:.1%} steady eligibility. NOT native physical motion.
Real Controller-only probe: {ctrl['commands']} consumed PositionCommands, {ctrl['control_outputs']} real SO3 outputs,2 different Plan SHAs and a switch; plugin/.so verified; no Plant/ARM/motors publisher or final SO3 publication. Failed nonce/setup attempts remain preserved.
Final{tests} nonphysical tests PASS. Raw commands/exits/logs cover proof identity/expiry, missing controller/Plant chains, false switch, new obstacle, map change, stop reachability, exact optimizations and strict regressions. Intermediate failed tests/timing attempts are retained, not claimed PASS.

## Separate strict-prefix study
{json.dumps(prefix['checks'],indent=2)}
The prefix/reaction geometry passes but the robust brake family still enters strict UNKNOWN. No formal F1 rule was replaced. An offline modeled brake or simulator freeze is not actual real-aircraft braking qualification.
Nonphysical preflight scope: {qual['scope']}. Plant steps0; NEW S/C7 not run. Binding SHA={boundsha}.
''';(O/'SIM_DEV_ORACLE_DESIGN_AND_TESTS.md').write_text(text)
 text=f'''# SIM_DEV GO / NO-GO
GO to request researcher authorization for NEW same-binding stationary S and C7 using the prepared entry.
NO-GO NOW for ARM/development Plant movement until those actual prerequisites PASS and are reviewed.
STRICT_DEPTH F1/B0 stays NO-GO; no max-range clear-ray proof was issued.

What is proven: actual CUDA identity and thin miss;69/69 SIM diagnostic candidate+prefix checks;100% steady native-rate eligibility; separately labeled synthetic0.8m timing; actual two-Plan Controller consumption and switch. Plant feedback and physical brake/STOP in this NEW binding are NOT_RUN.

Remaining engineering/execution steps:
1. Human-approved S with frozen runtime_v2 source/scene/plugin/binary identity; review actual RESULT.
2. Human-approved same-binding C7, all7 faults, real Plant STOP, dynamics and owned cleanup. Old S/C7/F0 are not inherited.
3. Review those exact files, then separately approve one SIM_DEV_ORACLE0.8m attempt. The entry validates the new C7 campaign binding/case hashes and derives its nonphysical preflight from actual tests, never from a manually filled PASS.
4. Inspect real Depth, complete MINCO, PositionCommand, Controller->SO3->Supervisor->Plant lineage, >=2 actual consumed Plan IDs and switch, progress about0.8m (stop trigger0.80, fixed acceptance>=0.75), tracking<=0.15, dynamics, GT clearance, STOP and cleanup. Denial means STOP, no automatic retry.
5. Only after success, preregister a separate>=2m/>=20-plan run; it is not enabled automatically and cannot count as dist4 F1 or strict B0.

No new startup whole-path Depth-only condition is imposed on this SIM route. The monitor is conditional on the stated continuous dynamics/tracking/latency bounds; simulator freeze is not real-UAV braking and no deployment qualification is claimed. A live causal chain is assessed only from real logs, not fixture substitutions.
HEAD/verified remote=68ec920b10087cdc8a10551dd368fe1326368fb5. Previous two dirty Controller identity files and excluded report script preserved; new code uncommitted/unpushed. Protected YOPO files checked{binding['upstream_files_checked']}, prior B0 files checked{binding['previous_B0_files_checked']}, zero SHA drift.
Binding SHA={boundsha};{tests} final nonphysical tests PASS. No physical S/C7/Plant motion/ARM/training/Push ran this turn.
''';(O/'SIM_DEV_GO_NO_GO.md').write_text(text)
 text=r'''# Manual sequence - NOT executed here
Do not paste physical blocks as an unattended script. Each stage needs separate researcher authorization and previous actual RESULT review. This mode is simulation development only.

## Read-only now
```bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
ENTRY=experiments/sim_dev_oracle_v1/manual.py
BASE=/home/lizheng/query_minco_mainline/research_outputs/sim_dev_oracle_20261118_v1
"$PY" -B "$ENTRY" --mode SIM_DEV_ORACLE --check-only
```
Expected SOURCE_PASS_PREREQUISITE_PENDING_NO_ROS_NO_ARM: interface tests passed, NEW S/C7 pending. Exit0 is not physical qualification.

## Only after stationary-S authorization
```bash
S_RUN="$BASE/runs/S_$(date +%Y%m%d_%H%M%S)_$$"
test ! -e "$S_RUN" && "$PY" -B "$ENTRY" --mode SIM_DEV_ORACLE --stage S --allow-physics RESEARCHER_MANUAL_S_ONLY --output "$S_RUN"
```
Review the actual "$S_RUN/RESULT.json" before proceeding.

## Only after S PASS review and C7 authorization
```bash
test -f "$S_RUN/RESULT.json" || { echo "Missing actual new S RESULT"; return 1 2>/dev/null || exit 1; }
C7_RUN="$BASE/runs/C7_$(date +%Y%m%d_%H%M%S)_$$"
test ! -e "$C7_RUN" && "$PY" -B "$ENTRY" --mode SIM_DEV_ORACLE --stage C --allow-physics RESEARCHER_MANUAL_C_ONLY --prior "$S_RUN/RESULT.json" --output "$C7_RUN"
```
Review all7 cases and the exact campaign RESULT, with unchanged source/config/binaries.

## Only after new C7 review and separate0.8m motion approval
```bash
test -f "$C7_RUN/RESULT.json" || { echo "Missing actual new C7 RESULT"; return 1 2>/dev/null || exit 1; }
"$PY" -B "$ENTRY" --mode SIM_DEV_ORACLE --check-only --prior "$C7_RUN/RESULT.json"
DEV_RUN="$BASE/runs/SIM_DEV_ORACLE_$(date +%Y%m%d_%H%M%S)_$$"
test ! -e "$DEV_RUN" && "$PY" -B "$ENTRY" --mode SIM_DEV_ORACLE --stage SIM_DEV_ORACLE --allow-physics RESEARCHER_MANUAL_SIM_DEV_ORACLE_ONLY --prior "$C7_RUN/RESULT.json" --output "$DEV_RUN"
```
Only valid prerequisites yield PREFLIGHT_PASS_NO_ROS_NO_ARM SIM_DEV_ORACLE_ONLY; still not physical PASS. In a new terminal restore variables to reviewed REAL paths; never use /absolute/path placeholders or auto-select unreviewed runs.

STOP on GT prefix/brake/fault-stop denial, fresh measured occupancy, sensor loss, tracking/dynamics violation, map identity change, expired lease, guardian/process loss, identity conflict or evidence loss/overflow. Existing owner performs nonce/PID-owned STOP/cleanup. Preserve failures; do not retry automatically.
Save RESULT, RUN_IDENTITY, SCENE_IDENTITY, native_grid.*, sensors/ raw Depth+pose index, Supervisor events, Reference status/events, Controller CONSUMPTION and LOADED identity, Truth records, Plant process/STOP and cleanup logs. Never substitute synthetic Plant data. Subsequent2m/>=20-plan experiment requires a separate reviewed contract.
''';(O/'SIM_DEV_MANUAL_COMMANDS.md').write_text(text+'\nBinding SHA='+boundsha+'\n')
 write('DELIVERY_SUMMARY.json',dict(provenance=common,mode='SIM_DEV_ORACLE',nonphysical_interface='PASS',new_S='NOT_RUN',new_C7='NOT_RUN',Plant_motion='NOT_RUN',ARM='NOT_RUN',strict_depth='NO_GO',next='HUMAN_NEW_S_THEN_C7_THEN_SEPARATELY_APPROVED_SIM_0P8',reports=['DEPTH_20M_SEMANTICS.md','MAX_RANGE_REPLAY_RESULT.json','SIM_DEV_ORACLE_DESIGN_AND_TESTS.md','SIM_DEV_GO_NO_GO.md','SIM_DEV_MANUAL_COMMANDS.md']))
 print('DELIVERED',tests,'nonphysical tests; Plant0; new S/C7 pending')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.seal:seal()
 if a.report:report()
