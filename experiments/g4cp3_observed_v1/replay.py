"""Offline source-time replay only. GT clearance is reported separately,never fed into the observed ledger."""
from pathlib import Path
import sys,json,time,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.observed_space_v1.map import ObservedMap,CertificationError,BudgetExceeded
from query_minco.integration.observed_space_v1.space import bootstrap,diagnose_legacy_ball
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1/replay_final2';O.mkdir()
folder=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101/pose0'
certificate=json.loads((folder/'POSE_CERTIFICATE.json').read_text());camera=json.loads((folder/'CAMERA_BINDING.json').read_text())
raw=folder/'render/depth_z.f32';depth=np.fromfile(raw,np.float32).reshape(90,160);pose=dict(p=certificate['position_world'],q=[0,0,0,1],frame='world')
plan=build(np.array(pose['p']),np.zeros(3),np.zeros(3),pose['q'],np.array(pose['p'])+[.4,0,0],10.)
radius=.3+plan.maxima[0]*float(plan.d.sum())/256
first=diagnose_legacy_ball(depth,pose,plan.sample(0)[0],radius);first['trajectory_time_s']=0.
m=ObservedMap(scene_id='frozen_forest3101')
m.add(depth,pose,10.,10.,10.);snap=m.snapshot(10.);boot=bootstrap(snap,pose['p'],10.)
assert first['reason']=='NEAR_FIELD_BLIND_ZONE' and boot['status']=='FAIL'
try:snap.certify(plan,10.);raise AssertionError('UNPROVED_BOOTSTRAP_ACCEPTED')
except CertificationError as e:curve=e.record
except BudgetExceeded as e:curve=dict(reason=str(e),status='FAIL_CLOSED',earliest_geometric_failure=boot['evidence'],trajectory_time=0.)
np.savez_compressed(O/'FOREST_STATIC_REPLAY.npz',depth=depth,pose=pose['p'],quaternion=pose['q'],states=snap.states,origin=snap.origin,coefficients=plan.c,durations=plan.d)
qualification=dict(module_sha256={str(p.resolve()):sha(p) for p in Path('query_minco/integration/observed_space_v1').glob('*.py')},status='STATIC_PASS',current_forest_admission='FAIL',first_legacy_failure=first,new_map_bootstrap=boot,new_map_curve_failure=curve,
 source_sha256={str(p):sha(p) for p in [raw,folder/'POSE_CERTIFICATE.json',folder/'CAMERA_BINDING.json']},
 camera=camera,offline_truth_clearance_m=certificate['clearance_lower_bound_m'],
 truth_scope='offline explanatory audit ONLY;not supplied to ObservedMap or online admission',sensor_information_limit='Two worlds with identical forward depth can differ by an obstacle behind the optical origin inside the tracking annulus. A single forward frame cannot certify this annulus.',
 independent_startup_contract='No GT/operator free bubble. Require actual bounded-age multiview observations covering all initial tube cells;otherwise NO-GO.',
 no_return_pixels=int((~(np.isfinite(depth)&(depth>=.04)&(depth<20))).sum()),new_native_runs=0)
write(O/'FOREST_BOOTSTRAP_REPLAY.json',qualification)
# Actual D open-scene images are a source-integrity/rotation replay,not forest qualification.
root=R/'research_outputs/g4cp2_manual_D_20261111_182327_10722/run/none'
events=[json.loads(x) for x in (root/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
poses={e['stamp']:e['payload'] for e in events if e.get('kind')=='odom' and e.get('accepted')}
files=[]
for p in (root/'reference_data').glob('depth_*.npz'):
 with np.load(p) as a:files.append((float(a['stamp']),p))
records=[];ledger=ObservedMap(scene_id='actual_D_open_sensor_replay');times=[]
for stamp,p in sorted(files):
 if stamp not in poses:
  records.append(dict(stamp=stamp,path=str(p),sha256=sha(p),status='EXCLUDED',reason='NO_EXACT_POSE_SOURCE_STAMP'));continue
 with np.load(p) as a:d=a['depth'].copy()
 begin=time.perf_counter()
 try:
  ledger.add(d,poses[stamp],stamp,stamp,stamp);s=ledger.snapshot(stamp);elapsed=time.perf_counter()-begin;times.append(elapsed)
  records.append(dict(stamp=stamp,path=str(p),sha256=sha(p),status='STATIC_PASS',quaternion=poses[stamp]['q'],snapshot=s.id,update_s=elapsed,
   states={str(i):int((s.states==i).sum()) for i in [0,1,2]},bootstrap=bootstrap(s,poses[stamp]['p'],stamp)['status']))
 except Exception as e:records.append(dict(stamp=stamp,path=str(p),sha256=sha(p),status='FAIL_CLOSED',reason=repr(e)))
result=dict(status='STATIC_PASS' if not any(r['status']=='FAIL_CLOSED' for r in records) else 'FAIL',records=records,
 frames=len(records),exact_pose_frames=len(times),excluded=sum(r['status']=='EXCLUDED' for r in records),
 update_s=dict(max=max(times or [0]),mean=float(np.mean(times)) if times else None),
 measured_nonidentity_quaternions=int(sum(np.linalg.norm(np.array(r.get('quaternion',[0,0,0,1]))-[0,0,0,1])>1e-6 for r in records)),
 scope='frozen real D native depth/physical odom replay,not a new physical run or forest-free proof')
write(O/'ACTUAL_D_SENSOR_REPLAY.json',result)
print('FOREST_FIRST_FAILURE',first);print('BOOTSTRAP',boot['status'])
print('ACTUAL_D_REPLAY',result['status'],len(records),len(times),result['update_s'])
assert result['status']=='STATIC_PASS'
