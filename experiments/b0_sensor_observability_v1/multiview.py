"""Virtual multi-view observability: no execution/physical qualification."""
from pathlib import Path
import sys,time,json,math,itertools,collections
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from experiments.b0_sensor_observability_v1.reference import *
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
from query_minco.integration.depth_minco_t2_v1.space import Space

def astar_check(snap,now):
 begin=time.perf_counter()
 try:
  plan,proof=DepthPlanner(lambda:now,space=Space(snap,lambda:now)).propose(dict(p=SPAWN.tolist(),v=[0,0,0],a=[0,0,0],q=[0,0,0,1]),np.array([3.25,-.25,2.]),now)
  return dict(status='CERTIFIED',plan_id=plan.id,proof=proof,endpoint=plan.sample(float(plan.d.sum()))[0].tolist(),profile=plan._planning_profile,elapsed_s=time.perf_counter()-begin)
 except Exception as ex:return dict(status='REJECTED',reason=str(ex),details=getattr(ex,'record',None),elapsed_s=time.perf_counter()-begin)
def main():
 torch.set_num_threads(1);records=json.loads((O/'virtual_views/INPUTS.json').read_text())['records'];ledger=ObservedMap(CFG);rows=[];free_union={mode:None for mode in ['uncapped_square','conservative_rectangle']};occ_union=None;latest=None;orientation_rates=[]
 for index,item in enumerate(records):
  now=item['stamp'];dep=np.load(item['path'])['depth'];ledger.add(dep,item['pose'],now,now,now);begin=time.perf_counter()
  try:sensor=ledger.snapshot(now)
  except Exception as ex:
   rows.append(dict(frame=index,status='MAP_FAIL_CLOSED',reason=str(ex)));continue
  native_map_s=time.perf_counter()-begin
  ids=np.indices(CFG.shape).transpose(1,2,3,0);centers=(sensor.origin+(ids+.5)*.1).reshape(-1,3);g=project(centers,item['pose']);masks={mode:evaluate(dep,g,mode) for mode in free_union}
  # Original map's all-view occupancy always dominates. Rectangle free mask is
  # the union of individually safe witnesses, not inferred unobserved volume.
  occ_union=sensor.states==2
  for mode,(mask,reasons) in masks.items():
   visible=np.array([not any(reason in ['NEAR_PLANE','FULL_ERROR_WINDOW_OUTSIDE_FOV'] for reason in why) for why in reasons])
   if free_union[mode] is None:free_union[mode]=mask.copy()
   else:
    free_union[mode][visible]=mask[visible]
    free_union[mode]|=mask
  snaps=dict(current=authorized_snapshot(sensor,BirthGrant(tuple(SPAWN),.6,now+.2,'VIRTUAL_ONLY'),now,sensor.epoch,lambda:sensor.epoch))
  for mode,mask in free_union.items():snaps[mode]=make_snapshot(sensor,mask,now,'VIRTUAL_ONLY_NOT_PHYSICAL')
  probes=[]
  for distance in [.6,.8]:
   plan=build(SPAWN,np.zeros(3),np.zeros(3),[0,0,0,1],SPAWN+[distance,0,0],now)
   probes.append(dict(distance=distance,certification={mode:certify(snap,plan,now) for mode,snap in snaps.items()}))
  rows.append(dict(frame=index,stamp=now,pose=item['pose'],yaw_deg=item['yaw_deg'],pitch_deg=item['pitch_deg'],frames_in_history=len(sensor.frames),native_map_s=native_map_s,offline_reference_total_s=time.perf_counter()-begin,free_cells={mode:int((snap.states==1).sum()) for mode,snap in snaps.items()},occupied_cells=int(occ_union.sum()),probes=probes,valid_until=min(s.valid_until for s in snaps.values()),scope='VIRTUAL_TELEPORT_POSES_WITH_REAL_STAMPS_NOT_EXECUTED'))
  latest=(sensor,snaps)
  # Record exact current tube result for every view, independently of MINCO.
  rows[-1]['forward_query']={mode:snap.ball([.55,-.25,2.],.33,now) for mode,snap in snaps.items()}
  if index:
   from scipy.spatial.transform import Rotation
   q0=Rotation.from_quat(records[index-1]['pose']['q']);q1=Rotation.from_quat(item['pose']['q']);angle=float((q0.inv()*q1).magnitude());dt=item['stamp']-records[index-1]['stamp'];orientation_rates.append(dict(angle_deg=math.degrees(angle),dt_s=dt,required_average_rate_rad_s=angle/dt))
 # Source-time history test: no laundering of old source stamps or fake refresh.
 expiry_tests=[]
 sensor,snaps=latest;now=records[-1]['stamp'];source_expired=False
 try:ledger.snapshot(now+.20001)
 except Exception as ex:source_expired=True;expiry_tests.append(dict(case='NO_NEW_DEPTH_AFTER_LIVE_AGE',rejected=True,reason=str(ex)))
 # Check a legal fresh repeated final pose after a 2+second source gap; old
 # views must be discarded and must not continue supplying a free corridor.
 final=records[-1];future=now+2.01;ledger.add(np.load(final['path'])['depth'],final['pose'],future,future,future);future_snap=ledger.snapshot(future)
 expiry_tests.append(dict(case='FRESH_LAST_VIEW_AFTER_HISTORY_GAP',history_frames=len(future_snap.frames),old_views_retained=len(future_snap.frames)>1,epoch=future_snap.epoch))
 # Check yaw-only nominal visibility necessary condition over every corner.
 exact=json.loads((O/'B0_EXACT_RAY_REFERENCE.json').read_text());witnesses=[]
 for voxel in exact['voxel_diagnosis']['voxels']:
  lo,hi=map(np.array,voxel['world_bounds']);corners=np.array(list(itertools.product(*zip(lo,hi))));d=corners-SPAWN;rho=np.linalg.norm(d[:,:2],axis=1);vertical=45-80*d[:,2]/rho
  impossible=(vertical<0)|(vertical>89)
  if impossible.any():witnesses.append(dict(world_center=voxel['world_center'],corner_world=corners[np.flatnonzero(impossible)[0]].tolist(),best_over_all_yaws_pixel_v=float(vertical[np.flatnonzero(impossible)[0]])))
 result=dict(scope='VIRTUAL_MULTIVIEW_GEOMETRY_ONLY_NOT_ACTUAL_BODY_TURNING',views=len(records),source_span_s=records[-1]['stamp']-records[0]['stamp'],live_age=.2,history_ttl=2.,maximum_history_frames=32,all_views_within_history_ttl=(records[-1]['stamp']-records[0]['stamp'])<2.,no_baseline_old_recording_fused=True,rows=rows,final_astar={mode:astar_check(snap,now) for mode,snap in snaps.items()},orientation_rates=orientation_rates,expiry_tests=expiry_tests,yaw_only_nominal_impossibility_witnesses=witnesses,actual_yaw_command_current=0.,actual_pitch_roll_direct_control=False,actual_body_turning_proven=False,reason='Native Renderer was driven by virtual odometry; discontinuous orientation jumps have no Controller/SO3/Plant causal chain. Existing position-command path fixes yaw0; pitch/roll arise from translational force, not an independent stationary camera turn.',GT_used=False,exit_code=0)
 write('B0_SENSOR_OBSERVABILITY_RESULT.json',result);print('MULTIVIEW',[(x['distance'],{m:y['status'] for m,y in x['certification'].items()}) for x in rows[-1]['probes']]);print('ASTAR',result['final_astar']);print('YAW_ONLY_WITNESSES',len(witnesses));print('RATES',orientation_rates)
if __name__=='__main__':main()
