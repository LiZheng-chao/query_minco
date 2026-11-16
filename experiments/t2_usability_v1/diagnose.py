"""All12 frozen samples, single/multi-frame, short/outside goals. No ROS."""
import sys,time,json,csv,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared
from query_minco.integration.observed_space_v1.map import Config
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot,Space
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
from query_minco.integration.yopo_execution_v4.trajectory import build
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O=Path('research_outputs/t2_usability_20261116_v1')
 prior=json.loads(Path('research_outputs/t2_acceptance_20261116_v1/replay/RESULT.json').read_text())
 first=Path(prior['rows'][0]['path']);run=first.parents[1]
 events=[json.loads(x) for x in (run/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
 odom={x['stamp']:x for x in events if x.get('kind')=='odom' and x.get('accepted')}
 imu={x['stamp']:x for x in events if x.get('kind')=='imu' and x.get('accepted')}
 dep={x['stamp']:x for x in events if x.get('kind')=='depth' and x.get('accepted')}
 truth=[json.loads(x) for x in (run/'truth_data/EVENTS.jsonl').read_text().splitlines()]
 truth=[x for x in truth if x.get('kind')=='truth' and x['status']=='PASS']
 bc=(run/'BIRTH_CERTIFICATE.json').read_bytes();args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
 selected=json.loads((O/'BIRTH_SELECTION.json').read_text())
 files=sorted((run/'reference_data').glob('*.npz'))
 rows=[];detail=[];profiles=[];multi=ObservedMap(Config(shape=(32,24,16)));processed=-float('inf')
 for idx,oldrow in enumerate(prior['rows']):
  path=Path(oldrow['path']);ts=oldrow['stamp'];now=dep[ts].get('ros_now',ts);clock=lambda:now
  state=dict(odom[ts]['payload'],a=imu[ts]['payload']['a_world'])
  with np.load(path) as data:d=data['depth'].copy()
  tr=[x for x in truth if 0<=dep[ts]['monotonic']-x['monotonic']<=.2][-1]
  grant=BirthGrant.from_records(bc,tr,args,selected,now,dep[ts]['monotonic'])
  accum_start=time.perf_counter()
  for f in files:
   with np.load(f) as z:stamp=float(z['stamp']);image=z['depth'].copy()
   if processed<stamp<=ts and stamp in odom and stamp in dep:
    multi.add(image,odom[stamp]['payload'],stamp,stamp,dep[stamp].get('ros_now',stamp));processed=stamp
  accumulate_s=time.perf_counter()-accum_start
  valid=int((np.isfinite(d)&(d>=.04)&(d<20)).sum())
  for mode in ['single','accumulated']:
   ledger=ObservedMap(Config(shape=(32,24,16))) if mode=='single' else multi
   start=time.perf_counter()
   if mode=='single':ledger.add(d,state,ts,ts,now)
   try:
    sensor=ledger.snapshot(now);snap=authorized_snapshot(sensor,grant,now,idx,lambda:idx)
   except Exception as error:
    detail.append(dict(frame=idx,mode=mode,error=str(error)));continue
   map_s=time.perf_counter()-start
   counts={name:int((snap.states==v).sum()) for name,v in [('UNKNOWN',0),('KNOWN_FREE',1),('OCCUPIED',2)]}
   distribution=[]
   xyz=snap.origin+(np.indices(snap.states.shape).transpose(1,2,3,0)+.5)*.1
   for axis in [0,1,2]:
    distribution.append({name:[int(((snap.states==v)&((xyz[...,axis]-state['p'][axis])<0)).sum()),int(((snap.states==v)&((xyz[...,axis]-state['p'][axis])>=0)).sum())] for name,v in [('UNKNOWN',0),('KNOWN_FREE',1),('OCCUPIED',2)]})
   # Exhaustive frozen A* graph connectivity, no runtime search deadline.
   reachable={(0,0)};queue=[(0,0)];edges=[]
   while queue and len(reachable)<2000:
    u=queue.pop(0);a=np.asarray(state['p'])+np.array([u[0]*.2,u[1]*.2,0])
    for dx in [-1,0,1]:
     for dy in [-1,0,1]:
      if not(dx or dy):continue
      v=(u[0]+dx,u[1]+dy)
      if v in reachable or abs(v[0])>15 or abs(v[1])>10:continue
      b=np.asarray(state['p'])+np.array([v[0]*.2,v[1]*.2,0])
      checks=[snap.ball(x,.33,now) for x in np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.05)+1))]
      bad=next((x for x in checks if not x['passed']),None)
      if bad:edges.append(dict(from_node=u,to_node=v,reason=bad.get('reason'),evidence=bad))
      else:reachable.add(v);queue.append(v)
   diag=dict(frame=idx,mode=mode,source_stamp=ts,matched_odom_stamp=ts,
     camera_pose=state,valid_pixels=valid,total_pixels=d.size,FOV_degrees=[90.,float(2*np.degrees(np.arctan(45/80)))],
     source_sha=sha(path),sensor_counts={name:int((sensor.states==v).sum()) for name,v in [('UNKNOWN',0),('KNOWN_FREE',1),('OCCUPIED',2)]},
     authorized_counts=counts,distribution=distribution,birth=dict(center=grant.center,radius=grant.radius,expires=grant.expires,whole_cells=snap.birth_cells),
     connected_centers=sorted(reachable),failed_edges=edges)
   detail.append(diag)
   for label,offset in [('birth_short',.28),('outside_birth',1.)]:
    goal=np.asarray(state['p'])+[offset,0,0];space=Space(snap,clock);fail_edges=[]
    space.edge_rejected=lambda a,b,x,r:fail_edges.append(dict(start=a.tolist(),end=b.tolist(),first_failed_point=x.tolist(),radius=r,reason=snap.ball(x,r,now).get('reason')))
    t=time.perf_counter();status='FAIL';reason=None;plan=None;proof=None
    try:plan,proof=DepthPlanner(clock,space=space).propose(state,goal,ts);status='PASS'
    except Exception as error:reason=str(error)
    planning_s=time.perf_counter()-t
    # Independent straight physical reference diagnostic, not an execution choice.
    t=time.perf_counter();dyn='NOT_RUN';collision='NOT_RUN';peaks=None;cert_s=None
    try:
     direct=build(np.array(state['p']),np.array(state['v']),np.array(state['a']),state['q'],goal,ts)
     dyn='PASS';peaks=direct.maxima;minco_s=time.perf_counter()-t;t=time.perf_counter()
     try:snap.certify(Prepared(direct),now);collision='PASS'
     except Exception as error:collision=str(error)
     cert_s=time.perf_counter()-t
    except Exception as error:dyn=str(error);minco_s=time.perf_counter()-t
    row=dict(frame=idx,mode=mode,task=label,source_stamp=ts,matched_odom_stamp=ts,input_sha=sha(path),valid_pixels=valid,
     known_free=counts['KNOWN_FREE'],occupied=counts['OCCUPIED'],unknown=counts['UNKNOWN'],
     birth_radius=.60,reachable_nodes=len(reachable),max_connected_x=max(v[0]*.2 for v in reachable),
     goal_outside_ROI=bool(((goal<snap.origin)|(goal>=snap.origin+np.asarray(snap.cfg.shape)*.1)).any()),
     planner=status,reason=reason,discrete_to_minco='PASS' if plan else 'NOT_RUN',direct_dynamics=dyn,direct_collision=collision,
     map_s=map_s,planning_s=planning_s,minco_s=minco_s,certification_s=cert_s,accumulation_s=accumulate_s)
    rows.append(row);profiles.append(dict(**row,peaks=peaks,failed_astar_edges=fail_edges,
       certificate=proof,plan_stages=getattr(plan,'_planning_profile',{}) if plan else None,plan_record=plan.record if plan else None))
 with (O/'REAL_DEPTH_12_FAILURE_MATRIX.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (O/'OBSERVABILITY_RECORDS.json').write_text(json.dumps(detail,indent=2))
 (O/'MAP_AND_PLANNING_PROFILE.json').write_text(json.dumps(profiles,indent=2))
 print('ROWS',len(rows),'PLANNER',dict(Counter((r['mode'],r['task'],r['planner']) for r in rows)),'DIRECT',dict(Counter((r['task'],r['direct_collision']) for r in rows)))
if __name__=='__main__':main()
