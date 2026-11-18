"""First-blocker and fixed goal-directed MINCO probes; never uses GT."""
from pathlib import Path
import sys,json,time,math,collections,hashlib,subprocess,os
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
import numpy as np,torch
from query_minco.integration.observed_space_v2.core import ObservedMap
from query_minco.integration.observed_space_v1.map import Config
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot
from query_minco.integration.yopo_execution_v4.trajectory import build

def write(name,d):(O/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(1);records=json.loads((O/'capture_ordered/INPUTS.json').read_text())['records'];rows=json.loads((O/'replay_ordered/ROUTE_FRAMES.json').read_text());i=records[0];now=i['stamp'];dep=np.load(i['path'])['depth'];cfg=Config(shape=(32,24,16));ledger=ObservedMap(cfg)
 begin=time.perf_counter();ledger.add(dep,i['pose'],now,now,now);sensor=ledger.snapshot(now);map_s=time.perf_counter()-begin
 snap=authorized_snapshot(sensor,BirthGrant((.25,-.25,2.),.6,now+.2,'NONPHYSICAL_ENGINEERING_BIRTH_NOT_QUALIFICATION'),now,sensor.epoch,lambda:sensor.epoch)
 probes=[]
 for distance in [.6,.8]:
  begin=time.perf_counter();plan=build(np.array(i['pose']['p']),np.zeros(3),np.zeros(3),i['pose']['q'],np.array(i['pose']['p'])+[distance,0,0],now);minco_s=time.perf_counter()-begin
  begin=time.perf_counter()
  try:proof=snap.certify(plan,now);status='CERTIFIED'
  except Exception as ex:proof=getattr(ex,'record',dict(reason=str(ex)));status='REJECTED'
  probes.append(dict(scope='GOAL_DIRECTED_DIAGNOSTIC_NOT_ASTAR_OUTPUT',distance=distance,plan=plan.record,maxima=plan.maxima,minco_s=minco_s,certification_s=time.perf_counter()-begin,status=status,proof=proof))
 diagnostics=[]
 for cell in rows[0]['forward_probe']['blocking_cells']:
  world=np.array(cell['center']);delta=world-np.array(i['pose']['p']);x,y,z=-delta[1],-delta[2],delta[0];radius=.1*math.sqrt(3)/2+cfg.position_error+np.linalg.norm(delta)*math.sin(cfg.angle_error)
  u=80+80*x/z;v=45+80*y/z
  us=[80+80*(x+s*radius)/max(z+t*radius,1e-6) for s in [-1,1] for t in [-1,1]];vs=[45+80*(y+s*radius)/max(z+t*radius,1e-6) for s in [-1,1] for t in [-1,1]]
  pixel_radius=math.ceil(max(max(us)-u,u-min(us),max(vs)-v,v-min(vs))+1.5+cfg.pixel_guard);ui=int(np.rint(u));vi=int(np.rint(v));why=[]
  if z-radius<.04:why.append('NEAR_PLANE')
  if pixel_radius>cfg.max_filter_radius:why.append('PROJECTION_WINDOW_EXCEEDS_COMPUTE_BOUND')
  if ui-pixel_radius<0 or ui+pixel_radius>=160 or vi-pixel_radius<0 or vi+pixel_radius>=90:why.append('FULL_ERROR_WINDOW_OUTSIDE_CAMERA_FOV')
  window=None
  if not why:
   w=dep[vi-pixel_radius:vi+pixel_radius+1,ui-pixel_radius:ui+pixel_radius+1];valid=np.isfinite(w)&(w>=.04)&(w<20)
   if not valid.all():why.append('MISSING_OR_NO_RETURN_DEPTH')
   elif not w.min()>z+radius+cfg.depth_error:why.append('INSUFFICIENT_DEPTH_MARGIN')
   window=dict(valid_pixels=int(valid.sum()),pixels=w.size,min_valid_depth=float(w[valid].min()) if valid.any() else None)
  diagnostics.append(dict(cell=cell,optical=[x,y,z],projected_center=[u,v],pixel_radius=pixel_radius,reasons=why,window=window))
 write('FIRST_BLOCKER_DETAILS.json',dict(map_s=map_s,start_tube_030=snap.ball(i['pose']['p'],.30,now),start_tube_033=snap.ball(i['pose']['p'],.33,now),map_states=dict(collections.Counter(map(str,snap.states.ravel()))),first_route=rows[0],forward_unknown_diagnostics=diagnostics,reason_counts=dict(collections.Counter(reason for x in diagnostics for reason in x['reasons'])),minco_probes=probes,GT_used=False,exit_code=0))
 print('PROBES',[(x['distance'],x['status'],x['proof']) for x in probes]);print('UNKNOWN_CAUSES',collections.Counter(reason for x in diagnostics for reason in x['reasons']))
if __name__=='__main__':main()
