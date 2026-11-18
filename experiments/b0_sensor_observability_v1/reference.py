"""Offline conservative ray reference; no GT, ROS, or production mutations."""
from pathlib import Path
import sys,json,time,hashlib,math,itertools,collections
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/b0_sensor_observability_20261118_v1';BASE=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
import numpy as np,torch
from query_minco.integration.observed_space_v1.map import Config,Snapshot,birth_covers_intersection
from query_minco.integration.observed_space_v2.core import ObservedMap
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation,OPTICAL_TO_BODY
CFG=Config(shape=(32,24,16));SPAWN=np.array([.25,-.25,2.])
def write(n,d):(O/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def project(centers,pose,cfg=CFG):
 optical=(centers-np.asarray(pose['p']))@rotation(pose['q'])@OPTICAL_TO_BODY;x,y,z=optical.T
 r=cfg.resolution*math.sqrt(3)/2+cfg.position_error+np.linalg.norm(optical,axis=1)*math.sin(cfg.angle_error)
 us=np.stack([80+80*(x+s*r)/np.maximum(z+t*r,1e-6) for s in [-1,1] for t in [-1,1]])
 vs=np.stack([45+80*(y+s*r)/np.maximum(z+t*r,1e-6) for s in [-1,1] for t in [-1,1]])
 u=80+80*x/np.maximum(z,1e-6);v=45+80*y/np.maximum(z,1e-6);pad=1.5+cfg.pixel_guard
 radius=np.ceil(np.maximum.reduce([us.max(0)-u,u-us.min(0),vs.max(0)-v,v-vs.min(0)])+pad).clip(0,100000).astype(int)
 ui=np.rint(u).clip(-100000,100000).astype(int);vi=np.rint(v).clip(-100000,100000).astype(int)
 square=np.stack([ui-radius,ui+radius,vi-radius,vi+radius],1)
 # Conservative rectangular envelope of SAME full-cell circumsphere + errors.
 # Box projection contains the sphere; guard margin and max-depth test unchanged.
 rectangle=np.stack([np.floor(us.min(0)-pad),np.ceil(us.max(0)+pad),np.floor(vs.min(0)-pad),np.ceil(vs.max(0)+pad)],1).clip(-100000,100000).astype(int)
 return dict(optical=optical,r=r,z=z,radius=radius,square=square,rectangle=rectangle,projection=np.stack([us.min(0),us.max(0),vs.min(0),vs.max(0)],1))
def evaluate(depth,g,mode,cfg=CFG):
 windows=g['square'] if mode in ['current','uncapped_square'] else g['rectangle'];free=np.zeros(len(windows),bool);reasons=[]
 for k,(left,right,top,bottom) in enumerate(windows):
  why=[]
  if g['z'][k]-g['r'][k]<.04:why.append('NEAR_PLANE')
  if mode=='current' and g['radius'][k]>cfg.max_filter_radius:why.append('COMPUTE_WINDOW_LIMIT_24')
  if left<0 or right>=160 or top<0 or bottom>=90:why.append('FULL_ERROR_WINDOW_OUTSIDE_FOV')
  if not why:
   w=depth[top:bottom+1,left:right+1];valid=np.isfinite(w)&(w>=.04)&(w<20)
   if not valid.all():why.append('NO_VALID_DEPTH_RETURN')
   elif not w.min()>g['z'][k]+g['r'][k]+cfg.depth_error:why.append('OCCLUDED_OR_INSUFFICIENT_DEPTH_MARGIN')
  free[k]=not why;reasons.append(why)
 return free,reasons

def make_snapshot(sensor,free,now,pose_identity='OFFLINE_REFERENCE_BIRTH_NOT_PHYSICAL_QUALIFICATION'):
 states=np.where(sensor.states==2,2,np.where(free.reshape(sensor.states.shape),1,0)).astype(np.uint8)
 ref=Snapshot(states,sensor.origin,sensor.cfg,sensor.frames,now,sensor.epoch,lambda:sensor.epoch)
 ref.valid_until=sensor.valid_until
 return authorized_snapshot(ref,BirthGrant(tuple(SPAWN),.6,now+.2,pose_identity),now,ref.epoch,lambda:ref.epoch)
def certify(snapshot,plan,now):
 begin=time.perf_counter()
 try:proof=snapshot.certify(plan,now);status='CERTIFIED'
 except Exception as ex:proof=getattr(ex,'record',dict(reason=str(ex)));status='REJECTED'
 return dict(status=status,proof=proof,elapsed_s=time.perf_counter()-begin)
def main():
 torch.set_num_threads(1);records=json.loads((BASE/'capture_ordered/INPUTS.json').read_text())['records'];ledger=ObservedMap(CFG);rows=[];cache={};first=None
 for item in records:
  now=item['stamp'];dep=np.load(item['path'])['depth'];assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'];ledger.add(dep,item['pose'],now,now,now);sensor=ledger.snapshot(now)
  ids=np.indices(CFG.shape).transpose(1,2,3,0);centers=(sensor.origin+(ids+.5)*.1).reshape(-1,3);key=hashlib.sha256(dep.tobytes()).hexdigest()
  if key not in cache:
   begin=time.perf_counter();g=project(centers,item['pose']);masks={mode:evaluate(dep,g,mode) for mode in ['current','uncapped_square','conservative_rectangle']};cache[key]=(g,masks,time.perf_counter()-begin)
  g,masks,reference_s=cache[key];snaps={mode:make_snapshot(sensor,v[0],now) for mode,v in masks.items()}
  plans=[]
  for distance in [.6,.8]:
   plan=build(SPAWN,np.zeros(3),np.zeros(3),item['pose']['q'],SPAWN+[distance,0,0],now)
   plans.append(dict(distance=distance,plan_id=plan.id,maxima=plan.maxima,certification={mode:certify(snap,plan,now) for mode,snap in snaps.items()}))
  row=dict(frame=item['index'],stamp=now,depth_content_sha256=key,reference_s=reference_s,free_cells={mode:int(v[0].sum()) for mode,v in masks.items()},plans=plans,valid_until=min(s.valid_until for s in snaps.values()))
  rows.append(row)
  if first is None:
   q=np.array([.55,-.25,2.]);rad=.33;lo=centers-.05;hi=centers+.05;intersect=np.linalg.norm(np.minimum(np.maximum(q,lo),hi)-q,axis=1)<=rad+1e-12;grant=snaps['current'].birth_ball;covered=birth_covers_intersection(lo,hi,q,rad,grant);states=snaps['current'].states.ravel();selected=np.flatnonzero(intersect&(states==0)&~covered);voxels=[]
   for j in selected:
    corners=centers[j]+np.array(list(itertools.product([-.05,.05],repeat=3)));opt=(corners-np.asarray(item['pose']['p']))@rotation(item['pose']['q'])@OPTICAL_TO_BODY
    uv=np.stack([80+80*opt[:,0]/np.maximum(opt[:,2],1e-6),45+80*opt[:,1]/np.maximum(opt[:,2],1e-6)],1)
    actual_out=bool((opt[:,2]<.04).any() or (uv[:,0]<0).any() or (uv[:,0]>159).any() or (uv[:,1]<0).any() or (uv[:,1]>89).any())
    voxels.append(dict(world_center=centers[j].tolist(),world_bounds=[lo[j].tolist(),hi[j].tolist()],optical_center=g['optical'][j].tolist(),full_error_radius=float(g['r'][j]),continuous_projection_interval=g['projection'][j].tolist(),legacy_pixel_window=g['square'][j].tolist(),rectangular_pixel_window=g['rectangle'][j].tolist(),legacy_radius=int(g['radius'][j]),exceeds_compute_cap=bool(g['radius'][j]>24),birth_covers_query_intersection=False,actual_nominal_cell_corners_outside_fov=actual_out,nominal_projected_voxel_corners=uv.tolist(),evaluations={mode:dict(free=bool(v[0][j]),reasons=v[1][j]) for mode,v in masks.items()}))
   first=dict(query=q.tolist(),radius=rad,unauthorized_unknown_voxels=len(voxels),voxels=voxels,reference_current_free_mask_equals_production=bool(np.array_equal(masks['current'][0].reshape(CFG.shape)&(sensor.states!=2),sensor.states==1)))
 result=dict(scope='OFFLINE_CONSERVATIVE_GEOMETRY_NOT_FLIGHT_QUALIFICATION',input_frames=len(rows),unique_depth_contents=len(cache),voxel_diagnosis=first,frames=rows,certified_counts={mode:{str(d):sum(row['plans'][k]['certification'][mode]['status']=='CERTIFIED' for row in rows) for k,d in enumerate([.6,.8])} for mode in ['current','uncapped_square','conservative_rectangle']},reference_definition='uncapped_square is algebraically identical to Frame.carve without radius24 admission bound; conservative_rectangle checks all pixels covering SAME error box with unchanged guard and depth margin. Neither clips out-of-FOV windows nor accepts no-return.',GT_used=False,exit_code=0)
 write('B0_EXACT_RAY_REFERENCE.json',result);print('EXACT',result['certified_counts'],'VOXELS',len(first['voxels']),'NOMINAL_FOV',sum(x['actual_nominal_cell_corners_outside_fov'] for x in first['voxels']),flush=True)
if __name__=='__main__':main()
