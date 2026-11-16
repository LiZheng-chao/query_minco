"""Bounded sensor-only observed voxel ledger. UNKNOWN is the default,never GT-completed."""
from dataclasses import dataclass,asdict
from collections import deque
import hashlib,json,time,math,threading,functools
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation,OPTICAL_TO_BODY
KNOWN_FREE=1;OCCUPIED=2;UNKNOWN=0
COMPONENT_ID="bounded_full_cell_depth_carving_epoch_ledger_v1"
@dataclass(frozen=True)
class Config:
 resolution:float=.1
 shape:tuple=(40,40,30)
 max_cells:int=60000
 max_frames:int=32
 history_ttl:float=2.
 live_age:float=.2
 depth_error:float=.1
 position_error:float=.02
 angle_error:float=math.pi/360
 pixel_guard:int=1
 max_filter_radius:int=24
 update_budget_s:float=.15
 query_budget_s:float=.05
 static_scene:bool=True
 def check(self):
  if not 1<=self.max_cells<=60000 or not .05<=self.resolution<=.5 or len(self.shape)!=3 or any(type(n)is not int or not 2<=n<=128 for n in self.shape) or np.prod(self.shape)>self.max_cells:raise ValueError('GRID_BUDGET')
  if not 1<=self.max_frames<=32 or not 0<self.history_ttl<=3 or not 0<self.live_age<=.2:raise ValueError('HISTORY_CONTRACT')
  if any(not math.isfinite(v) or v<0 for v in [self.depth_error,self.position_error,self.angle_error]) or self.angle_error>.1:raise ValueError('ERROR_BOUNDS')
  if self.pixel_guard<1 or not 1<=self.max_filter_radius<=64 or not 0<self.update_budget_s<=.15 or not 0<self.query_budget_s<=.05 or type(self.static_scene)is not bool:raise ValueError('COMPUTE_CONTRACT')
class BudgetExceeded(ValueError):pass
class CertificationError(ValueError):
 def __init__(self,record):self.record=record;super().__init__(record['reason'])
def _sha(value):return hashlib.sha256(value).hexdigest()
class Frame:
 def __init__(self,depth,pose,stamp,pose_stamp,now,scene_id,camera_id):
  d=np.asarray(depth);p=np.asarray(pose['p'],float);q=np.asarray(pose['q'],float)
  if d.dtype!=np.float32 or d.shape!=(90,160) or p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('DEPTH_POSE_SHAPE')
  if not all(math.isfinite(v) and v>0 for v in [stamp,pose_stamp,now]) or stamp!=pose_stamp or now-stamp>.2 or stamp-now>.01:raise ValueError('SOURCE_TIME_BINDING')
  if pose.get('frame','world')!='world':raise ValueError('WORLD_FRAME')
  self.depth=d.copy();self.depth.flags.writeable=False;self.p=p.copy();self.q=q.copy();self.R=rotation(q);self.stamp=stamp;self.cache={};self.filters={}
  depth_hash=_sha(d.tobytes());self.content_id=_sha((depth_hash+json.dumps([p.tolist(),q.tolist(),scene_id,camera_id])).encode())
  hit=np.isfinite(d)&(d>=.04)&(d<20);vv,uu=np.nonzero(hit);dd=d[vv,uu].astype(float)
  rays=np.stack([(uu-80)/80*dd,(vv-45)/80*dd,dd],1)
  world_hits=rays@OPTICAL_TO_BODY.T@self.R.T+self.p
  self.hit_bounds=(world_hits.min(0),world_hits.max(0)) if len(dd) else None
  self.hit_tree=cKDTree(world_hits) if len(dd) else None
  self.max_hit_depth=float(dd.max()) if len(dd) else 0.
  self.id=_sha((self.content_id+repr(stamp)).encode());self.record=dict(id=self.id,depth_sha256=depth_hash,pose=p.tolist(),quaternion=q.tolist(),stamp=stamp,pose_stamp=pose_stamp,scene_id=scene_id,camera_id=camera_id,
   calibration='native160x90_K80_80_80_45_optical_Z_origin0_Roptical_to_body_frozen_v1')
 def carve(self,centers,origin,cfg,deadline,reuse=None):
  key=tuple(origin)
  if key in self.cache:return self.cache[key]
  h=cfg.resolution;shape=cfg.shape;B=OPTICAL_TO_BODY
  free_full=np.zeros(len(centers),bool);visible_full=np.zeros(len(centers),bool);compute=np.ones(len(centers),bool)
  if self.cache:
   old_origin,old=next(iter(self.cache.items()))
   if reuse is not None and tuple(old_origin)==reuse['old_origin']:
    overlap=reuse['overlap'];ij_old=reuse['old_indices']
   else:
    ij_old=np.floor((centers-np.asarray(old_origin))/h).astype(int);overlap=((ij_old>=0)&(ij_old<np.asarray(shape))).all(1)
   free_full[overlap]=old[0][tuple(ij_old[overlap].T)];visible_full[overlap]=old[2][tuple(ij_old[overlap].T)];compute[overlap]=False
  selected=np.flatnonzero(compute);local=centers[selected]
  optical=(local-self.p)@self.R@B;x,y,z=optical.T
  error=cfg.position_error+np.linalg.norm(optical,axis=1)*math.sin(cfg.angle_error)
  r=h*math.sqrt(3)/2+error
  valid=(z-r>=.04)
  safe_z=np.maximum(z-r,1e-6)
  us=np.stack([80+80*(x+s*r)/(z+t*r).clip(min=1e-6) for s in [-1,1] for t in [-1,1]])
  vs=np.stack([45+80*(y+s*r)/(z+t*r).clip(min=1e-6) for s in [-1,1] for t in [-1,1]])
  u=80+80*x/np.maximum(z,1e-6);v=45+80*y/np.maximum(z,1e-6)
  radius=np.ceil(np.maximum.reduce([us.max(0)-u,u-us.min(0),vs.max(0)-v,v-vs.min(0)])+1.5+cfg.pixel_guard).clip(0,100000).astype(int)
  ui=np.rint(u).clip(-100000,100000).astype(int);vi=np.rint(v).clip(-100000,100000).astype(int)
  valid&=(radius<=cfg.max_filter_radius)&(ui-radius>=0)&(ui+radius<160)&(vi-radius>=0)&(vi+radius<90)
  hits=np.isfinite(self.depth)&(self.depth>=.04)&(self.depth<20)
  lower=np.where(hits,self.depth,-np.inf);free=np.zeros(len(local),bool)
  for radius_value in np.unique(radius[valid]):
   if time.perf_counter()>deadline:raise BudgetExceeded('CARVE_DEADLINE')
   idx=np.flatnonzero(valid&(radius==radius_value))
   if int(radius_value) not in self.filters:self.filters[int(radius_value)]=minimum_filter(lower,size=2*int(radius_value)+1,mode='constant',cval=-np.inf)
   conservative=self.filters[int(radius_value)]
   free[idx]=conservative[vi[idx],ui[idx]]>z[idx]+r[idx]+cfg.depth_error
  occ_full=np.zeros(len(centers),bool)
  if self.cache:occ_full[overlap]=old[1][tuple(ij_old[overlap].T)]
  uncertainty=cfg.depth_error+cfg.position_error+self.max_hit_depth*(math.sin(cfg.angle_error)+(cfg.pixel_guard+.5)/80)
  envelope=(math.ceil(uncertainty/h)+1.5)*h
  if self.hit_tree is not None:
   possible=((centers[selected]>=self.hit_bounds[0]-envelope)&(centers[selected]<=self.hit_bounds[1]+envelope)).all(1)
   selected_hits=selected[possible]
   for first in range(0,len(selected_hits),4096):
    if time.perf_counter()>deadline:raise BudgetExceeded('HIT_QUERY_DEADLINE')
    idx=selected_hits[first:first+4096];distance,_=self.hit_tree.query(centers[idx],p=np.inf)
    occ_full[idx]=distance<=envelope
  visible=valid.copy() # Only fully projectable measured cells can revoke prior free proofs.
  free_full[selected]=free;visible_full[selected]=visible
  result=(free_full.reshape(shape),occ_full.reshape(shape),visible_full.reshape(shape))
  self.cache={key:result}
  return result
def birth_covers_intersection(lo,hi,p,r,birth):
 """A conservative AABB enclosing cell intersect query-ball, not FREE carving."""
 p=np.asarray(p);lo=np.asarray(lo);hi=np.asarray(hi)
 delta=np.maximum(np.maximum(lo-p,p-hi),0.)
 d2=delta*delta;other=d2.sum(axis=-1,keepdims=True)-d2
 extent=np.sqrt(np.maximum(0.,(r+1e-12)**2-other))
 left=np.maximum(lo,p-extent);right=np.minimum(hi,p+extent)
 far=np.maximum(np.abs(left-np.asarray(birth.center)),np.abs(right-np.asarray(birth.center)))
 return (left<=right).all(axis=-1)&(np.linalg.norm(far,axis=-1)<=birth.radius-1e-12)

class Snapshot:
 def __init__(self,states,origin,cfg,frames,now,epoch,epoch_getter):
  self.states=states.copy();self.states.flags.writeable=False;self.origin=np.asarray(origin).copy();self.origin.flags.writeable=False;self.cfg=cfg;self.frames=frames;self.now=now;self.epoch=epoch;self._epoch_getter=epoch_getter
  self.valid_until=min([frames[-1]['stamp']+cfg.live_age]+[f['stamp']+cfg.history_ttl for f in frames]) if frames else now
  self.id=_sha(self.states.tobytes()+json.dumps([self.origin.tolist(),asdict(cfg),[f['id'] for f in frames],epoch],sort_keys=True).encode())
 def current(self,now):
  if not math.isfinite(now) or now<self.now-.01 or now>self.valid_until or self._epoch_getter()!=self.epoch:raise CertificationError(dict(reason='STALE_OR_INVALIDATED_MAP',snapshot=self.id))
 def ball(self,world,radius,now):
  self.current(now);p=np.asarray(world,float);h=self.cfg.resolution
  if p.shape!=(3,) or not np.isfinite(p).all() or not math.isfinite(radius) or radius<=0:raise ValueError('BALL_CONTRACT')
  low=np.floor((p-radius-self.origin)/h).astype(int);high=np.floor((p+radius-self.origin)/h).astype(int)
  if (low<0).any() or (high>=np.asarray(self.cfg.shape)).any():return dict(passed=False,reason='UNKNOWN_OUTSIDE_ROLLING_GRID',world=p.tolist(),radius=radius)
  ids=np.stack(np.meshgrid(*[np.arange(a,b+1) for a,b in zip(low,high)],indexing='ij'),-1).reshape(-1,3)
  lo=self.origin+ids*h;nearest=np.minimum(np.maximum(p,lo),lo+h);intersect=np.linalg.norm(nearest-p,axis=1)<=radius+1e-12;ids=ids[intersect]
  values=self.states[tuple(ids.T)];n_unknown=int((values==UNKNOWN).sum());n_occupied=int((values==OCCUPIED).sum())
  birth=getattr(self,'birth_ball',None)
  contained=bool(birth is not None and np.linalg.norm(p-np.asarray(birth.center))+radius<=birth.radius)
  if contained and not n_occupied:return dict(passed=True,reason='FINITE_ENGINEERING_BIRTH_BALL',unknown_covered_by_birth=n_unknown,snapshot=self.id)
  covered=0
  if birth is not None and n_unknown:
   cell_lo=self.origin+ids*h
   cover=(values==UNKNOWN)&birth_covers_intersection(cell_lo,cell_lo+h,p,radius,birth)
   covered=int(cover.sum());n_unknown-=covered
  return dict(passed=not(n_unknown or n_occupied),reason='KNOWN_FREE' if not(n_unknown or n_occupied) else ('OCCUPIED_ENVELOPE' if n_occupied else 'UNKNOWN_CELLS'),
   world=p.tolist(),radius=radius,cells=len(ids),unknown=n_unknown,unknown_covered_by_birth=covered,occupied=n_occupied,snapshot=self.id)
 def certify(self,plan,now,start=0.):
  begin=time.perf_counter();self.current(now)
  times=np.linspace(max(0,float(start)),float(plan.d.sum()),129);pad=plan.maxima[0]*float(times[1]-times[0])/2
  radius=.15+.15+pad;points=np.stack([plan.sample(t)[0] for t in times]);h=self.cfg.resolution
  low=np.floor((points-radius-self.origin)/h).astype(int);high=np.floor((points+radius-self.origin)/h).astype(int)
  outside=((low<0)|(high>=np.asarray(self.cfg.shape))).any(1)
  birth=getattr(self,'birth_ball',None)
  if birth is not None and not outside.any() and not (self.states==OCCUPIED).any() and (np.linalg.norm(points-np.asarray(birth.center),axis=1)+radius<=birth.radius).all():
   # Exact finite-birth containment plus absence of ANY occupied ROI cell.
   # This is stronger than the original per-sample occupied-cell check.
   self.current(now)
   if time.perf_counter()-begin>self.cfg.query_budget_s:raise BudgetExceeded('CURVE_CERTIFICATION_DEADLINE')
   return dict(passed=True,snapshot=self.id,epoch=self.epoch,valid_until=self.valid_until,sources=[f['id'] for f in self.frames],query_s=time.perf_counter()-begin,scope='FINITE_BIRTH_ALL_QUERY_BALLS_CONTAINED')
  n=int(math.ceil(radius/h))+1
  offsets=np.stack(np.meshgrid(*[np.arange(-n,n+1)]*3,indexing='ij'),-1).reshape(-1,3)
  # Cache exact geometric cell incidence, NOT occupancy or permission.
  key=(tuple(self.origin),tuple(self.cfg.shape),h,radius,points.tobytes())
  cache=getattr(plan,'_cell_incidence',None)
  if cache is None or cache[0]!=key:
   rows=[];cols=[]
   for first in range(0,len(points),16):
    chunk=points[first:first+16];base=np.floor((chunk-self.origin)/h).astype(int)
    ids=base[:,None,:]+offsets[None,:,:]
    in_grid=((ids>=0)&(ids<np.asarray(self.cfg.shape))).all(2)
    lo=self.origin+ids*h;near=np.minimum(np.maximum(chunk[:,None,:],lo),lo+h)
    intersect=np.linalg.norm(near-chunk[:,None,:],axis=2)<=radius+1e-12
    ri,ci=np.nonzero(intersect&in_grid)
    rows.append(ri+first);cols.append(np.ravel_multi_index(ids[ri,ci].T,self.cfg.shape))
   row=np.concatenate(rows);col=np.concatenate(cols)
   cache=(key,row,col);plan._cell_incidence=cache
  _,row,col=cache
  values=self.states.ravel()[col]
  occupied=np.bincount(row,weights=values==OCCUPIED,minlength=len(points)).astype(int)
  unknown=np.bincount(row,weights=values==UNKNOWN,minlength=len(points)).astype(int)
  birth=getattr(self,'birth_ball',None)
  if birth is not None:
   contained=np.linalg.norm(points-np.asarray(birth.center),axis=1)+radius<=birth.radius
   unknown[contained]=0 # Exact containment, not blanket UNKNOWN -> FREE.
   selection=(values==UNKNOWN)&~contained[row]
   if selection.any():
    ids=np.stack(np.unravel_index(col[selection],self.cfg.shape),axis=1)
    lo=self.origin+ids*h
    covered=birth_covers_intersection(lo,lo+h,points[row[selection]],radius,birth)
    subtract=np.bincount(row[selection][covered],minlength=len(points)).astype(int)
    unknown-=subtract
  bad=np.flatnonzero(outside|(occupied>0)|(unknown>0))
  self.current(now)
  if time.perf_counter()-begin>self.cfg.query_budget_s:raise BudgetExceeded('CURVE_CERTIFICATION_DEADLINE')
  if len(bad):
   i=int(bad[0]);raise CertificationError(dict(reason='UNKNOWN_OUTSIDE_ROLLING_GRID' if outside[i] else 'OCCUPIED_ENVELOPE' if occupied[i] else 'UNKNOWN_CELLS',
    trajectory_time=float(times[i]),world=points[i].tolist(),radius=radius,unknown=int(unknown[i]),occupied=int(occupied[i]),snapshot=self.id))
  return dict(passed=True,snapshot=self.id,epoch=self.epoch,valid_until=self.valid_until,sources=[f['id'] for f in self.frames],query_s=time.perf_counter()-begin)

def locked(fn):
 @functools.wraps(fn)
 def call(self,*args,**kwargs):
  with self.lock:return fn(self,*args,**kwargs)
 return call
class ObservedMap:
 def __init__(self,config=None,scene_id='native_static_scene'):
  self.lock=threading.RLock();self.cfg=config or Config();self.cfg.check();self.scene_id=scene_id;self.frames=deque(maxlen=self.cfg.max_frames);self.records=deque(maxlen=64);self.last_stamp=0.;self.last_content=None;self.epoch=0;self.origin=None;self._last_free=None;self.latest_snapshot=None
 @locked
 def invalidate(self,reason):
  self.frames.clear();self.epoch+=1;self._last_free=None;self.latest_snapshot=None;self.records.append(dict(reason=reason,epoch=self.epoch))
 @locked
 def reset_scene(self,scene_id):
  self.invalidate('SCENE_EPOCH_CHANGE');self.scene_id=scene_id;self.last_stamp=0.;self.last_content=None
 @locked
 def add(self,depth,pose,stamp,pose_stamp,now,camera_id='native_camera'):
  try:frame=Frame(depth,pose,stamp,pose_stamp,now,self.scene_id,camera_id)
  except (ValueError,TypeError,KeyError) as e:self.invalidate('INPUT_CONTRACT '+str(e));raise
  if stamp<self.last_stamp:self.invalidate('BACKWARD_STAMP');raise ValueError('BACKWARD_STAMP')
  if stamp==self.last_stamp:
   if frame.content_id!=self.last_content:self.invalidate('DUPLICATE_CONTENT_CONFLICT');raise ValueError('DUPLICATE_CONTENT_CONFLICT')
   return False
  if self.last_stamp and stamp-self.last_stamp>self.cfg.live_age:self.invalidate('SOURCE_GAP_HISTORY_RESET')
  self.epoch+=1 # A novel observation fences every earlier execution proof until fusion.
  self.last_stamp=stamp;self.last_content=frame.content_id
  # Replacing a repeated view is allowed only for unchanged actual measured content.
  if self.frames and frame.content_id==self.frames[-1].content_id:self.frames.pop()
  self.frames.append(frame);self.records.append(frame.record)
  return True
 @locked
 def snapshot(self,now):
  begin=time.perf_counter();deadline=begin+self.cfg.update_budget_s
  if not self.frames or now-self.last_stamp>self.cfg.live_age or self.last_stamp-now>.01:raise ValueError('NO_FRESH_DEPTH')
  horizon=self.cfg.history_ttl if self.cfg.static_scene else 0.
  usable=[f for f in self.frames if now-f.stamp<=horizon] if horizon else [self.frames[-1]]
  if not usable:raise ValueError('NO_UNEXPIRED_OBSERVATIONS')
  center=self.frames[-1].p;h=self.cfg.resolution
  origin=np.floor(center/h)*h-np.asarray(self.cfg.shape)//2*h
  previous_origin=self.origin.copy() if self.origin is not None else None
  if self.origin is None or not np.array_equal(origin,self.origin):
   self.epoch+=1;self.origin=origin;self._last_free=None
  indices=np.stack(np.meshgrid(*[np.arange(n) for n in self.cfg.shape],indexing='ij'),-1).reshape(-1,3)
  reuse=None
  if previous_origin is not None:
   old_indices=indices+np.rint((origin-previous_origin)/h).astype(int);overlap=((old_indices>=0)&(old_indices<np.asarray(self.cfg.shape))).all(1)
   reuse=dict(old_origin=tuple(previous_origin),old_indices=old_indices,overlap=overlap)
  centers=origin+(indices+.5)*h;free=np.zeros(self.cfg.shape,bool);occ=np.zeros(self.cfg.shape,bool)
  try:
   for f in usable:
    if time.perf_counter()>deadline:raise BudgetExceeded('FUSION_DEADLINE')
    ffree,focc,visible=f.carve(centers,origin,self.cfg,deadline,reuse)
    free[visible]=ffree[visible];free|=ffree;occ|=focc
   free&=~occ
   if self._last_free is not None and np.any(self._last_free&~free):self.epoch+=1;self.records.append(dict(reason='FREE_PROOF_LOST_OR_CONFLICT',epoch=self.epoch))
   states=np.where(occ,OCCUPIED,np.where(free,KNOWN_FREE,UNKNOWN)).astype(np.uint8)
   if time.perf_counter()>deadline:raise BudgetExceeded('FUSION_DEADLINE')
  except BudgetExceeded:
   self.invalidate('COMPUTE_BUDGET_EXCEEDED');raise
  self._last_free=free.copy()
  snap=Snapshot(states,origin,self.cfg,[f.record for f in usable],now,self.epoch,lambda:self.epoch);self.latest_snapshot=snap
  return snap
