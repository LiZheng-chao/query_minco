"""Independent finite-birth proof; Depth may veto, never expands permission."""
import time,hashlib
from pathlib import Path
import json
import numpy as np
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
from query_minco.integration.observed_space_v1.map import Frame,Config
from query_minco.integration.observed_space_v2.core import sample_batch
from query_minco.integration.yopo_execution_v4.trajectory import digest

def tube_cells(points,radius):
 h=Config().resolution
 lo=np.floor((points.min(0)-radius)/h).astype(int);hi=np.floor((points.max(0)+radius)/h).astype(int)
 if int(np.prod(hi-lo+1))>60000:raise ValueError('F0_TUBE_BUDGET')
 ids=np.stack(np.meshgrid(*[np.arange(a,b+1) for a,b in zip(lo,hi)],indexing='ij'),-1).reshape(-1,3)
 cell_lo=ids*h;needed=np.zeros(len(ids),bool)
 for p in points:
  nearest=np.minimum(np.maximum(p,cell_lo),cell_lo+h)
  needed|=np.linalg.norm(nearest-p,axis=1)<=radius+1e-12
 return cell_lo[needed]+h/2
def prepare_geometry(plan):
 ts=np.linspace(0,float(plan.d.sum()),129)
 points=sample_batch(plan,ts)[:,0];radius=.30+plan.maxima[0]*float(ts[1]-ts[0])/2
 plan._f0_geometry=(points,radius);plan._f0_centers=tube_cells(points,radius)
 return plan

class BirthAuthority:
 def __init__(self,loader,clock):
  self.loader=loader;self.clock=clock;self.frame=None;self.cache={};self.geometries={};self.serial=0
 @classmethod
 def from_run(cls,run,clock):
  selected_path=Path(run)/'BIRTH_SELECTION.json'
  if not selected_path.exists():selected_path=Path(__file__).resolve().parents[3]/'research_outputs/t2_depth_minco_20261116_v1/BIRTH_SELECTION.json'
  selected=json.loads(selected_path.read_text())
  return cls(lambda now:BirthGrant.load(run,selected,now,time.monotonic()),clock)
 def observe(self,depth,pose,stamp):
  new=Frame(depth,pose,stamp,stamp,self.clock(),'native_static_scene','native_camera')
  if self.frame is not None and new.stamp<=self.frame.stamp:
   if new.stamp<self.frame.stamp or new.content_id!=self.frame.content_id:raise ValueError('F0_DEPTH_STAMP')
   return
  self.frame=new;self.serial+=1;self.cache.clear()
 def check(self,plan=None,pva=None,now=None):
  begin=time.perf_counter();now=self.clock() if now is None else now
  grant=self.loader(now)
  if grant is None or now>grant.expires:raise ValueError('BIRTH_EXPIRED')
  frame=self.frame
  if frame is None or not 0<=now-frame.stamp<=.2:raise ValueError('F0_DEPTH_STALE')
  if plan is not None:
   geometry=getattr(plan,'_f0_geometry',None)
   if geometry is None:
    prepare_geometry(plan);geometry=plan._f0_geometry
   points,radius=geometry;identity=plan.id
  else:
   values=np.asarray(pva,float)
   if values.shape!=(3,3) or not np.isfinite(values).all() or np.linalg.norm(values[1:])>1e-6:raise ValueError('F0_FALSE_HOLD')
   points=values[0][None];radius=.30;identity=digest(values.tolist())
  # Whole swept tube, including the original 0.15m tracking envelope.
  if np.max(np.linalg.norm(points-np.asarray(grant.center),axis=1))+radius>grant.radius:raise ValueError('OUTSIDE_BIRTH_UNKNOWN')
  key=(identity,frame.id)
  if key not in self.cache:
   cfg=Config();h=cfg.resolution
   uncertainty=cfg.depth_error+cfg.position_error+frame.max_hit_depth*(np.sin(cfg.angle_error)+(cfg.pixel_guard+.5)/80)
   envelope=(np.ceil(uncertainty/h)+1.5)*h
   if identity not in self.geometries:
    centers=getattr(plan,'_f0_centers',None) if plan is not None else None
    if centers is None:centers=tube_cells(points,radius)
    if len(self.geometries)>=3:self.geometries.clear()
    self.geometries[identity]=centers
   centers=self.geometries[identity]
   if frame.hit_tree is not None:
    possible=((centers>=frame.hit_bounds[0]-envelope)&(centers<=frame.hit_bounds[1]+envelope)).all(1)
    if possible.any() and (frame.hit_tree.query(centers[possible],p=np.inf)[0]<=envelope).any():raise ValueError('F0_NEW_OCCUPIED_EVIDENCE')
   if time.perf_counter()-begin>.05:raise ValueError('F0_CERTIFICATION_DEADLINE')
   if len(self.cache)>=2:self.cache.clear()
   self.cache[key]=dict(cells=len(centers),query_s=time.perf_counter()-begin)
  expiry=min(grant.expires,frame.stamp+.2)
  if self.clock()>expiry:raise ValueError('F0_PROOF_EXPIRED_BEFORE_PUBLICATION')
  proof=dict(plan_id=identity,source_stamp=frame.stamp,matched_odom_stamp=frame.stamp,map_epoch=self.serial,snapshot_id=frame.id,birth_id=grant.identity,certificate_expiry=expiry,geometry='FINITE_BIRTH_FULL_TUBE_129_WITH_OCCUPIED_VETO',**self.cache[key])
  proof['certificate_id']=digest(proof);proof['acceptance_id']=digest(dict(certificate_id=proof['certificate_id'],plan_id=identity))
  return proof
