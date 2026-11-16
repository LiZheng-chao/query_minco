"""Finite birth grant + immutable sensor snapshot; no GT route input."""
import hashlib,json,time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from query_minco.integration.observed_space_v1.map import Snapshot,UNKNOWN,KNOWN_FREE
from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace as Search

@dataclass(frozen=True)
class BirthGrant:
 center:tuple
 radius:float
 expires:float
 identity:str
 @classmethod
 def load(cls,run,selected,now,mono):
  run=Path(run);raw=(run/'BIRTH_CERTIFICATE.json').read_bytes()
  truth=json.loads((run/'truth_data/HEARTBEAT.json').read_text())
  args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
  return cls.from_records(raw,truth,args,selected,now,mono)
 @classmethod
 def from_records(cls,raw,truth,args,selected,now,mono):
  d=json.loads(raw)
  nonce=next(x.split(':=',1)[1] for x in args if x.startswith('_run_nonce:='))
  c=np.asarray(d['center_world'],float)
  if c.shape!=(3,) or not np.isfinite(c).all() or c.tolist()!=selected['selected']['position']:raise ValueError('BIRTH_CENTER')
  if d['scope']!='ENGINEERING_BIRTH_BALL_ONLY' or d['radius_m']!=.60 or d['map_id']!=selected['map_id'] or d['run_nonce']!=nonce:raise ValueError('BIRTH_IDENTITY')
  if not all(np.isfinite(d[k]) for k in ['issued','expires']) or not 0<d['expires']-d['issued']<=30 or not d['issued']<=now<=d['expires']:raise ValueError('BIRTH_EXPIRED')
  if selected['selected']['clearance_lower_bound_m']<.60:raise ValueError('BIRTH_SELECTION')
  if truth['status']!='PASS' or truth['map_id']!=d['map_id'] or not 0<=mono-truth['monotonic']<=.2:raise ValueError('BIRTH_TRUTH_INVALID')
  until=min(d['expires'],now+.2-(mono-truth['monotonic']))
  return cls(tuple(c),.60,until,hashlib.sha256(raw).hexdigest())

def authorized_snapshot(sensor,grant,now,epoch,epoch_getter):
 """Authorize only WHOLE cells in finite grant; preserve OCCUPIED and provenance."""
 states=sensor.states.copy();mask=np.zeros(states.shape,bool)
 if grant is not None:
  if now>grant.expires:raise ValueError('BIRTH_EXPIRED')
  ij=np.indices(states.shape).transpose(1,2,3,0)
  centers=sensor.origin+(ij+.5)*sensor.cfg.resolution
  whole=np.linalg.norm(np.abs(centers-np.asarray(grant.center))+sensor.cfg.resolution/2,axis=-1)<=grant.radius-1e-12
  mask=whole&(states==UNKNOWN)
  states[mask]=KNOWN_FREE
 snap=Snapshot(states,sensor.origin,sensor.cfg,sensor.frames,now,epoch,epoch_getter)
 snap.valid_until=min(sensor.valid_until,grant.expires if grant else sensor.valid_until)
 snap.birth_ball=grant
 snap.birth_cells=int(mask.sum());snap.birth_id=grant.identity if grant else None
 snap.sensor_snapshot=sensor.id
 snap.id=hashlib.sha256((snap.id+json.dumps(dict(birth_id=snap.birth_id,valid_until=snap.valid_until),sort_keys=True)).encode()).hexdigest()
 return snap

class Space:
 def __init__(self,snapshot,clock):
  self.s=snapshot;self.clock=clock;self.deadline=None;self.cache={};self.unsafe_tree=None;self.tree_ready=False
 def snapshot(self):
  self.s.current(self.clock());return self.s
 def ball_free(self,p,r):
  if self.deadline is not None and time.perf_counter()>self.deadline:raise ValueError('SEARCH_DEADLINE')
  snap=self.snapshot();p=np.asarray(p,float)
  if p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(r) or r<=0:raise ValueError('BALL_CONTRACT')
  key=(tuple(p),float(r))
  if key in self.cache:return self.cache[key]
  if not self.tree_ready:
   ids=np.argwhere(snap.states!=KNOWN_FREE)
   self.unsafe_tree=cKDTree(snap.origin+(ids+.5)*snap.cfg.resolution) if len(ids) else None
   self.tree_ready=True
  h=snap.cfg.resolution
  low=np.floor((p-r-snap.origin)/h);high=np.floor((p+r-snap.origin)/h)
  inside=(low>=0).all() and (high<np.asarray(snap.cfg.shape)).all()
  # A cube fits in its circumsphere. A farther unsafe-cell center proves
  # no unsafe cube intersects the query ball; all near cases use exact original.
  separated=self.unsafe_tree is None or self.unsafe_tree.query(p)[0]>r+np.sqrt(3)*h/2+1e-12
  passed=True if inside and separated else snap.ball(p,r,self.clock())['passed']
  if len(self.cache)<8192:self.cache[key]=passed
  if self.deadline is not None and time.perf_counter()>self.deadline:raise ValueError('SEARCH_DEADLINE')
  return passed
 def search_goal(self,p,g):
  self.deadline=time.perf_counter()+.05
  try:return Search.search_goal(self,p,g)
  finally:self.deadline=None
