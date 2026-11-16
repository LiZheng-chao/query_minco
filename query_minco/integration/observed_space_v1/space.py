"""Shared map-to-planner interface. Search remains the frozen deterministic A*."""
import time,numpy as np
from .map import ObservedMap,Config,CertificationError
from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace as FrozenSearch
class MapSpace:
 def __init__(self,ledger,clock=time.time):
  self.ledger=ledger;self.clock=clock;self.cached=None;self.cached_stamp=None;self.proofs={};self.search_deadline=None
 @property
 def stamp(self):return self.ledger.last_stamp
 def snapshot(self):
  now=self.clock()
  if self.cached is None or self.cached_stamp!=self.ledger.last_stamp or self.cached.epoch!=self.ledger.epoch or now>self.cached.valid_until:
   self.cached=self.ledger.snapshot(now);self.cached_stamp=self.ledger.last_stamp
  self.cached.current(now);return self.cached
 def ball_free(self,world,radius):
  if self.search_deadline is not None and time.perf_counter()>self.search_deadline:raise CertificationError(dict(reason='SEARCH_DEADLINE'))
  return self.snapshot().ball(world,radius,self.clock())['passed']
 def certify(self,plan):
  snapshot=self.snapshot();now=self.clock();old=self.proofs.get(plan.id)
  if old and old['epoch']==snapshot.epoch and now<=old['valid_until']:return old
  proof=snapshot.certify(plan,now);self.proofs={plan.id:proof};return proof
 def search_goal(self,start,goal):
  self.search_deadline=time.perf_counter()+.05
  try:return FrozenSearch.search_goal(self,start,goal)
  finally:self.search_deadline=None
def bootstrap(snapshot,position,now,body_radius=.15,tracking_envelope=.15):
 # Actual vehicle occupancy is not proof of an unobserved tracking/safety annulus.
 result=snapshot.ball(position,body_radius+tracking_envelope,now)
 return dict(status='STATIC_PASS' if result['passed'] else 'FAIL',scope='initial body+tracking tube observed coverage',valid_until=snapshot.valid_until,epoch=snapshot.epoch,
  evidence=result,independent_contract='every intersecting voxel requires sensor-backed full-cell proof;no user/GT free bubble accepted')
def diagnose_legacy_ball(depth,pose,world,radius):
 from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation,OPTICAL_TO_BODY
 d=np.asarray(depth);p=np.asarray(world);R=rotation(pose['q']);optical=(p-np.asarray(pose['p']))@R@OPTICAL_TO_BODY;x,y,z=optical
 result=dict(world=p.tolist(),optical=optical.tolist(),radius=float(radius),near_plane=.04)
 if z-radius<.04:return dict(result,reason='NEAR_FIELD_BLIND_ZONE')
 us=[80+80*a/b for a in [x-radius,x+radius] for b in [z-radius,z+radius]];vs=[45+80*a/b for a in [y-radius,y+radius] for b in [z-radius,z+radius]]
 lo=np.floor([min(us),min(vs)]).astype(int);hi=np.ceil([max(us),max(vs)]).astype(int)
 result.update(rectangle=[lo.tolist(),hi.tolist()])
 if (lo<0).any() or hi[0]>=160 or hi[1]>=90:return dict(result,reason='OUTSIDE_FOV')
 block=d[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
 if not (np.isfinite(block)&(block>=.04)&(block<20)).all():return dict(result,reason='INVALID_OR_NO_RETURN_DEPTH')
 nearest=float(block.min());result['nearest_optical_Z']=nearest
 if nearest+.1<z-radius:return dict(result,reason='OCCLUDED_BY_NEARER_SURFACE')
 if nearest<=z+radius+.1:return dict(result,reason='OBSERVED_SURFACE_OR_CONSERVATIVE_RECTANGLE_OVERLAP')
 return dict(result,reason='LEGACY_RECTANGLE_FREE')
