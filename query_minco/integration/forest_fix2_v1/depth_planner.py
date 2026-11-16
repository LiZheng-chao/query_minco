"""Offline/background E1-B development adapter. No execution authority and no GT route input."""
import numpy as np,time
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared
from query_minco.integration.observed_space_v1.space import MapSpace
from query_minco.integration.yopo_execution_v4.trajectory import build
class DepthPlanner:
 def __init__(self,clock,space=None):
  self.ledger=ObservedMap() if space is None else None;self.space=MapSpace(self.ledger,clock=clock) if space is None else space;self.clock=clock
 def observe(self,depth,camera_pose,stamp):
  self.ledger.add(depth,camera_pose,stamp,stamp,self.clock())
 def propose(self,state,goal,stamp):
  if not 0<=self.clock()-stamp<=.2:raise ValueError('STATE_SOURCE_STALE')
  # Frozen deterministic A*:all route edges require observed free-space,never GT.
  snapshot=self.space.snapshot()
  p=np.asarray(state['p'])
  # Search toward the task goal inside the bounded ROI, not a fixed forward
  # point that could itself be inside an obstacle. Frozen A* returns the first
  # local route waypoint (>=0.5m), so MINCO remains a short receding segment.
  horizon=np.clip(np.asarray(goal),snapshot.origin+.33,snapshot.origin+np.asarray(snapshot.cfg.shape)*snapshot.cfg.resolution-.33)
  start=time.perf_counter();local=self.space.search_goal(p,horizon);astar_s=time.perf_counter()-start
  start=time.perf_counter()
  plan=build(np.asarray(state['p']),np.asarray(state['v']),np.asarray(state['a']),state['q'],local,stamp)
  minco_s=time.perf_counter()-start
  snap=self.space.snapshot()
  if snap is not snapshot:raise ValueError('SNAPSHOT_CHANGED_DURING_SEARCH')
  plan._spatial_prepared=Prepared(plan)
  start=time.perf_counter();proof=snap.certify(plan._spatial_prepared,self.clock());cert_s=time.perf_counter()-start
  plan._planning_profile=dict(astar_s=astar_s,minco_s=minco_s,certification_s=cert_s)
  return plan,proof
