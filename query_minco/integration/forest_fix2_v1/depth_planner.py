"""Offline/background E1-B development adapter. No execution authority and no GT route input."""
import numpy as np
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared
from query_minco.integration.observed_space_v1.space import MapSpace
from query_minco.integration.yopo_execution_v4.trajectory import build
class DepthPlanner:
 def __init__(self,clock):
  self.ledger=ObservedMap();self.space=MapSpace(self.ledger,clock=clock);self.clock=clock
 def observe(self,depth,camera_pose,stamp):
  self.ledger.add(depth,camera_pose,stamp,stamp,self.clock())
 def propose(self,state,goal,stamp):
  if not 0<=self.clock()-stamp<=.2:raise ValueError('STATE_SOURCE_STALE')
  # Frozen deterministic A*:all route edges require observed free-space,never GT.
  local=self.space.search_goal(state['p'],goal)
  plan=build(np.asarray(state['p']),np.asarray(state['v']),np.asarray(state['a']),state['q'],local,stamp)
  snap=self.space.snapshot();proof=snap.certify(Prepared(plan),self.clock())
  return plan,proof
