"""Depth-only candidate generation. NO oracle import or GT handle exists here."""
import time,hashlib,types,math
import numpy as np
from scipy.spatial import cKDTree
from query_minco.integration.observed_space_v1.map import Config,Frame
from query_minco.integration.observed_space_v2.core import ObservedMap
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot,Space
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace
from query_minco.integration.yopo_execution_v4.trajectory import build
def validate_input(depth,pose,stamp,actual_position=None):
 d=np.asarray(depth)
 if d.shape!=(90,160) or d.dtype!=np.float32 or (np.isfinite(d)&(d>=.04)&(d<=20)).mean()<.01:raise ValueError('SIM_DEPTH_MISSING_OR_FORMAT')
 if actual_position is not None:
  frame=Frame(d,pose,stamp,stamp,stamp,'SIM_OBSTACLE_VETO','native_camera');cfg=Config();h=.1
  U=cfg.position_error+cfg.depth_error*frame.hit_ray_norm+(frame.hit_depths+cfg.depth_error)*math.sqrt(2)*(cfg.pixel_guard+.5)/80+2*math.sin(cfg.angle_error/2)*frame.hit_depths*frame.hit_ray_norm
  envelope=(np.ceil(U/h)+1.5)*h;distance=np.linalg.norm(np.maximum(np.abs(frame.hit_points-np.asarray(actual_position))-envelope[:,None],0),axis=1)
  if (distance<=.30).any():raise ValueError('NEW_OBSERVED_OCCUPIED_NEAR_BODY')

class CandidateSpace:
 def __init__(self,snapshot):
  self.snapshot=snapshot;ids=np.argwhere(snapshot.states==2);self.tree=cKDTree(snapshot.origin+(ids+.5)*.1) if len(ids) else None;self.deadline=time.perf_counter()+.05
 def ball_free(self,p,r):
  if time.perf_counter()>self.deadline:raise ValueError('CANDIDATE_SEARCH_DEADLINE')
  p=np.asarray(p);s=self.snapshot
  if (p-r<s.origin).any() or (p+r>=s.origin+np.array(s.cfg.shape)*.1).any():return False
  # Candidate-only conservative measured-occupied veto. UNKNOWN remains UNKNOWN
  # and is NEVER execution authorization or a map FREE update.
  return self.tree is None or self.tree.query(p)[0]>r+np.sqrt(3)*.05
class LatestHitSpace:
 def __init__(self,depth,pose,stamp):
  self.deadline=time.perf_counter()+.05;self.cfg=Config(shape=(32,24,16));self.origin=np.floor(np.array(pose['p'])/.1)*.1-np.array(self.cfg.shape)//2*.1
  frame=Frame(depth,pose,stamp,stamp,stamp,'SIM_CANDIDATE_ONLY','native_camera');h=.1
  error=self.cfg.position_error+self.cfg.depth_error*frame.hit_ray_norm+(frame.hit_depths+self.cfg.depth_error)*math.sqrt(2)*(self.cfg.pixel_guard+.5)/80+2*math.sin(self.cfg.angle_error/2)*frame.hit_depths*frame.hit_ray_norm
  radii=(np.ceil(error/h)+1.5)*h;self.groups=[];self.cache={}
  for radius in np.unique(radii):
   pts=frame.hit_points[radii==radius];self.groups.append((radius,cKDTree(pts),pts.min(0),pts.max(0)))
  self.deadline=time.perf_counter()+.05
 def ball_free(self,p,r):
  if time.perf_counter()>self.deadline:raise ValueError('CANDIDATE_SEARCH_DEADLINE')
  p=np.asarray(p)
  if (p-r<self.origin).any() or (p+r>=self.origin+np.array(self.cfg.shape)*.1).any():return False
  key=(tuple(p),float(r))
  if key in self.cache:return self.cache[key]
  for radius,tree,lo,hi in self.groups:
   if np.max(np.maximum(np.maximum(lo-p,p-hi),0))>radius+r:continue
   if tree.query(p,p=np.inf)[0]<=radius+r:self.cache[key]=False;return False
  self.cache[key]=True;return True

class CandidatePlanner:
 def __init__(self):self.ledger=ObservedMap(Config(shape=(32,24,16)))
 def propose(self,depth,pose,state,goal,stamp,*,allow_diagnostic=False,prefer_strict=True):
  validate_input(depth,pose,stamp)
  if allow_diagnostic and not prefer_strict:
   tick=time.perf_counter();space=LatestHitSpace(depth,pose,stamp);horizon=np.clip(goal,space.origin+.33,space.origin+np.array(space.cfg.shape)*.1-.33);local=ObservedSpace.search_goal(space,np.array(state['p']),horizon);astar_s=time.perf_counter()-tick
   delta=local-np.array(state['p']);distance=np.linalg.norm(delta)
   if distance>.4:local=np.array(state['p'])+delta*(.4/distance)
   tick=time.perf_counter();plan=build(np.array(state['p']),np.array(state['v']),np.array(state['a']),state['q'],local,stamp)
   return plan,dict(mode='SIM_LATEST_DEPTH_HIT_ASTAR_DIAGNOSTIC_NOT_STRICT_DEPTH',source_stamp=stamp,depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),local_goal=local.tolist(),fixed_lookahead=.4,astar_s=astar_s,minco_s=time.perf_counter()-tick,execution_authority=False)
  validate_input(depth,pose,stamp)
  start=time.perf_counter();self.ledger.add(depth,pose,stamp,stamp,stamp);sensor=self.ledger.snapshot(stamp);map_s=time.perf_counter()-start
  snap=authorized_snapshot(sensor,BirthGrant((.25,-.25,2.),.6,stamp+.2,'OFFLINE_OR_SIM_STARTUP_ONLY'),stamp,sensor.epoch,lambda:sensor.epoch)
  strict_reason=None
  try:
   plan,proof=DepthPlanner(lambda:stamp,space=Space(snap,lambda:stamp)).propose(state,np.array(goal),stamp)
   return plan,dict(mode='STRICT_DEPTH_CANDIDATE',source_stamp=stamp,depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),proof=proof,map_s=map_s)
  except Exception as ex:strict_reason=str(ex)
  if not allow_diagnostic:raise ValueError('STRICT_DEPTH_NO_CANDIDATE '+strict_reason)
  space=CandidateSpace(sensor);horizon=np.clip(goal,sensor.origin+.33,sensor.origin+np.array(sensor.cfg.shape)*.1-.33);tick=time.perf_counter();local=ObservedSpace.search_goal(space,np.array(state['p']),horizon);astar_s=time.perf_counter()-tick
  tick=time.perf_counter();plan=build(np.array(state['p']),np.array(state['v']),np.array(state['a']),state['q'],local,stamp);minco_s=time.perf_counter()-tick
  return plan,dict(mode='SIM_DEPTH_OCCUPANCY_ASTAR_DIAGNOSTIC_NOT_STRICT_DEPTH',strict_failure=strict_reason,source_stamp=stamp,depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),map_epoch=sensor.epoch,local_goal=local.tolist(),map_s=map_s,astar_s=astar_s,minco_s=minco_s,execution_authority=False)
