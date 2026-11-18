"""Bounded single worker. Control consumes immutable, current proofs only."""
import copy,hashlib,json,time,threading
from collections import deque,Counter
import numpy as np
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
from query_minco.integration.yopo_execution_v4.trajectory import Admission,digest
from .space import Space,authorized_snapshot
from .revalidate import latest_proof
from query_minco.integration.observed_space_v1.map import Config

class Engine:
 def __init__(self,birth_loader,clock=time.time):
  self.clock=clock;self.birth_loader=birth_loader;self.ledger=ObservedMap(Config(shape=(32,24,16)))
  self.lock=threading.RLock();self.wake=threading.Event();self.closed=threading.Event()
  self.thread=None;self.frame=None;self.state=None;self.goal=None;self.goal_stamp=0.
  self.history=deque(maxlen=32);self.revalidated_sensor=None;self.last_computed_plan=None;self.pending=deque();self.evidence_overflow=False;self.revision=0;self.key=None;self.last_stamp=0.;self.committed=None;self.last_executed_plan=None;self.last_plan_time=-float('inf');self.next_plan_time=-float('inf')
  self.revoked_by_observation=0;self.events=deque(maxlen=2048);self.audit_overflow=False;self.event_sequence=0;self.counts=Counter();self.serial=0;self.published_ids=deque(maxlen=512);self.published_plan_count=0
 def record(self,event,**fields):
  if len(self.events)==self.events.maxlen:self.audit_overflow=True
  self.event_sequence+=1
  self.events.append(dict(event=event,event_sequence=self.event_sequence,monotonic=time.monotonic(),**fields));self.counts[event]+=1
 def revoke(self,reason):
  with self.lock:
   self.revision+=1;self.committed=None;self.record('REVOKED',reason=reason,map_epoch=self.revision)
 def observe(self,depth,pose,stamp,pose_stamp):
  d=np.asarray(depth);now=self.clock()
  if d.shape!=(90,160) or d.dtype!=np.float32 or stamp!=pose_stamp or not np.isfinite(stamp) or not 0<=now-stamp<=.2:
   self.revoke('INPUT_CONTRACT');raise ValueError('INPUT_CONTRACT')
  from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation
  try:
   if pose.get('frame','world')!='world' or np.asarray(pose['p']).shape!=(3,) or not np.isfinite(pose['p']).all():raise ValueError('POSE_FRAME')
   rotation(pose['q'])
  except (ValueError,KeyError,TypeError):
   self.revoke('POSE_CONTRACT');raise ValueError('POSE_CONTRACT')
  pose=dict(p=list(pose['p']),q=list(pose['q']),frame='world')
  key=hashlib.sha256(d.tobytes()+json.dumps(pose,sort_keys=True,allow_nan=False).encode()).hexdigest()
  with self.lock:
   if stamp<self.last_stamp or (stamp==self.last_stamp and key!=self.key):
    self.revision+=1;self.committed=None;raise ValueError('STAMP_CONFLICT')
   if stamp==self.last_stamp:return
   # Epoch describes geometry/pose, not arrival count. Exact repeated measured
   # content may refresh evidence but NEVER extends an already issued proof.
   if key!=self.key or stamp-self.last_stamp>.2:
    if self.committed is not None:self.revoked_by_observation+=1
    self.revision+=1;self.committed=None
   self.key=key;self.last_stamp=stamp
   pose=dict(pose,_received_ros=now)
   self.frame=(d.copy(),pose,stamp,self.revision)
   if len(self.pending)>=8:
    self.evidence_overflow=True;self.committed=None;raise ValueError('EVIDENCE_QUEUE_OVERFLOW')
   self.pending.append(self.frame);self.history.append((self.frame,now))
  self.wake.set()
 def set_state(self,state,stamp):
  x=copy.deepcopy(state)
  if not np.isfinite(stamp) or not 0<=self.clock()-stamp<=.2 or any(np.asarray(x[k]).shape!=(3,) or not np.isfinite(x[k]).all() for k in ['p','v','a']):
   self.revoke('STATE_CONTRACT');raise ValueError('STATE_CONTRACT')
  with self.lock:self.state=(x,stamp)
 def set_goal(self,goal,stamp):
  g=np.asarray(goal,float)
  if g.shape!=(3,) or not np.isfinite(g).all() or not np.isfinite(stamp) or not 0<=self.clock()-stamp<=.2:
   self.revoke('GOAL_CONTRACT');raise ValueError('GOAL_CONTRACT')
  with self.lock:self.goal=g.copy();self.goal_stamp=stamp
  self.wake.set()
 def current_revision(self):
  return self.revision
 def run_once(self):
  begin=time.perf_counter();now=self.clock();map_s=None;planning=None;cert_s=None;use_cache=False
  with self.lock:
   frame=self.frame;state=copy.deepcopy(self.state);goal=None if self.goal is None else self.goal.copy()
   goal_stamp=self.goal_stamp;old=self.committed
   if old is None and self.last_computed_plan is not None:old=dict(plan=self.last_computed_plan)
   pending=list(self.pending);history=list(self.history)
  if frame is None or state is None or goal is None:return False
  if old is None and now<self.next_plan_time and not pending:return False
  depth,pose,stamp,rev=frame
  try:
   if not 0<=now-stamp<=.2 or not 0<=now-state[1]<=.2 or not 0<=now-goal_stamp<=.2:raise ValueError('SOURCE_STALE')
   start=time.perf_counter()
   cached=self.revalidated_sensor
   use_cache=bool(cached is not None and self.last_computed_plan is not None
      and (goal>=cached.origin).all()
      and (goal<=cached.origin+np.asarray(cached.cfg.shape)*cached.cfg.resolution).all())
   if use_cache and pending:
    snap,_,sensor=latest_proof(cached,self.last_computed_plan,pending,self.birth_loader,self.clock)
    rev=pending[-1][3];stamp=pending[-1][2];grant=self.birth_loader(self.clock())
   elif use_cache:
    sensor=cached;grant=self.birth_loader(self.clock())
    snap=authorized_snapshot(sensor,grant,self.clock(),rev,lambda:rev)
   else:
    # Historical rays retain the REAL receipt time already checked by observe.
    for (dep,pos,ts,rv),received in history:
     if ts>self.ledger.last_stamp:self.ledger.add(dep,pos,ts,ts,received)
    sensor=self.ledger.snapshot(self.clock());grant=self.birth_loader(self.clock())
    snap=authorized_snapshot(sensor,grant,self.clock(),rev,lambda:rev)
   with self.lock:
    while self.pending and self.pending[0][2]<=stamp:self.pending.popleft()
   map_s=time.perf_counter()-start
   if map_s>.15:raise ValueError('MAP_DEADLINE')
   # Fusion is serviced at evidence cadence even while proposals are limited.
   if old is None and now<self.next_plan_time:
    with self.lock:self.record('EVIDENCE_FUSED_NO_PROPOSAL',source_stamp=stamp,map_s=map_s,pending=len(self.pending))
    return False
   space=Space(snap,self.clock);planning=time.perf_counter()
   replan=old is None or now>=self.next_plan_time
   if replan:
    self.next_plan_time=now+.2
    planner=DepthPlanner(self.clock,space=space)
    plan,proof=planner.propose(state[0],goal,state[1])
   else:
    plan=old['plan'];proof=snap.certify(getattr(plan,'_spatial_prepared',None) or Prepared(plan),self.clock())
   # Drain every frame received during planning; never skip intermediate hits.
   validation_base=sensor
   while True:
    with self.lock:
     newer=[x for x in self.pending if x[2]>stamp]
     current_rev=self.revision
    if current_rev==rev:break
    if not newer:raise ValueError('NONFRAME_INVALIDATION')
    snap,proof,validation_base=latest_proof(validation_base,plan,newer,self.birth_loader,self.clock)
    stamp=newer[-1][2];rev=newer[-1][3]
    with self.lock:
     if self.revision==rev:break
    if time.perf_counter()-begin>.15:raise ValueError('PLANNING_DEADLINE')
   proof.update(birth_id=snap.birth_id,sensor_snapshot=snap.sensor_snapshot)
   planning_s=time.perf_counter()-planning
   envelope=dict(kind='trajectory',plan=plan.record,pva=plan.at(self.clock()).tolist(),stamp=self.clock())
   envelope['pva']=plan.at(envelope['stamp']).tolist()
   gate=Admission();gate.plan=self.last_executed_plan
   def checked_proof(p):
    if p.id!=plan.id:raise ValueError('CERTIFICATE_PLAN_ID')
    snap.current(self.clock());return proof
   cert_s=proof.get('query_s')
   admission_start=time.perf_counter();gate.check(envelope,state[0],self.clock(),certify=checked_proof);admission_s=time.perf_counter()-admission_start
   if time.perf_counter()-begin>.15:raise ValueError('PLANNING_DEADLINE')
   commit_start=time.perf_counter()
   # Short atomic commit; no MINCO, map rebuild or certificate work in lock.
   with self.lock:
    if self.revision!=rev or self.evidence_overflow:raise ValueError('EPOCH_CHANGED_BEFORE_COMMIT')
    latest,latest_stamp=self.state
    if not 0<=self.clock()-latest_stamp<=.2:raise ValueError('STATE_STALE_BEFORE_COMMIT')
    if replan:
     actual=np.asarray([latest['p'],latest['v'],latest['a']])
     if any(np.linalg.norm(plan.sample(0)[k]-actual[k])>tol for k,tol in enumerate([.05,.1,.5])):raise ValueError('INITIAL_PVA_BEFORE_COMMIT')
    snap.current(self.clock())
    while self.pending and self.pending[0][2]<=stamp:self.pending.popleft()
    self.revalidated_sensor=validation_base;self.last_computed_plan=plan
    self.serial+=1;acceptance=digest(dict(plan=plan.id,snapshot=snap.id,serial=self.serial))
    self.committed=dict(plan=plan,proof=proof,map_epoch=rev,source_stamp=stamp,
                        snapshot=snap.id,acceptance_id=acceptance,birth_id=snap.birth_id,certificate_id=digest(dict(plan=plan.id,snapshot=snap.id,expiry=proof['valid_until'])))
    if replan:self.last_plan_time=now
    self.record('CERTIFIED_COMMITTED',plan_id=plan.id,plan=plan.record,map_epoch=rev,source_stamp=stamp,
                acceptance_id=acceptance,snapshot_id=snap.id,birth_id=snap.birth_id,replan=replan,map_s=map_s,planning_s=planning_s,
                total_s=time.perf_counter()-begin,certification_s=cert_s,admission_s=admission_s,commit_s=time.perf_counter()-commit_start,plan_stages=getattr(plan,'_planning_profile',{}),maxima=plan.maxima)
   return True
  except Exception as error:
   with self.lock:
    # A revoked candidate is not a permanent ROI to revalidate forever.
    # Rebuild from retained complete evidence after a failed cache proof.
    if use_cache:
     self.last_computed_plan=None;self.revalidated_sensor=None
    self.committed=None;self.record('REJECTED',reason=str(error),source_stamp=stamp,
                                   map_epoch=rev,total_s=time.perf_counter()-begin,map_s=map_s,planning_s=time.perf_counter()-planning if planning is not None else None,certification_s=cert_s)
   return False
 def command(self,now):
  # Bounded control work: arithmetic, one sample, no geometry or solver.
  with self.lock:
   c=self.committed
   if len(self.events)>=self.events.maxlen:self.audit_overflow=True
   if c is None or self.audit_overflow or self.evidence_overflow:return None
   if self.revision!=c['map_epoch'] or not 0<=now-c['source_stamp']<=.2 or now>c['proof']['valid_until'] or not 0<=now-self.goal_stamp<=.2:
    self.committed=None;self.record('REVOKED',reason='PROOF_EXPIRED_OR_EPOCH');return None
   if self.state is None or not 0<=now-self.state[1]<=.2:return None
   plan=c['plan']
   if not 0<=now-plan.epoch<=float(plan.d.sum())+.5:return None
   pva=plan.at(now)
   if np.linalg.norm(pva[0]-self.state[0]['p'])>.15:
    self.committed=None;self.record('REVOKED',reason='TRACKING_PROOF_ENVELOPE');return None
   proof=dict(type='OBSERVED_SPACE_FULL_TUBE_V1',plan_id=plan.id,source_stamp=c['source_stamp'],map_epoch=c['map_epoch'],snapshot_id=c['snapshot'],certificate_id=c['certificate_id'],acceptance_id=c['acceptance_id'],certificate_expiry=c['proof']['valid_until'],geometry='OBSERVED_SPACE_FULL_TUBE_129',evidence=c['proof'])
   return dict(kind='trajectory',authorization=proof,plan=c['plan'].record,stamp=now,pva=pva.tolist(),plan_id=plan.id,
               source_stamp=c['source_stamp'],map_epoch=c['map_epoch'],
               acceptance_id=c['acceptance_id'],snapshot_id=c['snapshot'],certificate_id=c['certificate_id'],certificate_expiry=c['proof']['valid_until'],depth_source_stamp=c['source_stamp'],matched_odom_stamp=c['source_stamp'])
 def take_events(self):
  # Supervisor streams these through its existing durable journal; no map work.
  with self.lock:
   rows=list(self.events);self.events.clear()
   return rows
 def with_current(self,now,action):
  """Serialize final publication with epoch revocation/worker commit."""
  with self.lock:
   envelope=self.command(now)
   if envelope is None:return None
   action(envelope)
   return envelope
 def summary(self):
  with self.lock:
   c=self.committed;now=self.clock()
   ready=bool(not self.audit_overflow and len(self.events)<self.events.maxlen and c and c['map_epoch']==self.revision and now<=c['proof']['valid_until'] and 0<=now-c['source_stamp']<=.2 and 0<=now-self.goal_stamp<=.2)
   return dict(ready=ready,map_epoch=self.revision,source_stamp=c['source_stamp'] if c else None,valid_until=c['proof']['valid_until'] if c else None,plan_id=c['plan'].id if c else None,acceptance_id=c['acceptance_id'] if c else None,published_plans=self.published_plan_count)
 def published(self,envelope):
  with self.lock:
   if self.committed is None or self.committed['plan'].id!=envelope['plan_id']:raise ValueError('PUBLICATION_IDENTITY')
   self.last_executed_plan=self.committed['plan']
   if envelope['plan_id'] not in self.published_ids:
    self.published_ids.append(envelope['plan_id']);self.published_plan_count+=1
   self.record('POSITION_COMMAND_PUBLISHED',**{k:envelope[k] for k in ['plan_id','source_stamp','map_epoch','acceptance_id','stamp']})
 def start(self):
  if self.thread is not None:raise ValueError('ALREADY_STARTED')
  def loop():
   while not self.closed.is_set():
    self.wake.wait(.02);self.wake.clear()
    if not self.closed.is_set():self.run_once()
  self.thread=threading.Thread(target=loop,name='depth-minco-worker',daemon=True);self.thread.start()
 def close(self):
  self.closed.set();self.wake.set()
  if self.thread:self.thread.join(1.)
  if self.thread and self.thread.is_alive():raise RuntimeError('PLANNING_WORKER_NOT_STOPPED')
