"""Separate SIM bridge; existing strict Engine and ROS Supervisor stay untouched."""
from pathlib import Path
import json,time,threading,multiprocessing,collections,hashlib
import numpy as np
from experiments.sim_dev_oracle_v1.planner import CandidatePlanner,validate_input
from experiments.sim_dev_oracle_v1.state_sync import StatePairs,StateInvalid,validate_sample
from experiments.sim_dev_oracle_v1.guardian import Guardian,MODE
from query_minco.integration.yopo_execution_v4.trajectory import Plan,digest,build

def guardian_process(conn,grid,nonce):
 try:
  until=time.monotonic()+15
  while not Path(str(grid)+'.json').exists() and time.monotonic()<until:time.sleep(.02)
  monitor=Guardian(grid,nonce);conn.send(dict(ready=True))
  while True:
   request=conn.recv()
   if request is None:return
   try:conn.send(dict(ok=True,proof=monitor.check(**request)))
   except Exception as ex:conn.send(dict(ok=False,reason=str(ex)))
 except EOFError:pass
 except Exception as ex:
  try:conn.send(dict(ok=False,reason='GUARDIAN_PROCESS '+str(ex)))
  except Exception:pass
 finally:conn.close()
class Engine:
 def __init__(self,run,clock):
  self.run=Path(run);self.clock=clock;runtime=json.loads((self.run/'RUNTIME_PARAMETERS.json').read_text());self.nonce=runtime['run_nonce'];self.mode=runtime['experiment_mode']
  if self.mode!=MODE:raise ValueError('SIM_MODE_REQUIRED')
  self.lock=threading.RLock();self.stop=threading.Event();self.pending=collections.deque();self.events=collections.deque();self.audit_overflow=False;self.evidence_overflow=False;self.revision=0;self.frame=None;self.actual=None;self.actual_pair=None;self.state_pending=False;self.goal=None;self.committed=None;self.published_ids=set();self.published_plan_count=0;self.next_plan=0.;self.last_executed=None;self.last_stamp=0.;self.guardian_ready=False;self.proposed=None;self.input_fault=False
  self.sensor_dir=self.run/'sensors';self.sensor_dir.mkdir();self.sensor_index=(self.sensor_dir/'INDEX.jsonl').open('x');self.planner=CandidatePlanner();ctx=multiprocessing.get_context('spawn');self.conn,child=ctx.Pipe();self.guard=ctx.Process(target=guardian_process,args=(child,str(self.run/'native_grid'),self.nonce),name='sim-gt-guardian');self.guard.start();child.close();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();self.guard_worker=threading.Thread(target=self.guard_loop,daemon=True);self.guard_worker.start()
 def record(self,event,**fields):
  with self.lock:
   if len(self.events)>=2048:self.audit_overflow=True;self.committed=None;return
   self.events.append(dict(event=event,mode=MODE,monotonic=time.monotonic(),**fields))
 def observe(self,depth,pose,stamp):
  d=np.asarray(depth)
  try:validate_input(d,pose,stamp,self.actual[0]['p'] if self.actual is not None else None)
  except ValueError as ex:
   with self.lock:self.input_fault=True;self.committed=None
   self.record('SIM_INPUT_REJECTED',reason=str(ex),source_stamp=stamp,depth_sha256=hashlib.sha256(d.tobytes()).hexdigest())
   path=self.sensor_dir/('rejected_%d.npy'%int(round(stamp*1e9)))
   if not path.exists():
    with path.open('xb') as f:np.save(f,d,allow_pickle=False)
   raise
  if d.shape!=(90,160) or d.dtype!=np.float32 or pose.get('frame')!='world' or not 0<=self.clock()-stamp<=.2:raise ValueError('SIM_INPUT_CONTRACT')
  with self.lock:
   if stamp<=self.last_stamp:raise ValueError('SIM_DEPTH_ORDER')
   if len(self.pending)>=8:self.evidence_overflow=True;self.committed=None;raise ValueError('SIM_EVIDENCE_OVERFLOW')
   self.last_stamp=stamp;self.pending.append((d.copy(),dict(pose),stamp));self.frame=(d.copy(),dict(pose),stamp);self.revision+=1
 def invalidate_state(self,reason):
  with self.lock:self.input_fault=True;self.actual=None;self.actual_pair=None;self.state_pending=True;self.committed=None;self.proposed=None
  self.record('STATE_INVALID',reason=reason)
 def state(self,state,stamp,imu_stamp,pair=None):
  try:
   now=self.clock();validate_sample('odom',state,stamp,now);validate_sample('imu',dict(frame=state.get('frame'),a_world=state.get('a')),imu_stamp,now)
   if abs(stamp-imu_stamp)>.02:raise StateInvalid('STATE_TIME_MISMATCH')
  except StateInvalid as ex:self.invalidate_state(str(ex));raise
  with self.lock:self.actual=(dict(state),stamp,imu_stamp);self.actual_pair=pair;self.state_pending=False
  self.record('STATE_UPDATED',odom_stamp=stamp,imu_stamp=imu_stamp,state=state,state_pair=pair)
 def set_goal(self,goal,stamp):
  now=self.clock();expected=[3.25,-.25,2.]
  try:actual=list(goal);age=now-stamp
  except (TypeError,ValueError):raise ValueError('SIM_GOAL_FORMAT')
  self.record('GOAL_RECEIVED',goal=actual,expected_goal=expected,source_stamp=stamp,ros_now=now,age=age)
  reason=None
  if actual!=expected:reason='SIM_GOAL_COORDINATES'
  elif not np.isfinite(stamp) or not np.isfinite(now):reason='SIM_GOAL_TIMESTAMP_NONFINITE'
  elif age<0:reason='SIM_GOAL_FUTURE'
  elif age>.2:reason='SIM_GOAL_STALE'
  if reason:
   self.record('GOAL_REJECTED',reason=reason,goal=actual,expected_goal=expected,source_stamp=stamp,ros_now=now,age=age)
   raise ValueError(reason)
  with self.lock:self.goal=(actual,stamp)
 def loop(self):
  while not self.stop.wait(.005):
   with self.lock:frame=self.frame;actual=self.actual;goal=self.goal;pending=list(self.pending)
   if frame is None or actual is None or goal is None:continue
   now=self.clock();depth,pose,source=frame
   try:
    # Lossless fusion/ACK is independent of the proposal frequency.
    for dep,pos,stamp in pending:
     path=self.sensor_dir/('depth_%d.npy'%int(round(stamp*1e9)))
     with path.open('xb') as f:np.save(f,dep,allow_pickle=False)
     self.sensor_index.write(json.dumps(dict(path=str(path),pose=pos,stamp=stamp,sha256=hashlib.sha256(dep.tobytes()).hexdigest()))+'\n');self.sensor_index.flush()
    with self.lock:
     while self.pending and self.pending[0][2]<=source:self.pending.popleft()
    if now<self.next_plan or self.state_pending:continue
    if not 0<=now-source<=.2 or abs(actual[2]-source)>.02:continue
    self.next_plan=now+.2
    matched=dict(pose,a=actual[0]['a']);plan,info=self.planner.propose(depth,pose,matched,goal[0],source,allow_diagnostic=True,prefer_strict=False)
    with self.lock:fresh=self.actual;fresh_pair=self.actual_pair
    if fresh is None or self.input_fault:continue
    # Re-anchor an already Depth-selected goal, without consulting GT.
    plan=build(np.array(fresh[0]['p']),np.array(fresh[0]['v']),np.array(fresh[0]['a']),fresh[0]['q'],np.array(plan.record['goal']),fresh[1])
    with self.lock:self.proposed=dict(plan=plan,planner=info)
    self.record('CANDIDATE_GENERATED',plan_id=plan.id,plan=plan.record,planner=info,state_pair=fresh_pair)
   except Exception as ex:self.record('CANDIDATE_REJECTED',reason=str(ex))
 def guard_loop(self):
  while not self.stop.wait(.02):
   try:
    if not self.guardian_ready:
     if self.conn.poll(0):
      message=self.conn.recv()
      if not message.get('ready'):raise ValueError('GUARDIAN_START_FAILURE')
      self.guardian_ready=True
     else:continue
    with self.lock:proposal=self.proposed;old=self.committed;actual=self.actual;goal=self.goal;source=self.last_stamp;rev=self.revision;executed=self.last_executed;state_pair=self.actual_pair;state_pending=self.state_pending
    if actual is None or goal is None or state_pending or (proposal is None and old is None):continue
    if old is not None and state_pair is not None and old.get('state_pair',{}).get('state_pair_id')==state_pair['state_pair_id']:continue
    now=self.clock()
    if not 0<=now-source<=.2 or not 0<=now-actual[1]<=.2 or not 0<=now-actual[2]<=.2 or not 0<=now-goal[1]<=.2:continue
    if self.audit_overflow or self.evidence_overflow or self.input_fault:continue
    candidate=proposal if proposal is not None else old;plan=candidate['plan'];info=candidate['planner']
    if executed is not None and executed.id!=plan.id:
     before=executed.at(plan.epoch);after=plan.sample(0)
     if any(np.linalg.norm(before[k]-after[k])>tol for k,tol in enumerate([.05,.1,.5])):raise ValueError('SIM_SWITCH_PVA')
    issued=self.clock();self.record('ORACLE_PREFIX_REQUESTED',plan_id=plan.id,issued=issued,source_stamp=source,revision=rev,state_pair=state_pair);self.conn.send(dict(record=plan.record,state=actual[0],now=issued,source_stamp=source,revision=rev))
    if not self.conn.poll(.15):raise ValueError('GUARDIAN_DEADLINE')
    result=self.conn.recv()
    self.record('ORACLE_PREFIX_RETURNED',plan_id=plan.id,issued=issued,ok=result['ok'],reason=result.get('reason'),expires=result.get('proof',{}).get('expires'),oracle=result.get('proof'))
    if not result['ok']:raise ValueError(result['reason'])
    proof=result['proof']
    if proof['plan_id']!=plan.id or proof['nonce']!=self.nonce or proof['source_stamp']!=source or proof['revision']!=rev:raise ValueError('GUARDIAN_REPLY_ASSOCIATION')
    if self.clock()>proof['expires']:raise ValueError('GUARDIAN_REPLY_EXPIRED')
    with self.lock:
     if self.state_pending or self.input_fault or any(not 0<=self.clock()-ts<=.2 for ts in actual[1:]):
      self.record('ORACLE_REPLY_NOT_COMMITTED',plan_id=plan.id,issued=issued,expires=proof['expires'],reason='STATE_PAIR_PENDING' if self.state_pending else ('INPUT_FAULT' if self.input_fault else 'CAPTURED_STATE_STALE'))
      continue
     self.committed=dict(plan=plan,proof=proof,planner=info,state_pair=state_pair)
    self.record('ORACLE_PREFIX_COMMITTED',plan_id=plan.id,oracle=proof,state_pair=state_pair)
   except Exception as ex:
    self.record('SIM_REJECTED',reason=str(ex))
    # Failed proposals never extend an old lease. Renewing the still-approved
    # plan is a separate fresh GT check, not reuse of an expired proof.
    with self.lock:
     if self.committed is not None and self.proposed is not None and self.proposed['plan'].id!=self.committed['plan'].id:self.proposed=self.committed
 def command(self,now):
  with self.lock:
   c=self.committed;actual=self.actual;goal=self.goal
   if c is None or actual is None or goal is None or self.audit_overflow or self.evidence_overflow or self.input_fault:return None
   plan=c['plan'];proof=c['proof']
   for row in proof.get('map_files',[]):
    try:st=Path(row['path']).stat()
    except OSError:return None
    if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:return None
   if not proof['issued']<=now<=proof['expires'] or not 0<=now-self.last_stamp<=.2 or not 0<=now-actual[1]<=.2 or not 0<=now-actual[2]<=.2 or not 0<=now-goal[1]<=.2:return None
   guard_pair=c.get('state_pair')
   if guard_pair is not None and any(not 0<=now-guard_pair[k]<=.2 for k in ['odom_stamp','imu_stamp']):return None
   if np.linalg.norm(plan.at(now)[0]-actual[0]['p'])>.15:return None
   return dict(kind='trajectory',mode=MODE,state_pair=self.actual_pair,guardian_state_pair=c.get('state_pair'),plan=plan.record,plan_id=plan.id,stamp=now,pva=plan.at(now).tolist(),authorization=dict(type='SIM_DEV_ORACLE_PREFIX_BRAKE',geometry='SIM_DEV_ORACLE_PREFIX_BRAKE',plan_id=plan.id,certificate_expiry=proof['expires'],oracle=proof),source_stamp=proof['source_stamp'],map_epoch=proof['revision'],snapshot_id=proof['map_id'],certificate_id=proof['identity'],acceptance_id=proof['identity'])
 def with_current(self,now,action):
  with self.lock:
   command=self.command(now)
   if command is not None:action(command)
   return command
 def published(self,envelope):
  with self.lock:
   if self.committed is None or self.committed['plan'].id!=envelope['plan_id']:raise ValueError('SIM_PUBLICATION_IDENTITY')
   self.last_executed=self.committed['plan'];self.published_ids.add(envelope['plan_id']);self.published_plan_count=len(self.published_ids)
   self.record('SIM_POSITION_COMMAND_PUBLISHED',plan_id=envelope['plan_id'])
 def request_arm(self,nonce):
  if nonce!=self.nonce:raise ValueError('SIM_ARM_NONCE')
  with self.lock:
   if getattr(self,'arm_request_stamp',None) is None:
    self.arm_request_stamp=self.clock();self.arm_sent=False
    self.record('WAIT_FOR_VALID_PLAN_AND_GUARDIAN_PROOF',requested=self.arm_request_stamp)
 def with_arm_ready(self,now,action):
  with self.lock:
   requested=getattr(self,'arm_request_stamp',None)
   if requested is None or getattr(self,'arm_sent',False) or self.state_pending:return None
   command=self.command(now)
   if command is None:return None
   proof=self.committed['proof']
   # The READY decision must use a fresh check issued after the delayed request arrived.
   if proof['issued']<requested:return None
   expected=np.asarray(command['pva']);actual=np.asarray([self.actual[0][k] for k in ['p','v','a']])
   bounds=proof['assumed_pva_error_bounds']
   if any(np.linalg.norm(actual[k]-expected[k])>bounds[k] for k in range(3)):return None
   self.record('READY_TO_ARM',plan_id=command['plan_id'],certificate_id=command['certificate_id'],issued=proof['issued'],expires=proof['expires'],state_pair=self.actual_pair)
   action(command);self.arm_sent=True
   self.record('SIM_ARM_PERMIT_PUBLISHED',plan_id=command['plan_id'],certificate_id=command['certificate_id'],issued=proof['issued'],expires=proof['expires'])
   return command
 def summary(self):
  with self.lock:return dict(ready=self.command(self.clock()) is not None,published_plans=self.published_plan_count,mode=MODE,valid_until=self.committed['proof']['expires'] if self.committed else None)
 def take_events(self):
  with self.lock:rows=list(self.events);self.events.clear();return rows
 def close(self):
  self.stop.set();self.worker.join(1.);self.guard_worker.join(1.)
  if self.worker.is_alive() or self.guard_worker.is_alive():raise RuntimeError('SIM_WORKER_NOT_STOPPED')
  if self.guard.is_alive():self.conn.send(None);self.guard.join(2.)
  if self.guard.is_alive():self.guard.terminate();self.guard.join(2.);raise RuntimeError('SIM_GUARDIAN_FORCED_STOP')
  self.conn.close();self.sensor_index.close()
class SupervisorBridge:
 def __init__(self,run,clock):
  self.engine=Engine(run,clock);self.clock=clock;self.pairs=StatePairs()
 def observe(self,depth,pose,stamp):self.engine.observe(depth,pose,stamp)
 def sensor(self,kind,stamp,data,now):
  try:
   added=self.pairs.add(kind,data,stamp,now)
   if added:self.engine.record('STATE_SENSOR_RECEIVED',kind=kind,source_stamp=stamp,receipt_ros=now,data=data,source='NATIVE_ROS_'+kind.upper())
  except StateInvalid as ex:self.engine.invalidate_state(str(ex));raise
 def state(self,last,stamp,imu_stamp):
  try:pair=self.pairs.select(self.clock())
  except StateInvalid as ex:self.engine.invalidate_state(str(ex));raise
  with self.engine.lock:self.engine.state_pending=pair['status']=='STATE_PAIR_PENDING'
  self.engine.record(pair['status'],**{k:v for k,v in pair.items() if k!='status'})
  if pair['status']=='STATE_PAIR_READY':
   metadata={k:v for k,v in pair.items() if k!='state'}
   self.engine.state(pair['state'],pair['odom_stamp'],pair['imu_stamp'],metadata)
  return pair
 def request(self,envelope,last,now,armed,odom_stamp,imu_stamp):
  if envelope.get('kind')!='forest_goal' or envelope.get('frame')!='world':raise ValueError('SIM_GOAL_REQUEST')
  self.engine.set_goal(envelope['goal'],envelope['stamp'])
  pair=self.state(last,odom_stamp,imu_stamp)
  if pair['status']=='STATE_PAIR_PENDING':return None
  return None if armed else np.array([pair['state']['p'],[0,0,0],[0,0,0]])
 def close(self):self.engine.close()
