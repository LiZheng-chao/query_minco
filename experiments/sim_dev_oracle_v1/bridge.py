"""Separate SIM bridge; existing strict Engine and ROS Supervisor stay untouched."""
from pathlib import Path
import json,time,threading,multiprocessing,collections,hashlib
import numpy as np
from experiments.sim_dev_oracle_v1.planner import CandidatePlanner,validate_input
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
  self.lock=threading.RLock();self.stop=threading.Event();self.pending=collections.deque();self.events=collections.deque();self.audit_overflow=False;self.evidence_overflow=False;self.revision=0;self.frame=None;self.actual=None;self.goal=None;self.committed=None;self.published_ids=set();self.published_plan_count=0;self.next_plan=0.;self.last_executed=None;self.last_stamp=0.;self.guardian_ready=False;self.proposed=None;self.input_fault=False
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
 def state(self,state,stamp,imu_stamp):
  if abs(stamp-imu_stamp)>.02 or not 0<=self.clock()-stamp<=.2 or any(not np.isfinite(state[k]).all() for k in ['p','v','a']):raise ValueError('SIM_STATE_CONTRACT')
  with self.lock:self.actual=(dict(state),stamp,imu_stamp)
 def set_goal(self,goal,stamp):
  if list(goal)!=[3.25,-.25,2.] or not 0<=self.clock()-stamp<=.2:raise ValueError('SIM_FIXED_GOAL')
  with self.lock:self.goal=(list(goal),stamp)
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
    if now<self.next_plan:continue
    if not 0<=now-source<=.2 or abs(actual[2]-source)>.02:continue
    self.next_plan=now+.2
    matched=dict(pose,a=actual[0]['a']);plan,info=self.planner.propose(depth,pose,matched,goal[0],source,allow_diagnostic=True,prefer_strict=False)
    with self.lock:fresh=self.actual
    # Re-anchor an already Depth-selected goal, without consulting GT.
    plan=build(np.array(fresh[0]['p']),np.array(fresh[0]['v']),np.array(fresh[0]['a']),fresh[0]['q'],np.array(plan.record['goal']),fresh[1])
    with self.lock:self.proposed=dict(plan=plan,planner=info)
    self.record('CANDIDATE_GENERATED',plan_id=plan.id,plan=plan.record,planner=info)
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
    with self.lock:proposal=self.proposed;old=self.committed;actual=self.actual;goal=self.goal;source=self.last_stamp;rev=self.revision;executed=self.last_executed
    if actual is None or goal is None or (proposal is None and old is None):continue
    now=self.clock()
    if not 0<=now-source<=.2 or not 0<=now-actual[1]<=.2 or not 0<=now-goal[1]<=.2:continue
    if self.audit_overflow or self.evidence_overflow or self.input_fault:continue
    candidate=proposal if proposal is not None else old;plan=candidate['plan'];info=candidate['planner']
    if executed is not None and executed.id!=plan.id:
     before=executed.at(plan.epoch);after=plan.sample(0)
     if any(np.linalg.norm(before[k]-after[k])>tol for k,tol in enumerate([.05,.1,.5])):raise ValueError('SIM_SWITCH_PVA')
    issued=self.clock();self.conn.send(dict(record=plan.record,state=actual[0],now=issued,source_stamp=source,revision=rev))
    if not self.conn.poll(.15):raise ValueError('GUARDIAN_DEADLINE')
    result=self.conn.recv()
    if not result['ok']:raise ValueError(result['reason'])
    proof=result['proof']
    if proof['plan_id']!=plan.id or proof['nonce']!=self.nonce or proof['source_stamp']!=source or proof['revision']!=rev:raise ValueError('GUARDIAN_REPLY_ASSOCIATION')
    if self.clock()>proof['expires']:raise ValueError('GUARDIAN_REPLY_EXPIRED')
    with self.lock:self.committed=dict(plan=plan,proof=proof,planner=info)
    self.record('ORACLE_PREFIX_COMMITTED',plan_id=plan.id,oracle=proof)
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
   if not proof['issued']<=now<=proof['expires'] or not 0<=now-self.last_stamp<=.2 or not 0<=now-actual[1]<=.2 or not 0<=now-goal[1]<=.2:return None
   if np.linalg.norm(plan.at(now)[0]-actual[0]['p'])>.15:return None
   return dict(kind='trajectory',mode=MODE,plan=plan.record,plan_id=plan.id,stamp=now,pva=plan.at(now).tolist(),authorization=dict(type='SIM_DEV_ORACLE_PREFIX_BRAKE',geometry='SIM_DEV_ORACLE_PREFIX_BRAKE',plan_id=plan.id,certificate_expiry=proof['expires'],oracle=proof),source_stamp=proof['source_stamp'],map_epoch=proof['revision'],snapshot_id=proof['map_id'],certificate_id=proof['identity'],acceptance_id=proof['identity'])
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
 def __init__(self,run,clock):self.engine=Engine(run,clock);self.clock=clock
 def observe(self,depth,pose,stamp):self.engine.observe(depth,pose,stamp)
 def state(self,last,stamp,imu_stamp):self.engine.state(dict(last['odom'],a=last['imu']['a_world']),stamp,imu_stamp)
 def request(self,envelope,last,now,armed,odom_stamp,imu_stamp):
  if envelope.get('kind')!='forest_goal' or envelope.get('frame')!='world':raise ValueError('SIM_GOAL_REQUEST')
  self.state(last,odom_stamp,imu_stamp);self.engine.set_goal(envelope['goal'],envelope['stamp'])
  return None if armed else np.array([last['odom']['p'],[0,0,0],[0,0,0]])
 def close(self):self.engine.close()
