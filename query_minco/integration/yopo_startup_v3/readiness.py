"""P1 content/unique-stamp protocol. No dynamics integration or control publication."""
import math
import numpy as np
from .initial_condition import initial_motor_rpm
COMPONENT_ID="g4cp1_unique_content_explicit_prepare_ack_arm_v3"
FRAMES=dict(odom='world',imu='world',depth='camera_link',request='world',raw='/quadrotor',plant='world')
class Readiness:
 def __init__(self,prepare_authorized=False,arm_authorized=False,motor_initial_state='zero'):
  self.motor_initial_state=motor_initial_state
  self.prepare_authorized=prepare_authorized;self.arm_authorized=arm_authorized
  self.mode='HELD';self.reason=None;self.samples={};self.counts={};self.duplicates={}
  self.sent=set();self.acks=set();self.last_sent=0.;self.transitions=[];self.last_steps=0
 def abort(self,reason):
  if self.mode!='ABORT':self.reason=str(reason);self.transition('ABORT')
 def transition(self,mode):
  self.transitions.append(dict(before=self.mode,after=mode));self.mode=mode
 def observe(self,kind,stamp,now,data):
  if self.mode=='ABORT':return False
  try:
   if kind not in FRAMES:raise ValueError('UNKNOWN_STREAM')
   if not all(math.isfinite(x) and x>0 for x in [stamp,now]) or now-stamp>.2 or stamp-now>.01:raise ValueError('STAMP')
   if data.get('frame')!=FRAMES[kind]:raise ValueError('FRAME')
   previous=self.samples.get(kind)
   if previous:
    if stamp<previous[0]:raise ValueError('BACKWARD_STAMP')
    if stamp==previous[0]:
     self.duplicates[kind]=self.duplicates.get(kind,0)+1;return False
    if self.mode!='HELD' and (stamp-previous[0]>.2 or now-previous[1]>.2):raise ValueError('GAP')
   self.validate(kind,data)
   self.samples[kind]=(stamp,now,data);self.counts[kind]=self.counts.get(kind,0)+1
   return True
  except (ValueError,TypeError,KeyError,IndexError) as e:self.abort(kind+'_'+str(e));return False
 def validate(self,k,d):
  def vec(v,n):
   a=np.asarray(v,float)
   if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('SHAPE_OR_NONFINITE')
   return a
  if k=='odom':
   p=vec(d['p'],3);v=vec(d['v'],3);q=vec(d['q'],4)
   if abs(np.linalg.norm(q)-1)>1e-3 or np.linalg.norm(v)>1+1e-6:raise ValueError('STATE_LIMIT')
  elif k=='imu':vec(d['a_world'],3)
  elif k=='depth':
   if type(d['valid_count']) is not int or not 0<d['valid_count']<=14400 or d['source_odom_stamp']!=d['stamp']:raise ValueError('NO_VALID_DEPTH_OR_UNBOUND_POSE')
  elif k=='request':
   a=np.asarray(d['pva'],float)
   if a.shape!=(3,3) or not np.isfinite(a).all() or np.linalg.norm(a[1:])>1e-6:raise ValueError('NON_HOLD_REQUEST')
  elif k=='raw':
   f=vec(d['force_world'],3);q=vec(d['q'],4);vec(d['kR'],3);vec(d['kOm'],3)
   if abs(np.linalg.norm(q)-1)>1e-3 or not d['motors_enabled'] or f[2]<=0:raise ValueError('SO3_CONTRACT')
  elif k=='plant':
   if d['phase'] not in ['HELD','PREPARE','ACK','ARMED','ABORT']:raise ValueError('PHASE')
   if type(d['integration_steps']) is not int or d['integration_steps']<self.last_steps:raise ValueError('STEPS')
   if not math.isclose(d['mass'],.98,abs_tol=1e-12) or not math.isclose(d['gravity'],9.81,abs_tol=1e-12):raise ValueError('PHYSICS_IDENTITY')
   if d['integration_steps']>0 and not(self.mode=='ARMED' or (self.mode=='ACK' and self.arm_authorized and d['phase']=='ARMED')):raise ValueError('UNAUTHORIZED_INTEGRATION')
   previous=self.samples.get('plant')
   if previous:
    allowed={'HELD':['HELD','PREPARE'],'PREPARE':['PREPARE','ACK'],'ACK':['ACK','ARMED'],'ARMED':['ARMED'],'ABORT':['ABORT']}
    if d['phase'] not in allowed[previous[2]['phase']]+['ABORT']:raise ValueError('PLANT_PHASE_SKIP')
   if d['motor_initial_state']!=self.motor_initial_state or not math.isclose(d['kf'],8.98132e-9,rel_tol=1e-12) or not math.isclose(d['motor_tau_s'],1/30,rel_tol=1e-12):raise ValueError('MOTOR_IDENTITY')
   if d['integration_steps']==0:
    expected=initial_motor_rpm(self.motor_initial_state,d['mass'],d['gravity'],d['kf'])
    rpm=vec(d['rpm_actual'],4)
    if np.max(np.abs(rpm-expected))>1e-6:raise ValueError('INITIAL_RPM')
   self.last_steps=d['integration_steps']
 def forwarded(self,stamp):
  if not math.isfinite(stamp) or self.mode not in ['PREPARE','ACK','ARMED'] or stamp<=self.last_sent:return self.abort('FINAL_COMMAND_SEQUENCE')
  self.sent.add(stamp);self.last_sent=stamp
 def tick(self,now):
  if self.mode=='ABORT':return self.mode
  if not math.isfinite(now) or now<=0:self.abort('CLOCK');return self.mode
  for k,(stamp,receipt,data) in self.samples.items():
   if now-stamp>.2 or now-receipt>.2 or stamp-now>.01:self.abort(k+'_STALE');return self.mode
  if not all(self.counts.get(k,0)>=5 for k in FRAMES):return self.mode
  p=self.samples['plant'][2];phase=p['phase'];steps=p['integration_steps']
  if phase=='ABORT':self.abort('PLANT_ABORT');return self.mode
  if self.mode in ['HELD','PREPARE','ACK'] and (steps!=0 or phase=='ARMED') and not(self.mode=='ACK' and self.arm_authorized and phase=='ARMED' and steps>0):
   self.abort('UNAUTHORIZED_OR_SKIPPED_INTEGRATION');return self.mode
  od=self.samples['odom'][2];raw=self.samples['raw'][2];req=np.asarray(self.samples['request'][2]['pva'])
  if self.mode=='HELD':
   if phase!='HELD':self.abort('SKIPPED_PREPARE');return self.mode
   if not self.prepare_authorized:return self.mode
   hover=p['mass']*p['gravity'];f=np.asarray(raw['force_world'])
   if np.linalg.norm(req[0]-od['p'])>1e-4 or np.linalg.norm(od['v'])>1e-6 or abs(f[2]-hover)>.05*hover or np.linalg.norm(f[:2])>.05*hover:
    self.abort('INITIAL_HOLD_FORCE_OR_PV');return self.mode
   if abs(float(np.dot(raw['q'],od['q'])))<math.cos(.05/2):self.abort('HOLD_ATTITUDE');return self.mode
   self.transition('PREPARE');return self.mode
  if self.mode=='PREPARE':
   if phase not in ['HELD','PREPARE','ACK']:self.abort('PREPARE_PHASE');return self.mode
   stamp=p['final_command_stamp']
   if p['command_valid'] and stamp>0:
    if stamp not in self.sent:self.abort('ACK_WITHOUT_SENT_FINAL');return self.mode
    self.acks.add(stamp)
   if phase=='ACK' and p['final_count']>=5 and len(self.acks)>=5:self.transition('ACK')
   return self.mode
  if self.mode=='ACK':
   if phase=='ARMED' and self.arm_authorized and steps>0:self.transition('ARMED')
   elif phase!='ACK':self.abort('ACK_PHASE_OR_AUTHORITY')
   return self.mode
  if self.mode=='ARMED' and (phase!='ARMED' or steps<=0):self.abort('ARMED_PHASE')
  return self.mode
 def forward_allowed(self,now):
  self.tick(now)
  return self.mode in ['PREPARE','ACK','ARMED'] and self.samples['plant'][2]['phase'] in ['PREPARE','ACK','ARMED']
