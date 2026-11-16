"""Content-based pre-start HOLD handshake. WAIT never forwards any command."""
import math
import numpy as np
COMPONENT_ID='explicit_held_plant_hold_prepare_ack_start_v2'
class Readiness:
 def __init__(self,authorized_prepare=False):
  self.authorized=authorized_prepare;self.mode='WAIT';self.reason=None;self.samples={};self.counts={};self.prepare_stamp=None;self.ack_stamps=set()
 def abort(self,reason):self.mode='ABORT';self.reason=reason
 def observe(self,kind,stamp,now,payload):
  if self.mode=='ABORT':return
  if not math.isfinite(stamp) or stamp<=0 or not math.isfinite(now) or now-stamp>.2 or stamp-now>.01:return self.abort(kind+'_INVALID_STAMP')
  self.samples[kind]=(stamp,payload);self.counts[kind]=self.counts.get(kind,0)+1
 def tick(self,now):
  if self.mode=='ABORT':return self.mode
  if not math.isfinite(now) or now<=0:self.abort('INVALID_CLOCK');return self.mode
  required=['odom','imu','depth','request','raw','plant']
  if any(k not in self.samples or self.counts.get(k,0)<5 for k in required):return self.mode
  if any(now-self.samples[k][0]>.2 for k in required):self.abort('STALE_INPUT');return self.mode
  plant=self.samples['plant'][1];pva=np.asarray(self.samples['request'][1],float)
  if pva.shape!=(3,3) or not np.isfinite(pva).all() or np.linalg.norm(pva[1:])>1e-6:self.abort('NON_HOLD_REQUEST');return self.mode
  pos=np.asarray(self.samples['odom'][1]['p']);vel=np.asarray(self.samples['odom'][1]['v'])
  if pos.shape!=(3,) or vel.shape!=(3,):self.abort('ODOM_SHAPE');return self.mode
  force=np.asarray(self.samples['raw'][1]['force'],float)
  if force.shape!=(3,) or not np.isfinite(force).all():self.abort('RAW_FORCE_NONFINITE_OR_SHAPE');return self.mode
  if not np.isfinite(pos).all() or not np.isfinite(vel).all() or np.linalg.norm(vel)>1+1e-6:self.abort('STATE_OR_SPEED');return self.mode
  if self.mode=='WAIT' and not self.authorized:return self.mode
  if self.mode=='WAIT':
   if plant['phase']!='HELD_BEFORE_INTEGRATION' or plant['integration_steps']!=0:self.abort('PLANT_ALREADY_ADVANCED');return self.mode
   if np.linalg.norm(pva[0]-pos)>1e-4 or np.linalg.norm(vel)>1e-6:self.abort('INITIAL_HOLD_PV');return self.mode
   raw=self.samples['raw'][1];force=np.asarray(raw['force']);hover=plant['mass']*plant['gravity']
   if not np.isfinite(force).all() or hover<=0 or abs(force[2]-hover)>.05*hover or np.linalg.norm(force[:2])>.05*hover:self.abort('INVALID_HOVER_FORCE');return self.mode
   self.mode='PREPARE_HOLD';self.prepare_stamp=now
  if self.mode=='PREPARE_HOLD' and plant['final_command_stamp']>=self.prepare_stamp and plant['command_valid']:
   self.ack_stamps.add(plant['final_command_stamp'])
   if len(self.ack_stamps)>=5:self.mode='PREPARED_HOLD_ACK'
  if self.mode=='PREPARED_HOLD_ACK' and plant['phase']=='INTEGRATING':self.mode='ARMED_HOLD'
  return self.mode
 def forward_allowed(self,now):
  self.tick(now);return self.mode in ['PREPARE_HOLD','PREPARED_HOLD_ACK','ARMED_HOLD']
