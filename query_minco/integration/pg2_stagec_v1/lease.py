"""Versioned command lease. No ROS Publisher, no timer-only fallback success."""
from dataclasses import dataclass
from enum import Enum
import math
class State(str,Enum):
 DISARMED='DISARMED';SHADOW='SHADOW';READY_NO_AUTHORITY='READY_NO_AUTHORITY';ACTIVE_SIM_ONLY='ACTIVE_SIM_ONLY';REFERENCE_EXPIRED='REFERENCE_EXPIRED';ABORT='ABORT';FALLBACK_REQUESTED='FALLBACK_REQUESTED';FALLBACK_VERIFIED='FALLBACK_VERIFIED'
@dataclass(frozen=True)
class Packet:
 owner:str
 sequence:int
 heartbeat:int
 source_stamp:float
 expires_at:float
 digest:str
 mode:str='MOTION'
@dataclass(frozen=True)
class Gates:
 explicit_sim_authorization:bool=False
 no_hardware:bool=False
 unique_controller:bool=False
 native_selector_query:object=None
 sensor_body_certificate:bool=False
 qualified_controller:bool=False
 qualified_splice:bool=False
 qualified_fallback:bool=False
 certificate_scope:str='UNKNOWN'
class Arbiter:
 def __init__(self,ttl=.1):
  if not 0<ttl<=.25:raise ValueError('LEASE_TTL')
  self.ttl=ttl;self.state=State.DISARMED;self.packet=None;self.last_rx=None;self.last_now=None;self.events=[];self.query_permission=False
 def transition(self,state,reason,now):
  self.events.append(dict(before=self.state.value,after=state.value,reason=reason,now=now));self.state=state;self.query_permission=False
 def clock(self,now):
  if not math.isfinite(now):
   self.packet=None;self.last_rx=None;self.transition(State.ABORT,'CLOCK_NONFINITE',None);raise ValueError('CLOCK_NONFINITE')
  if self.last_now is not None and now<self.last_now:
   self.packet=None;self.last_rx=None;self.transition(State.ABORT,'CLOCK_ROLLBACK_CLEAR_LEASE',now);raise ValueError('CLOCK_ROLLBACK')
  self.last_now=now
 def shadow(self,now):
  self.clock(now)
  if self.state not in (State.DISARMED,State.SHADOW):raise ValueError('EXPLICIT_RESET_REQUIRED')
  self.transition(State.SHADOW,'NO_CONTROL_SHADOW',now)
 def receive(self,p,now):
  self.clock(now)
  if self.state in (State.DISARMED,State.ABORT,State.FALLBACK_REQUESTED,State.FALLBACK_VERIFIED,State.REFERENCE_EXPIRED):raise ValueError('DISARMED_OR_LATCHED')
  if p.mode not in ('MOTION','PERSISTENT_HOVER') or not p.owner or not p.digest or type(p.sequence)!=int or type(p.heartbeat)!=int or p.sequence<0 or p.heartbeat<0:raise ValueError('PACKET_ID')
  if not all(math.isfinite(x) for x in (p.source_stamp,p.expires_at)) or p.source_stamp<=0 or p.source_stamp>now or p.expires_at<now:raise ValueError('REFERENCE_TIME')
  old=self.packet
  if old and p.owner!=old.owner:self.transition(State.ABORT,'OWNER_CONFLICT',now);raise ValueError('OWNER_CONFLICT')
  continuation=old is not None and p.mode==old.mode=='PERSISTENT_HOVER' and p.sequence==old.sequence
  if continuation:
   if p.source_stamp!=old.source_stamp or p.expires_at!=old.expires_at or p.digest!=old.digest or p.heartbeat<=old.heartbeat:raise ValueError('HOVER_IDENTITY_OR_DUPLICATE_HEARTBEAT')
  else:
   if old and p.sequence<=old.sequence:raise ValueError('REFERENCE_SEQUENCE')
   if not 0<=now-p.source_stamp<=self.ttl:raise ValueError('OLD_MOTION_OR_HOVER_INITIAL_SOURCE')
  self.packet=p;self.last_rx=now;self.transition(State.READY_NO_AUTHORITY,'VALID_RECEIPT_NOT_EXECUTION_AUTHORITY',now)
 def tick(self,now):
  self.clock(now)
  if self.state in (State.ABORT,State.FALLBACK_REQUESTED,State.FALLBACK_VERIFIED):return self.state
  if self.packet and (now>self.packet.expires_at or now-self.last_rx>self.ttl):self.transition(State.REFERENCE_EXPIRED,'REFERENCE_OR_RECEIPT_EXPIRED',now)
  return self.state
 def request_authority(self,g,now):
  self.clock(now)
  if self.tick(now)!=State.READY_NO_AUTHORITY or not (g.explicit_sim_authorization and g.no_hardware and g.unique_controller and g.native_selector_query is not None and g.sensor_body_certificate and g.qualified_controller and g.qualified_splice and g.qualified_fallback and g.certificate_scope=='QUALIFIED_SENSOR_SIMULATION'):raise ValueError('AUTHORITY_GATES_BLOCKED')
  self.transition(State.ACTIVE_SIM_ONLY,'ALL_EXTERNAL_QUALIFICATION_GATES',now);self.query_permission=True
 def conflict(self,now):self.clock(now);self.transition(State.ABORT,'CONTROLLER_OR_OWNER_CONFLICT',now)
 def request_fallback(self,now):self.clock(now);self.transition(State.FALLBACK_REQUESTED,'REQUEST_ONLY_NO_PHYSICAL_STOP_PROOF',now)
 def verify_fallback(self,evidence,now):
  self.clock(now)
  raise ValueError('PHYSICAL_EVIDENCE_VERIFIER_NOT_IMPLEMENTED')
 def snapshot(self):return dict(version='command_lease_v1',state=self.state.value,query_permission=self.query_permission,source_stamp=None if self.packet is None else self.packet.source_stamp,last_valid_receipt=self.last_rx,events=self.events)
