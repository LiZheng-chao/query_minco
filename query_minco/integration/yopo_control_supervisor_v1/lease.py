"""Simulation-only lease supervision. No plant model,GT or network selection."""
from pathlib import Path
import os,signal,time
import numpy as np
COMPONENT_ID='sim_only_dual_lease_latched_hold_and_owned_process_stop_v1'
class Lease:
 def __init__(self,timeout=.2):
  self.timeout=timeout;self.mode='WAIT';self.reason=None;self.last={};self.position=None;self.velocity=None;self.request=None;self.transitions=[]
 def trip(self,reason,now):
  if self.mode!='ABORT':
   self.mode='ABORT';self.reason=reason;self.transitions.append(dict(mode='ABORT',reason=reason,monotonic=now))
 def touch(self,kind,now,stamp,ros_now,values=None):
  if self.mode=='ABORT':return
  if not all(np.isfinite(x) for x in [now,stamp,ros_now]) or ros_now-stamp>self.timeout or stamp-ros_now>.01:return self.trip(kind+'_STAMP',now)
  if values is not None and not np.isfinite(values).all():return self.trip(kind+'_NAN',now)
  if self.mode=='ACTIVE' and kind in self.last and now-self.last[kind][0]>self.timeout:return self.trip(kind+'_TIMEOUT_LATE',now)
  if kind in self.last and stamp<=self.last[kind][1]:return self.trip(kind+'_NONMONOTONIC',now)
  self.last[kind]=(now,stamp)
 def odom(self,now,stamp,ros_now,p,v):
  self.touch('odom',now,stamp,ros_now,np.r_[p,v])
  if self.mode!='ABORT':self.position=np.asarray(p,float);self.velocity=np.asarray(v,float)
  if self.mode!='ABORT' and np.linalg.norm(self.velocity)>1+1e-6:self.trip('MEASURED_SPEED',now)
 def depth(self,now,stamp,ros_now,valid): 
  if not valid:return self.trip('depth_INVALID',now)
  self.touch('depth',now,stamp,ros_now)
 def raw(self,now,stamp,ros_now,values):self.touch('controller',now,stamp,ros_now,values)
 def command(self,now,stamp,ros_now,pva):
  pva=np.asarray(pva,float)
  if pva.shape!=(3,3):return self.trip('request_SHAPE',now)
  previous_stamp=self.last.get('request',(None,None))[1]
  self.touch('request',now,stamp,ros_now,pva)
  if self.mode=='ABORT':return
  if np.linalg.norm(pva[1])>1+1e-6 or np.linalg.norm(pva[2])>8:return self.trip('request_LIMIT',now)
  if self.position is None:return
  if self.mode=='WAIT':
   if any(k not in self.last for k in ['odom','depth','controller']):return
   if np.linalg.norm(pva[0]-self.position)>.05 or np.linalg.norm(pva[1]-self.velocity)>.1:return self.trip('INITIAL_PV',now)
   self.mode='ACTIVE';self.transitions.append(dict(mode='ACTIVE',monotonic=now))
  if self.request is not None and previous_stamp is not None:
   dt=stamp-previous_stamp;expected=self.request.copy();expected[0]+=self.request[1]*dt+.5*self.request[2]*dt*dt;expected[1]+=self.request[2]*dt
   if any(np.linalg.norm(pva[k]-expected[k])>limit for k,limit in enumerate([.05,.1,.5])):return self.trip('SWITCH_DISCONTINUITY',now)
  if np.linalg.norm(pva[0]-self.position)>.25 or np.linalg.norm(self.velocity)>1+1e-6:return self.trip('TRACKING_ENVELOPE',now)
  self.request=pva.copy()
 def tick(self,now):
  if self.mode=='ACTIVE':
   for kind in ['request','odom','depth','controller']:
    if kind not in self.last or now-self.last[kind][0]>self.timeout:
     self.trip(kind+'_TIMEOUT',now);break
  if self.mode=='ABORT':
   # Hold is a measured-PV command replacement,not a physical-braking certificate.
   return np.stack([self.position,np.zeros(3),np.zeros(3)]) if self.position is not None else None
  if self.mode=='WAIT' and self.position is not None:return np.stack([self.position,np.zeros(3),np.zeros(3)])
  return self.request if self.mode=='ACTIVE' else None
 def forward_allowed(self,now):
  self.tick(now)
  return self.mode=='ACTIVE' or (self.mode=='WAIT' and all(k in self.last and now-self.last[k][0]<=self.timeout for k in ['odom','depth','controller']))
class OwnedProcessStop:
 """Authoritative simulation fail-stop for an exclusively owned process group.
 Stopping SO3 publication alone is never treated as braking."""
 def __init__(self,process):
  if process.pid==os.getpid() or os.getpgid(process.pid)!=process.pid:raise ValueError('EXCLUSIVE_PROCESS_GROUP_REQUIRED')
  self.process=process
 def stop(self,reason):
  started=time.monotonic();p=self.process;stopped=False
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGSTOP)
   for _ in range(100):
    stat=Path('/proc/%d/stat'%p.pid)
    if not stat.exists():break
    if stat.read_text().rsplit(')',1)[1].split()[0] in ['T','t']:stopped=True;break
    time.sleep(.001)
   # Terminate the simulation,not a claim of hover or real-aircraft emergency control.
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  code=p.wait(timeout=3)
  return dict(reason=reason,pid=p.pid,stop_state_observed=stopped,actual_exit_code=code,terminal_state='OWNED_PROCESS_TERMINATED',elapsed_s=time.monotonic()-started)

def parent_death_preexec():
 """Bind an owned simulation child to its guardian before exec;Linux only."""
 import ctypes
 expected=os.getpid();libc=ctypes.CDLL(None,use_errno=True)
 def bind():
  if libc.prctl(1,int(signal.SIGKILL),0,0,0)!=0:os._exit(126)
  if os.getppid()!=expected:os._exit(125)
 return bind
