"""Timestamped100Hz source history. No invented zero acceleration or future Odom."""
import dataclasses,threading,collections
import numpy as np
@dataclasses.dataclass(frozen=True)
class SourceState:
 stamp:float
 received_mono:float
 position:np.ndarray
 velocity:np.ndarray
 acceleration:np.ndarray
 rotation:np.ndarray
 derivative_dt:float
class SourceHistory:
 def __init__(self):self.lock=threading.Lock();self.previous=None;self.rows=collections.deque(maxlen=128)
 def add(self,stamp,received_mono,position,velocity,rotation):
  p=np.array(position,float,copy=True);v=np.array(velocity,float,copy=True);R=np.array(rotation,float,copy=True)
  if p.shape!=(3,) or v.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(np.r_[stamp,received_mono,p,v,R.ravel()]).all():raise ValueError('SOURCE_STATE_SHAPE')
  with self.lock:
   past=self.previous
   if past is not None and stamp<=past[0]:self.previous=None;self.rows.clear();raise ValueError('SOURCE_CLOCK_RESET')
   if past is not None and stamp-past[0]<.005:return
   self.previous=(float(stamp),v.copy())
   if past is None or stamp-past[0]>.05:return
   dt=stamp-past[0];a=(v-past[1])/dt # compute from saved previous, NEVER updated v
   for x in (p,v,a,R):x.setflags(write=False)
   self.rows.append(SourceState(float(stamp),float(received_mono),p,v,a,R,dt))
 def source(self,stamp,now_ros,now_mono):
  with self.lock:
   row=next((r for r in reversed(self.rows) if abs(r.stamp-stamp)<1e-7),None)
  if row is None:raise ValueError('NO_CAUSAL_100HZ_SOURCE_STATE')
  if not 0<=now_ros-row.stamp<=.25 or row.received_mono>now_mono:raise ValueError('STALE_OR_FUTURE_SOURCE_STATE')
  return row

from query_minco.inference.dynamic_proposal import Inbox
import struct
class SourceBoundInbox(Inbox):
 def source(self,stamp):
  return self.request(b'I'+struct.pack('<d',stamp))
