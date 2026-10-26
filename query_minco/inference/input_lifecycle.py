"""Bounded latest-input lifecycle. Exactly one terminal, no ROS/control methods.
Processing completion is not a future activation/safety qualification.
"""
import collections,copy,dataclasses,threading,time
@dataclasses.dataclass(frozen=True)
class InputTicket:
 sequence:int
 source_stamp_ns:int
 received_mono_ns:int
class SessionStopped(ValueError):pass
class FaultLatch:
 def __init__(self):self._lock=threading.Lock();self._fault=None
 @property
 def stopped(self):
  with self._lock:return self._fault is not None
 @property
 def reason(self):
  with self._lock:return self._fault['reason'] if self._fault else ''
 def trip(self,reason,now_ns=None):
  with self._lock:
   if self._fault is None:self._fault=dict(reason=str(reason),mono_ns=time.monotonic_ns() if now_ns is None else int(now_ns))
 def require_live(self):
  if self.stopped:raise SessionStopped(self.reason)
class InputLifecycle:
 TERMINALS=frozenset(('COMPLETED','REJECTED','ABORTED','DROPPED'))
 def __init__(self):self._lock=threading.Lock();self._next=0;self._open={};self._terminal={}
 def receive(self,stamp_ns,received_ns=None):
  with self._lock:
   self._next+=1;t=InputTicket(self._next,int(stamp_ns),time.monotonic_ns() if received_ns is None else int(received_ns))
   self._open[t.sequence]=dict(ticket=t,started=None);return t
 def begin(self,t,start_ns=None):
  now=time.monotonic_ns() if start_ns is None else int(start_ns)
  with self._lock:
   entry=self._open.get(t.sequence)
   if entry is None or entry['ticket']!=t:raise ValueError('INPUT_NOT_OPEN')
   if entry['started'] is not None:raise ValueError('INPUT_ALREADY_STARTED')
   if now<t.received_mono_ns:
    self._end(t,'ABORTED','CLOCK_ORDER',now,{});return False
   entry['started']=now;return True
 def _end(self,t,status,reason,now,details):
  entry=self._open.get(t.sequence)
  if entry is None or entry['ticket']!=t:raise ValueError('DUPLICATE_OR_UNKNOWN_TERMINAL')
  if status not in self.TERMINALS:raise ValueError('TERMINAL_STATUS')
  snapshot=copy.deepcopy(details)
  del self._open[t.sequence]
  valid=now>=t.received_mono_ns and (entry['started'] is None or now>=entry['started'])
  self._terminal[t.sequence]=dict(input_id=t.sequence,source_stamp_ns=t.source_stamp_ns,received_mono_ns=t.received_mono_ns,processing_start_mono_ns=entry['started'],terminal_mono_ns=now,clock_order_valid=valid,status=status if valid else 'ABORTED',reason=str(reason) if valid else 'CLOCK_ORDER',details=snapshot)
 def finish(self,t,status,reason='',now_ns=None,**details):
  with self._lock:self._end(t,status,reason,time.monotonic_ns() if now_ns is None else int(now_ns),details)
 def close(self,reason='COLLECTION_CLOSED',now_ns=None):
  now=time.monotonic_ns() if now_ns is None else int(now_ns)
  with self._lock:
   for entry in list(self._open.values()):self._end(entry['ticket'],'ABORTED',reason,now,{})
 def snapshot(self):
  with self._lock:return dict(received=self._next,open=len(self._open),terminals=copy.deepcopy(list(self._terminal.values())))
class LatestInputMailbox:
 def __init__(self,lifecycle,session):
  self.life=lifecycle;self.session=session;self._lock=threading.Lock();self._slot=None;self._closed=False;self._last_source=None;self.received=0;self.taken=0;self.replaced=0
 def receive(self,payload,stamp_ns,now_ns=None):
  now=time.monotonic_ns() if now_ns is None else int(now_ns)
  with self._lock:
   t=self.life.receive(stamp_ns,now);self.received+=1
   if self._closed or self.session.stopped:
    self.life.finish(t,'ABORTED',self.session.reason or 'COLLECTION_CLOSED',now);return t
   if self._last_source is not None and stamp_ns<=self._last_source:
    self.life.finish(t,'REJECTED','DUPLICATE_OR_OLD_SOURCE',now);return t
   self._last_source=stamp_ns
   if self._slot is not None:
    old,_=self._slot;self.life.finish(old,'DROPPED','QUEUE_REPLACED',now);self.replaced+=1
   self._slot=(t,payload);return t
 def take(self,now_ns=None):
  with self._lock:
   if self.session.stopped or self._closed or self._slot is None:return None
   ticket,payload=self._slot;self._slot=None
   if not self.life.begin(ticket,now_ns):return None
   self.taken+=1;return ticket,payload
 def close(self,reason='COLLECTION_CLOSED',now_ns=None):
  with self._lock:
   self._closed=True;self._slot=None;self.life.close(self.session.reason or reason,now_ns)
