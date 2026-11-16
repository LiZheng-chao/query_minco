"""Sensor-only bounded certification pipeline. Diagnostic permissions are not ARM authority."""
import threading,time,copy,hashlib,json
import numpy as np
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared
from query_minco.integration.yopo_execution_v4.trajectory import Plan,digest
class Pipeline:
 def __init__(self,clock=time.time):
  self.clock=clock;self.lock=threading.RLock();self.wake=threading.Event();self.ledger=ObservedMap()
  self.generation=0;self.last_stamp=0.;self.last_content=None;self.last_rejection=None;self.pending=None;self.plan=None;self.proof=None;self.closed=False;self.fault=None;self.dropped=0;self.processed=0;self.last_latency=None
  self.worker=threading.Thread(target=self._run,daemon=True);self.worker.start()
 def fail(self,reason):
  with self.lock:self.generation+=1;self.proof=None;self.fault=str(reason);self.pending=None
 def ingest(self,depth,pose,stamp,pose_stamp):
  now=self.clock()
  d=np.asarray(depth)
  if d.dtype!=np.float32 or d.shape!=(90,160) or not np.isfinite([stamp,pose_stamp,now]).all() or stamp!=pose_stamp or not 0<=now-stamp<=.2:
   self.fail('INPUT_CONTRACT');raise ValueError('INPUT_CONTRACT')
  content=hashlib.sha256(d.tobytes()+json.dumps(pose,sort_keys=True).encode()).hexdigest()
  with self.lock:
   if self.closed or self.fault:return False
   if stamp<self.last_stamp:self.fail('SOURCE_ROLLBACK');return False
   if stamp==self.last_stamp:
    if content!=self.last_content:self.fail('DUPLICATE_CONTENT_CONFLICT')
    return False
   if self.last_stamp and stamp-self.last_stamp>.2:self.fail('SOURCE_GAP');return False
   self.last_stamp=stamp;self.last_content=content;self.generation+=1;self.proof=None
   if self.pending is not None:self.dropped+=1
   self.pending=(d.copy(),copy.deepcopy(pose),stamp,pose_stamp,self.generation)
   self.wake.set();return True
 def request(self,record):
  ident=digest(record)
  with self.lock:
   if self.closed or self.fault:return False
   if self.plan is None or self.plan[0]!=ident:
    self.generation+=1;self.proof=None;self.plan=(ident,copy.deepcopy(record));self.wake.set()
   return True
 def _run(self):
  while True:
   self.wake.wait(.05);self.wake.clear()
   with self.lock:
    if self.closed:return
    item=self.pending;self.pending=None;plan=self.plan;generation=self.generation
    if self.fault:continue
   if item is None:continue
   begin=time.perf_counter()
   try:
    d,pose,stamp,ps,_=item;self.ledger.add(d,pose,stamp,ps,self.clock())
    snap=self.ledger.snapshot(self.clock())
    if time.perf_counter()-begin>.15:raise ValueError('MAP_DEADLINE')
    proof=None
    if plan:
     start=time.perf_counter();p=Plan(plan[1]);prepared=Prepared(p)
     proof=snap.certify(prepared,self.clock())
     if time.perf_counter()-start>.05:raise ValueError('CERT_DEADLINE')
     proof.update(plan_id=plan[0],generation=generation)
    with self.lock:
     self.last_rejection=None
     if not self.closed and self.generation==generation and not self.fault:self.proof=proof
   except Exception as e:
    with self.lock:
     self.proof=None
     # UNKNOWN is a normal rejection;deadline/source failures are latched.
     self.last_rejection=str(e)
     if 'DEADLINE' in str(e) or 'STAMP' in str(e):self.fault=str(e)
   finally:
    with self.lock:self.processed+=1;self.last_latency=time.perf_counter()-begin
 def consume(self,plan_id):
  with self.lock:
   p=self.proof
   if self.closed or self.fault or p is None or p['generation']!=self.generation or p['plan_id']!=plan_id or self.clock()>p['valid_until']:return None
   return copy.deepcopy(p)
 def status(self):
  with self.lock:return dict(generation=self.generation,processed=self.processed,dropped=self.dropped,pending=self.pending is not None,fault=self.fault,rejection=self.last_rejection,last_latency_s=self.last_latency,closed=self.closed,motion_authority=False)
 def close(self):
  with self.lock:self.closed=True;self.proof=None
  self.wake.set();self.worker.join(2)
  if self.worker.is_alive():raise RuntimeError('WORKER_SHUTDOWN_TIMEOUT')
