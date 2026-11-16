"""Versioned optimization facade;frozen geometry and trajectory mathematics are reused."""
import time,threading,concurrent.futures,weakref
import numpy as np
from query_minco.evaluation.validation import evaluate
from query_minco.integration.observed_space_v1.map import ObservedMap as OldMap,Frame,Snapshot,BudgetExceeded,CertificationError

def sample_batch(plan,times):
 t=np.asarray(times,dtype=float)
 if t.ndim!=1 or not np.isfinite(t).all() or (t<0).any() or (t>plan.d.sum()+1e-7).any():raise ValueError('TIME_CONTRACT')
 k=np.minimum(np.searchsorted(np.cumsum(plan.d),t,side='right'),2)
 local=t-np.r_[0.,np.cumsum(plan.d)[:-1]][k]
 out=np.empty((len(t),3,3))
 for seg in range(3):
  m=k==seg
  for order in range(3):
   out[m,order]=evaluate(plan.c[seg],local[m],order)@plan.R.T
 out[:,0]+=plan.origin
 return out

class Prepared:
 def __init__(self,plan):
  self.plan=plan;self.id=plan.id;self.d=plan.d;self.maxima=plan.maxima
  self.times=np.linspace(0,float(plan.d.sum()),129);self.values=sample_batch(plan,self.times)
  self.values.flags.writeable=False
 def sample(self,t):
  i=int(round(float(t)/float(self.d.sum())*128))
  if i<0 or i>128 or abs(self.times[i]-t)>1e-10:raise ValueError('PREPARED_GRID_MISMATCH')
  return self.values[i]

class ObservedMap(OldMap):
 """Identical carving rules;unchanged content reuses immutable geometric preprocessing."""
 def add(self,depth,pose,stamp,pose_stamp,now,camera_id='native_camera'):
  with self.lock:
   old=self.frames[-1] if self.frames else None
   result=super().add(depth,pose,stamp,pose_stamp,now,camera_id)
   if result and old is not None and self.frames[-1].content_id==old.content_id:
    new=self.frames[-1];new.hit_tree=old.hit_tree;new.hit_bounds=old.hit_bounds
    new.cache=old.cache;new.filters=old.filters
   return result

class AsyncCertificates:
 """Single bounded worker;never grant while pending,on epoch change or after expiry."""
 def __init__(self,ledger,clock=time.time):
  self.ledger=ledger;self.clock=clock;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
  self.lock=threading.Lock();self.future=None;self.proof=None;self.closed=False
 def submit(self,plan):
  with self.lock:
   if self.closed:raise ValueError('CLOSED')
   self.proof=None
   if self.future is not None and not self.future.done():return False
   epoch=self.ledger.epoch
   def work():
    start=time.perf_counter();now=self.clock();snap=self.ledger.snapshot(now)
    if time.perf_counter()-start>.15:raise BudgetExceeded('ASYNC_MAP_DEADLINE')
    begin=time.perf_counter();prepared=Prepared(plan)
    # Preparation is part of the50ms certificate budget.
    if time.perf_counter()-start>.2:raise BudgetExceeded('ASYNC_TOTAL_DEADLINE')
    proof=snap.certify(prepared,self.clock())
    if time.perf_counter()-begin>.05:raise BudgetExceeded('ASYNC_CERT_DEADLINE')
    proof.update(plan_id=plan.id)
    return proof
   self.future=self.pool.submit(work);return True
 def consume(self,plan_id):
  with self.lock:
   if self.closed or self.future is None or not self.future.done():return None
   try:p=self.future.result()
   except Exception:return None
   if p['plan_id']!=plan_id or p['epoch']!=self.ledger.epoch or self.clock()>p['valid_until']:return None
   return dict(p)
 def close(self):
  with self.lock:self.closed=True;self.proof=None
  self.pool.shutdown(wait=True)
