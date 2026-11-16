"""One bounded MINCO preparation worker; existing certified reference stays active."""
from concurrent.futures import ThreadPoolExecutor
import time
class Preparation:
 def __init__(self,builder):
  self.builder=builder;self.pool=ThreadPoolExecutor(max_workers=1);self.future=None;self.began=None
 def submit(self,*args):
  if self.future is not None:raise ValueError('F0_PREPARATION_ALREADY_PENDING')
  self.began=time.perf_counter();self.future=self.pool.submit(self.builder,*args)
 def take(self):
  if self.future is None:return None
  if time.perf_counter()-self.began>.15:raise ValueError('PLANNING_DEADLINE')
  if not self.future.done():return None
  plan=self.future.result();self.future=None
  return plan
 def close(self):
  self.pool.shutdown(wait=True)
