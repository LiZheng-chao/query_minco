"""Extend frozen handshake only at ARMED Request contract;all original freshness stays."""
import numpy as np
from query_minco.integration.yopo_startup_v3.readiness import Readiness as Startup
class Readiness(Startup):
 def validate(self,k,d):
  if k=='request' and self.mode=='ARMED':
   a=np.asarray(d['pva'],float)
   if a.shape!=(3,3) or not np.isfinite(a).all() or np.linalg.norm(a[1])>1+1e-6 or np.linalg.norm(a[2])>8:raise ValueError('REQUEST_LIMIT')
   return
  super().validate(k,d)
