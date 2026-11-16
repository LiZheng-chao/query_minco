"""Finite engineering launch-volume permission,not an unknown-space certificate."""
import json,time
from pathlib import Path
import numpy as np
from query_minco.integration.observed_space_v2.core import sample_batch
def certify(run,plan=None,pva=None,now=None):
 run=Path(run);now=time.time() if now is None else now
 d=json.loads((run/'BIRTH_CERTIFICATE.json').read_text());t=json.loads((run/'truth_data/HEARTBEAT.json').read_text())
 if d['scope']!='ENGINEERING_BIRTH_BALL_ONLY' or d['radius_m']!=.60 or d['center_world']!=[0.,0.,2.]:raise ValueError('BIRTH_SCOPE')
 if not d['issued']<=now<=d['expires'] or d['expires']-d['issued']>30:raise ValueError('BIRTH_EXPIRED')
 if t['status']!='PASS' or time.monotonic()-t['monotonic']>.2 or t['map_id']!=d['map_id']:raise ValueError('BIRTH_TRUTH_INVALID')
 center=np.asarray(d['center_world']);radius=.30
 if plan is not None:
  ts=np.linspace(0,float(plan.d.sum()),129);points=sample_batch(plan,ts)[:,0];radius+=plan.maxima[0]*float(ts[1]-ts[0])/2
 else:
  values=np.asarray(pva,float)
  if values.shape!=(3,3) or not np.isfinite(values).all():raise ValueError('PVA')
  points=values[0][None]
 if np.max(np.linalg.norm(points-center,axis=1))+radius>d['radius_m']:raise ValueError('OUTSIDE_BIRTH_UNKNOWN')
 return True
