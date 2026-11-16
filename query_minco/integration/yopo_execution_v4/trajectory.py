"""One authoritative MINCO/sampler/extrema path,shared by reference and execution gate."""
import hashlib,json,time
import numpy as np,torch
from query_minco.integration.yopo_shadow_v1.sensor_contract import rotation,sample_world,observation
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.evaluation.validation import numerical_reference,evaluate,norm_max
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
class Plan:
 def __init__(self,record):
  self.record=record;self.c=np.asarray(record['coefficients'],float);self.d=np.asarray(record['durations'],float)
  self.origin=np.asarray(record['origin'],float);self.R=np.asarray(record['rotation'],float);self.epoch=float(record['epoch'])
  if self.c.shape!=(3,3,6) or self.d.shape!=(3,) or not np.isfinite(self.c).all() or not np.isfinite(self.d).all() or not np.allclose(self.d,2/3,atol=1e-7,rtol=0) or not np.isfinite(self.epoch):raise ValueError('PLAN_SHAPE_TIME')
  self.maxima=[max(norm_max(c,t,k) for c,t in zip(self.c,self.d)) for k in [1,2,3]]
  if any(v>limit+1e-7 for v,limit in zip(self.maxima,[1.,8.,100.])):raise ValueError('CONTINUOUS_EXECUTION_LIMIT')
  # independent original reference solve plus all C4 endpoint/continuity derivatives
  head=np.asarray(record['head_pva']);tail=np.asarray(record['tail_pva']);inner=np.asarray(record['inner_points'])
  ref=numerical_reference(head,tail,inner,self.d);errors=[]
  for order,tol in zip(range(5),[.002,.01,.05,.25,2.]):
   err=max(float(np.abs(evaluate(a,np.linspace(0,t,65),order)-evaluate(b,np.linspace(0,t,65),order)).max()) for a,b,t in zip(self.c,ref,self.d));errors.append(err)
   if err>tol:raise ValueError('INDEPENDENT_MINCO_NUMERIC')
   if order<3:
    if np.max(np.abs(evaluate(self.c[0],0,order)-head[order]))>tol or np.max(np.abs(evaluate(self.c[-1],self.d[-1],order)-tail[order]))>tol:raise ValueError('BOUNDARY')
   for i in range(2):
    if np.max(np.abs(evaluate(self.c[i],self.d[i],order)-evaluate(self.c[i+1],0,order)))>tol:raise ValueError('C4')
  self.errors=errors;self.id=digest(record);self.sample(0)
 def sample(self,t):return sample_world(self.c,self.d,self.origin,self.R,float(np.clip(t,0,self.d.sum())))
 def at(self,stamp):return self.sample(stamp-self.epoch)
def build(p,v,a,q,goal,epoch):
 # Geometry-free state-goal reference only;no GT or optimization.
 R=rotation(q);p=np.asarray(p,float);goal=np.asarray(goal,float)
 # State/goal boundary interpolation constructs physical knots only;MINCO remains the sole trajectory solver.
 T=2.;vB=R.T@v;aB=R.T@a;gB=R.T@(goal-p)
 knots=[]
 for u in [1/3,2/3]:
  knots.append((10*u**3-15*u**4+6*u**5)*gB+(u-6*u**3+8*u**4-3*u**5)*T*vB+(.5*u*u-1.5*u**3+1.5*u**4-.5*u**5)*T*T*aB)
 head=np.stack([np.zeros(3),vB,aB]);tail=np.stack([gB,np.zeros(3),np.zeros(3)]);inner=np.stack(knots)
 make=lambda x:torch.tensor(x,dtype=torch.float32)[None,None]
 with torch.no_grad():tr=MincoAdapter(num_candidates=1)(make(head),make(tail),make(inner),torch.full((1,1,3),2/3))
 rec=dict(coefficients=tr.coeffs[0,0].tolist(),durations=tr.durations[0,0].tolist(),origin=p.tolist(),rotation=R.tolist(),epoch=float(epoch),
  head_pva=head.tolist(),tail_pva=tail.tolist(),inner_points=inner.tolist(),T=2.,solver='original MincoAdapter',goal=goal.tolist())
 return Plan(rec)
class Admission:
 def __init__(self):self.plan=None
 def check(self,envelope,state,now,certify=None,prepared=None):
  pva=np.asarray(envelope['pva'],float)
  if pva.shape!=(3,3) or not np.isfinite(pva).all() or np.linalg.norm(pva[1])>1+1e-6 or np.linalg.norm(pva[2])>8+1e-6:raise ValueError('REQUEST_PVA')
  committed=self.plan
  if envelope['kind']=='hold':
   if np.linalg.norm(pva[1:])>1e-6:raise ValueError('FALSE_HOLD')
  else:
   rec=envelope['plan'];identity=digest(rec)
   if self.plan is None or self.plan.id!=identity:
    if prepared is not None and (not isinstance(prepared,Plan) or prepared.id!=identity):raise ValueError("PREPARED_PLAN_IDENTITY")
    new=Plan(rec) if prepared is None else prepared
    if not 0<=now-new.epoch<=.2:raise ValueError('PLAN_AGE')
    actual=np.stack([state['p'],state['v'],state['a']])
    if any(np.linalg.norm(new.sample(0)[k]-actual[k])>tol for k,tol in enumerate([.05,.1,.5])):raise ValueError('INITIAL_PVA')
    if self.plan is not None:
     old=self.plan.at(new.epoch);start=new.sample(0)
     if any(np.linalg.norm(start[k]-old[k])>tol for k,tol in enumerate([.05,.1,.5])):raise ValueError('SWITCH_DISCONTINUITY')
    if certify is not None:certify(new)
    committed=new
   if envelope['stamp']>committed.epoch+float(committed.d.sum())+.5:raise ValueError('PLAN_HORIZON_EXPIRED')
   if np.max(np.abs(committed.at(envelope['stamp'])-pva))>2e-5:raise ValueError('PLAN_COMMAND_RACE')
  if np.linalg.norm(pva[0]-state['p'])>.25:raise ValueError('TRACKING_ENVELOPE')
  self.plan=committed
  return pva
