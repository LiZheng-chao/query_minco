"""Three-piece physical-time B0 anchor, no current-attitude re-rotation or extra gravity."""
from dataclasses import dataclass
import numpy as np
from query_minco.evaluation.validation import evaluate
@dataclass(frozen=True)
class Reference:
 coefficients:np.ndarray
 durations:np.ndarray
 origin:np.ndarray
 rotation:np.ndarray
 source_stamp:float
 sequence:int
 def __post_init__(self):
  c=np.array(self.coefficients,dtype=float,copy=True);d=np.array(self.durations,dtype=float,copy=True);o=np.array(self.origin,dtype=float,copy=True);R=np.array(self.rotation,dtype=float,copy=True)
  if c.shape!=(3,3,6) or d.shape!=(3,) or o.shape!=(3,) or R.shape!=(3,3) or not all(np.isfinite(x).all() for x in (c,d,o,R)) or np.any(d<=0) or not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or abs(np.linalg.det(R)-1)>1e-7 or not np.isfinite(self.source_stamp) or self.source_stamp<=0 or type(self.sequence)!=int or self.sequence<0:raise ValueError('REFERENCE_ARRAYS_FRAME_TIME')
  for name,v in [('coefficients',c),('durations',d),('origin',o),('rotation',R)]:v.setflags(write=False);object.__setattr__(self,name,v)
 def pva(self,execution_stamp):
  t=execution_stamp-self.source_stamp
  if not np.isfinite(t) or t<0 or t>self.durations.sum():raise ValueError('REFERENCE_EXPIRED_OR_FUTURE')
  ends=self.durations.cumsum();i=min(int(np.searchsorted(ends,t,side='right')),2);local=t-(ends[i-1] if i else 0.)
  return np.stack([self.rotation@evaluate(self.coefficients[i],local,k)+(self.origin if k==0 else 0) for k in range(3)])
 def world_coefficients(self):
  w=np.einsum('ij,pjk->pik',self.rotation,self.coefficients);w[:,:,0]+=self.origin;return w
 def seam_residuals(self):return [max(float(np.linalg.norm(evaluate(self.coefficients[i],self.durations[i],k)-evaluate(self.coefficients[i+1],0,k))) for i in range(2)) for k in range(3)]
 def splice_residuals(self,next_reference,activation):return np.linalg.norm(self.pva(activation)-next_reference.pva(activation),axis=1).tolist()
def controller_semantics(flag):return 'ACCELERATION_ONLY' if flag==1 else 'ORIGINAL_POSITION_VELOCITY_HOVER_FEEDBACK'
