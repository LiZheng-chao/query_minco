"""Fixed Teacher events, differentiable physical-time evaluation; roots are diagnostics."""
import numpy as np
import torch
import torch.nn.functional as F
from query_minco.training.io import require
from query_minco.evaluation.validation import evaluate,roots

def crossing_events(c,d,x,epsilon=1e-6):
 offset=0.;events=[];coincident=False
 for segment,(cf,T) in enumerate(zip(np.asarray(c,float),np.asarray(d,float))):
  poly=cf[0].copy();poly[0]-=float(x)
  if np.max(abs(poly))<1e-12:coincident=True
  candidates=roots(poly,float(T))
  candidates += [t for t in (0.,float(T)) if abs(float(np.polynomial.polynomial.polyval(t,poly)))<=1e-8]
  for t in candidates:
   point=evaluate(cf,t);speed=float(evaluate(cf,t,1)[0])
   events.append(dict(time_s=offset+float(t),segment=segment,local_time_s=float(t),position=point.tolist(),vx=speed,tangent=abs(speed)<=epsilon))
  offset+=float(T)
 unique=[]
 for e in sorted(events,key=lambda e:(e["time_s"],e["segment"])):
  if not unique or abs(e["time_s"]-unique[-1]["time_s"])>epsilon:unique.append(e)
 forward=[e for e in unique if e["vx"]>epsilon]
 if coincident:return dict(status="COINCIDENT_PLANE_SEGMENT_UNSUPPORTED",events=unique,selected=None)
 status="NO_CROSSING" if not unique else ("TANGENT_ONLY" if not forward else ("MULTIPLE_FORWARD_CROSSINGS" if len(forward)>1 else "UNIQUE_FORWARD_CROSSING"))
 return dict(status=status,events=unique,selected=forward[0] if forward else None)

def at_progress(c,d,s):
 require(c.shape==(3,3,6) and d.shape==(3,) and bool((d>0).all()) and bool(torch.isfinite(c).all()) and bool(torch.isfinite(d).all()),"PG3_PROGRESS_CURVE")
 s=torch.as_tensor(s,dtype=d.dtype,device=d.device).detach()
 require(bool(torch.isfinite(s).all()) and bool(((s>=0)&(s<=1)).all()),"PG3_PROGRESS_RANGE")
 times=s*d.sum();ends=d.cumsum(0)
 segment=torch.searchsorted(ends.detach().contiguous(),times.detach().contiguous(),right=True).clamp_max(2)
 starts=torch.cat([d.new_zeros(1),ends[:-1]])
 local=times-starts[segment]
 powers=local[:,None]**torch.arange(6,device=c.device,dtype=c.dtype)[None]
 return (c[segment]*powers[:,None,:]).sum(-1)

def bottleneck_loss(c,d,labels,sigma_perp=1.,sigma_parallel=1.,beta_parallel=.1):
 zero=(c.sum()+d.sum())*0
 require(sigma_perp>0 and sigma_parallel>0,"PG3_ANCHOR_SCALE")
 valid=[x for x in labels if x["valid"]]
 if not valid:return zero,dict(status="NO_CERTIFIED_BOTTLENECK",count=0)
 s=[x["progress"] for x in valid];prediction=at_progress(c,d,s)
 target=prediction.new_tensor([x["position"] for x in valid]).detach()
 normal=prediction.new_tensor([x["normal"] for x in valid]).detach()
 require(bool(torch.isclose(normal.norm(dim=-1),normal.new_ones(len(normal)),atol=1e-6).all()),"PG3_NORMAL_UNIT")
 delta=prediction-target;parallel=(delta*normal).sum(-1);perp=delta-parallel[:,None]*normal
 loss=F.smooth_l1_loss(perp/sigma_perp,torch.zeros_like(perp),reduction="none").mean(-1)+beta_parallel*F.smooth_l1_loss(parallel/sigma_parallel,torch.zeros_like(parallel),reduction="none")
 return loss.mean(),dict(status="TEACHER_PHYSICAL_TIME_ALIGNED_PROXY",count=len(valid),lateral_error_m=perp.detach().norm(dim=-1).tolist(),axial_error_m=parallel.detach().abs().tolist(),prediction=prediction.detach().tolist(),teacher=target.tolist(),teacher_progress=s,not_spatial_crossing_supervision=True)
