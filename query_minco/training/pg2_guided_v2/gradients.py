"""Independent gradients and deterministic training-set calibration, no optimizer."""
import math
import numpy as np
import torch
from query_minco.training.io import require
from query_minco.training.pg1_v2_r12.diagnostics import flat,cosine
from query_minco.training.pg2_guided_v1.ranking import safety_priority_projection
from .spec import COMPONENTS,FIXED,BUDGETS

def measure(model,z,groups=True):
 params=list(model.parameters());raw=z["raw"];pg={};rg={};summary={}
 for k in COMPONENTS:
  value=z["terms"][k]
  rg[k]=torch.autograd.grad(value,raw,retain_graph=True,allow_unused=False)[0].detach()
  pg[k]=flat(torch.autograd.grad(value,params,retain_graph=True,allow_unused=True),params)
  require(bool(torch.isfinite(rg[k]).all()) and bool(torch.isfinite(pg[k]).all()),"V2_GRAD_FINITE_"+k)
  if k not in ("ranking","validity"):require(not torch.count_nonzero(rg[k][...,19:21]),"V2_SCORE_LEAK")
 layers={}
 for k,g in pg.items():
  d={};i=0
  for n,p in model.named_parameters():
   part=g[i:i+p.numel()];i+=p.numel();root=n.split(".")[0]
   d[root]=d.get(root,0.)+float(part.square().sum())
  layers[k]={n:math.sqrt(v) for n,v in d.items()}
 for k in COMPONENTS:
  strata={}
  if groups:
   for matched in (True,False):
    for passed in (True,False):
     mask=(z["matched"]==matched)&(z["labels"]["passed"]==passed)
     key=("matched" if matched else "unmatched")+"_"+("pass" if passed else "fail")
     if bool(mask.any()):
      g=flat(torch.autograd.grad(raw,params,grad_outputs=rg[k]*mask[...,None],retain_graph=True,allow_unused=True),params)
      norm=float(g.norm())
     else:norm=0.
     strata[key]=dict(count=int(mask.sum()),parameter_norm=norm,raw_norm=float(rg[k][mask].norm()))
  summary[k]=dict(value=float(z["terms"][k].detach()),parameter_norm=float(pg[k].norm()),
   raw_norm=float(rg[k].norm()),time15_19=[float(rg[k][...,j].norm()) for j in range(15,19)],
   strata=strata,layers=layers[k])
 stack=torch.stack([pg[k] for k in COMPONENTS]).double();gram=stack@stack.T
 # Raw mathematical total coefficient/duration routing (weights will be calibrated later).
 gc,gd=torch.autograd.grad(z["loss"],[z["output"].trajectory.coeffs,z["output"].parameters.durations],retain_graph=True,allow_unused=True)
 require(gc is not None and gd is not None and bool(torch.isfinite(gc).all()) and bool(torch.isfinite(gd).all()),"V2_COEFF_DURATION")
 return dict(components=summary,gram=gram.tolist(),coefficient_norm=float(gc.norm()),duration_norm=float(gd.norm())),pg,rg

def calibrate(rows):
 # Predeclared target budgets, training set only. No development-set search.
 n=COMPONENTS.index("imitation")
 norms=np.array([[r["components"][k]["parameter_norm"] for k in COMPONENTS] for r in rows])
 require(np.isfinite(norms).all() and (norms[:,n]>0).all(),"V2_CALIBRATION_IM")
 ratios=norms/norms[:,n,None];weights=dict(FIXED);inactive=[]
 for k,budget in BUDGETS.items():
  j=COMPONENTS.index(k);positive=ratios[:,j][ratios[:,j]>1e-12]
  if not len(positive):weights[k]=0.;inactive.append(k)
  else:weights[k]=float(budget/np.median(positive))
 upper=np.array([sum(weights[k]*r["components"][k]["parameter_norm"]/r["components"]["imitation"]["parameter_norm"] for k in BUDGETS) for r in rows])
 factor=min(1.,2./max(float(upper.max()),1e-30))
 for k in BUDGETS:weights[k]*=factor
 require(set(weights)==set(COMPONENTS),"V2_CALIBRATION_COMPONENTS")
 results=[budget_record(r["gram"],weights) for r in rows]
 return dict(schema="pg2_v2_training_set_calibration_v1",weights=weights,budgets=BUDGETS,global_scale=factor,
  inactive_components=inactive,method="median_norm_ratio_then_fixed_sum_budget2_max_on_train",
  uses_development=False,rows=len(rows),budget_records=results,
  gradient_guard_passed=guard_pass([r["ratio"] for r in results]),optimizer_steps=0)

def guard_pass(ratios):
 streak=0
 for r in ratios:
  streak=streak+1 if r is not None and r>10 else 0
  if streak>=2:return False
 return True

def budget_record(gram,weights):
 g=np.asarray(gram,dtype=np.float64);w=np.array([weights[k] for k in COMPONENTS]);im=COMPONENTS.index("imitation")
 phys=w.copy();phys[im]=0.
 n=lambda a:math.sqrt(max(0.,float(a@g@a)))
 ratio=n(phys)/math.sqrt(max(float(g[im,im]),1e-30))
 safe=np.array([weights[k] if k in ("clearance","retreat","corridor","braking") else 0. for k in COMPONENTS])
 quality=np.array([weights[k] if k in ("acceleration","jerk","time") else 0. for k in COMPONENTS])
 denominator=n(safe)*n(quality);cos=float(safe@g@quality)/denominator if denominator else None
 return dict(ratio=ratio,total_norm=n(w),safe_norm=n(safe),quality_norm=n(quality),safe_quality_cosine=cos)

def total_gradient(pg,weights,projection=False):
 total=sum(weights[k]*pg[k] for k in COMPONENTS)
 safe=sum(weights[k]*pg[k] for k in ("clearance","retreat","corridor","braking"))
 quality=sum(weights[k]*pg[k] for k in ("acceleration","jerk","time"))
 projected,applied=safety_priority_projection(safe,quality,projection)
 return total-quality+projected,dict(projection_applied=applied,safe_quality_cosine=cosine(safe,quality),
  projected_quality_norm=float(projected.norm()),safe_projected_dot=float(safe.dot(projected)))
