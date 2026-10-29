"""Explicit structural zeros, parameter budgets and safety/quality cosine."""
import math
import torch
from query_minco.training.io import require
from query_minco.training.pg1_v2_r12.diagnostics import flat,cosine
from .spec import EXTRA

def audit(model,z,retain=True,details=True):
 pp=list(model.parameters());raw=z["raw"];c=z["output"].trajectory.coeffs;d=z["output"].parameters.durations
 terms=dict(imitation=z["base"]["base"]["terms"]["imitation"],base_safe=z["base"]["loss"],**z["terms"])
 vectors={};records={}
 for name,value in terms.items():
  g=torch.autograd.grad(value,pp,retain_graph=True,allow_unused=True);vector=flat(g,pp)
  rg=torch.autograd.grad(value,raw,retain_graph=True,allow_unused=True)[0]
  cg,dg=torch.autograd.grad(value,(c,d),retain_graph=True,allow_unused=True)
  require(bool(torch.isfinite(vector).all()),"PG3_PARAM_GRAD_FINITE")
  if rg is not None:
   require(bool(torch.isfinite(rg).all()) and not torch.count_nonzero(rg[...,19:21]),"PG3_RAW_SCORE_LEAK")
  layers={};offset=0
  for n,p in model.named_parameters():
   gg=vector[offset:offset+p.numel()];offset+=p.numel();layer=n.split(".")[0]
   layers[layer]=layers.get(layer,0.)+float(gg.square().sum())
  strata={}
  if details and rg is not None:
   for matched in (True,False):
    for passed in (True,False):
     mask=(z["matched"]==matched)&(z["labels"]["passed"]==passed)
     sg=flat(torch.autograd.grad(raw,pp,grad_outputs=rg*mask[...,None],retain_graph=True,allow_unused=True),pp)
     strata[("matched" if matched else "unmatched")+"_"+("pass" if passed else "fail")]=dict(count=int(mask.sum()),raw_norm=float(rg[mask].norm()),parameter_norm=float(sg.norm()))
  records[name]=dict(value=float(value.detach()),parameter_norm=float(vector.norm()),raw_norm=float(rg.norm()) if rg is not None else None,
   raw_time_norms=[float(rg[...,j].norm()) for j in range(15,19)] if rg is not None else None,
   coefficient_norm=float(cg.norm()) if cg is not None else None,duration_norm=float(dg.norm()) if dg is not None else None,
   raw_per_query_norm=rg.norm(dim=-1).tolist() if rg is not None else None,raw_waypoint_components=rg[...,:6].tolist() if rg is not None else None,coefficient_structural_none=cg is None,layers={k:math.sqrt(v) for k,v in layers.items()},strata=strata)
  vectors[name]=vector
 im=vectors["imitation"];total=flat(torch.autograd.grad(z["loss"],pp,retain_graph=retain,allow_unused=True),pp)
 physical=total-im;ratio=float(physical.norm())/float(im.norm()) if float(im.norm()) else (0. if not float(physical.norm()) else 1e30)
 safe=vectors["base_safe"]-im+sum(z["weights"][k]*vectors[k] for k in ("mode","bottleneck","convex_corridor","diversity"))
 quality=sum(z["weights"][k]*vectors[k] for k in ("qa","qj","qt","lateral"))
 return dict(components=records,physical_imitation_ratio=ratio,safety_quality_cosine=cosine(safe,quality),physical_imitation_cosine=cosine(physical,im),weighted_component_norms={k:z["weights"][k]*float(vectors[k].norm()) for k in EXTRA}),vectors,total

def calibrate(records):
 import numpy as np
 from .spec import EXTRA,BUDGETS
 weights={}
 for k in EXTRA:
  ratios=[r["components"][k]["parameter_norm"]/r["components"]["imitation"]["parameter_norm"] for r in records if r["components"][k]["parameter_norm"]>1e-12]
  weights[k]=float(BUDGETS[k]/np.median(ratios)) if ratios else 0.
 upper=[sum(weights[k]*r["components"][k]["parameter_norm"]/r["components"]["imitation"]["parameter_norm"] for k in EXTRA) for r in records]
 factor=min(1.,2./max(max(upper),1e-30))
 return dict(weights={k:v*factor for k,v in weights.items()},initial_budgets=BUDGETS,factor=factor,uses_development=False,rows=len(records),formula="train-only median component/Imitation ratio; new terms sum upper budget<=2; old PG2 weights unchanged")
