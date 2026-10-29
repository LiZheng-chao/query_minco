"""Frozen gradients and a future shared-engine adapter; no optimizer construction."""
import math
import torch
from query_minco.training.io import require
from query_minco.training.pg1_v2_r12.diagnostics import flat,cosine,ratio_guard
from .ranking import safety_priority_projection

def audit_gradients(model,z,groups=True):
 raw=z["raw"];params=list(model.parameters());cfg=z["recipe"]
 rg={};pg={};records={}
 for name,value in z["terms"].items():
  rg[name]=torch.autograd.grad(value,raw,retain_graph=True)[0].detach()
  pg[name]=flat(torch.autograd.grad(value,params,retain_graph=True,allow_unused=True),params)
  require(bool(torch.isfinite(rg[name]).all()) and bool(torch.isfinite(pg[name]).all()),"PG2_GRAD_FINITE")
  if name!="ranking":require(not torch.count_nonzero(rg[name][...,19:21]),"PG2_PHYSICAL_SCORE_LEAK")
 for name in ("clearance","retreat"):
  rg["safe_"+name]=torch.autograd.grad(z["safe"][name].mean(),raw,retain_graph=True)[0].detach()
 rg["corridor"]=torch.autograd.grad(z["corridor"].mean(),raw,retain_graph=True)[0].detach()
 safe=cfg["scales"]["safe"]*pg["safe"]
 quality=cfg["scales"]["control"]*pg["control"]
 projected,applied=safety_priority_projection(safe,quality,cfg["projection"])
 total=sum(cfg["scales"][k]*pg[k] for k in pg)-quality+projected
 physical=sum(cfg["scales"][k]*pg[k] for k in pg if k!="imitation")-quality+projected
 im=float(pg["imitation"].norm());pn=float(physical.norm())
 ratio=pn/im if im else (None if pn==0 else 1e30)
 combined=torch.autograd.grad(z["loss"],raw,retain_graph=True)[0].detach()
 require(bool(torch.isfinite(total).all()) and bool(torch.isfinite(combined).all()),"PG2_TOTAL_GRAD")
 for name in pg:
  strata={}
  if groups:
   for matched in (False,True):
    for passed in (False,True):
     mask=(z["matched"]==matched)&(z["labels"]["passed"]==passed)
     key=("matched" if matched else "unmatched")+"_"+("pass" if passed else "fail")
     g=flat(torch.autograd.grad(raw,params,grad_outputs=rg[name]*mask[...,None],retain_graph=True,allow_unused=True),params)
     strata[key]=dict(queries=int(mask.sum()),parameter_norm=float(g.norm()),raw_norm=float(rg[name][mask].norm()))
  layers={};offset=0
  for n,p in model.named_parameters():
   root=n.split(".")[0];vec=pg[name][offset:offset+p.numel()];offset+=p.numel()
   layers[root]=layers.get(root,0.)+float(vec.square().sum())
  records[name]=dict(parameter_norm=float(pg[name].norm()),weighted_parameter_norm=float(pg[name].norm())*cfg["scales"][name],
   nonzero_query_count=int((rg[name].norm(dim=-1)>0).sum()),raw_norm=float(rg[name].norm()),
   time15_19_norms=[float(rg[name][...,i].norm()) for i in range(15,19)],
   cosine_imitation=cosine(pg[name],pg["imitation"]),cosine_safe=cosine(pg[name],pg["safe"]),
   layer_norms={k:math.sqrt(v) for k,v in layers.items()},strata=strata)
 coefficient,duration=torch.autograd.grad(z["loss"],[z["output"].trajectory.coeffs,z["output"].parameters.durations],retain_graph=True,allow_unused=True)
 require(coefficient is not None and duration is not None and bool(torch.isfinite(coefficient).all()) and bool(torch.isfinite(duration).all()),"PG2_COEFF_DURATION_GRAD")
 return dict(terms=records,total_parameter_norm=float(total.norm()),physical_imitation_ratio=ratio,
  projection_applied=applied,projection_enabled=cfg["projection"],
  quality_safe_cosine=cosine(pg["control"],pg["safe"]),raw19_21_ranking_only=True,
  coefficient_norm=float(coefficient.norm()),duration_norm=float(duration.norm()),
  safety_projection_first_order_dot=float((safe*projected).sum())),rg,pg,total

def prepare_update(model,z,index,streak=0):
 """Callable from existing run_engine update_fn after new scoped authorization.
 This helper never takes an optimizer and never executes a parameter update.
 """
 record,rg,pg,total=audit_gradients(model,z,groups=True)
 physical=total-z["recipe"]["scales"]["imitation"]*pg["imitation"]
 _,new_streak=ratio_guard(float(pg["imitation"].norm()),float(physical.norm()),streak)
 offset=0
 for p in model.parameters():
  p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
 norm=torch.nn.utils.clip_grad_norm_(list(model.parameters()),10.)
 require(bool(torch.isfinite(norm)),"PG2_CLIP_FINITE")
 record.update(completed_optimizer_updates=index+1,ratio_alarm_streak=new_streak,
  global_grad_norm_before_clip=float(norm),optimizer_step_called=False)
 return record
