"""Detached training labels supervise existing raw19/20. No online certificate."""
import torch
import torch.nn.functional as F
from query_minco.training.io import require

def ranking_loss(raw,labels,quality,progress):
 require(raw.ndim==3 and raw.shape[1:]==(6,21),"PG2_RANK_SHAPE")
 passed=labels["passed"].detach()
 require(passed.shape==raw.shape[:2] and passed.dtype==torch.bool,"PG2_RANK_LABEL")
 clearance=labels["clearance"].detach()
 # Fixed dimensionless label cost; no target gradients to geometry or trajectory.
 cost=(-clearance/.15+.01*quality.detach()-progress.detach()).detach()
 losses=[];targets=[];selected=[]
 for b in range(len(raw)):
  ids=torch.nonzero(passed[b],as_tuple=False).flatten()
  if not len(ids):
   losses.append(raw[b,:,19].sum()*0);targets.append(None);selected.append(None);continue
  best=ids[cost[b,ids].argmin()]
  losses.append(F.cross_entropy(raw[b,:,19][None],best[None]))
  targets.append(int(best));selected.append(int(ids[raw[b,ids,19].detach().argmax()]))
 rank=torch.stack(losses)
 validity=F.binary_cross_entropy_with_logits(raw[:,:,20],passed.to(raw.dtype),reduction="none").mean(-1)
 return rank+validity,dict(rank=rank,validity=validity,targets=targets,
  oracle_restricted_selected=selected,all_failed=(~passed.any(-1)),cost=cost,
  semantics="TRAINING_LABELS_NOT_ONLINE_SAFETY")

def safety_priority_projection(safe_gradient,quality_gradient,enabled=False):
 require(safe_gradient.shape==quality_gradient.shape and bool(torch.isfinite(safe_gradient).all()) and bool(torch.isfinite(quality_gradient).all()),"PG2_PROJECTION")
 dot=(safe_gradient*quality_gradient).sum()
 if not enabled or float(dot)>=0:return quality_gradient,False
 den=safe_gradient.square().sum()
 if float(den)==0:return quality_gradient,False
 return quality_gradient-dot/den*safe_gradient,True
