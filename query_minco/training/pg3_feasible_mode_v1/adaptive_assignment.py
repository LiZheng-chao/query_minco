"""Training-only detached adaptive capacity assignment. No online selector."""
import copy,math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from query_minco.training.io import require
from .assignment import assign
from .bottleneck import bottleneck_loss
def configuration():
 return dict(schema="pg3_m1_adaptive_assignment_v1",alpha=1.,beta=.5,gamma=.5,count_penalty=.1,
  risk_ref_m=.30,distance_ref_m=1.,age_ref=5,max_wait=5,starvation_bonus=100.,
  parameter_distance_fraction=.75,bottleneck_distance_fraction=.25,capacity=2,
  risk_source="64_PER_SEGMENT_SAMPLED_SIGNED_SPHERE_CLEARANCE_NOT_CERTIFICATE",
  training_only=True,reference_source="existing radius+margin, PG3 sigma_W and six-query slot count; no dev fit")
class AdaptiveAssignment:
 def __init__(self,cfg=None,state=None):
  self.cfg=configuration() if cfg is None else copy.deepcopy(cfg);require(self.cfg==configuration(),"PG3_ADAPTIVE_FIXED_CONFIG")
  self.state=copy.deepcopy(state or {});self.pending=[]
 def choose(self,sid,inner,tail,matches,hard_pass,teachers,coeffs,durations,signed,step,capacity):
  require(capacity==self.cfg["capacity"],"PG3_ADAPTIVE_CAPACITY")
  # Existing allocator validates shapes/Hungarian and creates exact reserved occupancy.
  base=assign(inner,tail,matches,hard_pass,teachers,step,capacity)
  matched=base["matched"];bad=[q for q in range(6) if not matched[q] and not bool(hard_pass[q].detach())]
  slots=[m for m,t in enumerate(teachers) if t["certified"] and t["representable"] for _ in range(capacity-sum(m==int(x) for x in matches["teachers"]))]
  state=self.state.get(sid,[dict(count=0,age=0,longest=0,eligible=0) for _ in range(6)])
  scores={};details=[]
  with torch.no_grad():
   require(signed is None or signed.shape[0]==6 and bool(torch.isfinite(signed).all()),"PG3_ADAPTIVE_RISK")
   for q in range(6):
    surface=float(signed[q].min()) if signed is not None else None
    risk=min(1.,max(0.,(.03-surface)/self.cfg["risk_ref_m"])) if surface is not None else 0.
    hist=state[q];age=min(hist["age"]/self.cfg["age_ref"],1.);countfrac=hist["count"]/max(hist["eligible"],1)
    edges=[]
    for m in sorted(set(slots)):
     t=teachers[m]
     dr=torch.cat([(inner[q].detach()-torch.as_tensor(t["inner"]).to(inner)).reshape(-1),(tail[q].detach()-torch.as_tensor(t["tail"][0]).to(inner)).reshape(-1)])
     param=float(torch.sqrt(dr.square().mean()))/self.cfg["distance_ref_m"]
     _,bd=bottleneck_loss(coeffs[q].detach(),durations[q].detach(),t["bottlenecks"])
     bn=float(np.mean(bd["lateral_error_m"])) if bd["count"] else None
     dist=.75*min(param,1.)+.25*min(bn/self.cfg["distance_ref_m"],1.) if bn is not None else min(param,1.)
     score=self.cfg["alpha"]*risk-self.cfg["beta"]*dist+self.cfg["gamma"]*age-self.cfg["count_penalty"]*countfrac
     bonus=self.cfg["starvation_bonus"]+hist["age"] if hist["age"]>=self.cfg["max_wait"] else 0.
     scores[(q,m)]=score+bonus
     edges.append(dict(teacher=m,parameter_rms_m=param*self.cfg["distance_ref_m"],bottleneck_error_m=bn,normalized_distance=dist,risk_term=risk,distance_term=-.5*dist,age_term=.5*age,count_term=-.1*countfrac,priority=score,starvation_bonus=bonus,effective_priority=score+bonus,available_slots=slots.count(m)))
    details.append(dict(sample_id=sid,query=q,matched=matched[q],hard_pass=bool(hard_pass[q]),eligible=q in bad,
     signed_clearance_sampled_m=surface,risk=risk,supervision_count_before=hist["count"],age_before=hist["age"],longest_before=hist["longest"],eligible_before=hist["eligible"],edges=edges,
     inner_waypoints=inner[q].detach().tolist()))
  route=[int(t) if matched[q] else -1 for q,t in enumerate(base["route_teacher_id"])];selected=[];occupancy=[sum(int(x)==m for x in matches["teachers"]) for m in range(len(teachers))]
  reasons=["ORIGINAL_HUNGARIAN" if matched[q] else ("SAFE_UNMATCHED_PRESERVED" if bool(hard_pass[q]) else "NO_CERTIFIED_TEACHER_OR_CAPACITY") for q in range(6)]
  if bad and slots:
   matrix=np.array([[-scores[(q,m)] for m in slots] for q in bad],dtype=float);require(np.isfinite(matrix).all(),"PG3_ADAPTIVE_PRIORITY")
   ii,jj=linear_sum_assignment(matrix)
   for i,j in zip(ii,jj):
    q,m=bad[int(i)],slots[int(j)];route[q]=m;occupancy[m]+=1;selected.append(q);reasons[q]="ADAPTIVE_BAD_UNMATCHED_CAPACITY_SLOT"
   for q in bad:
    if q not in selected:reasons[q]="ADAPTIVE_CAPACITY_DEFERRED"
  require(all(n<=capacity for n in occupancy),"PG3_ADAPTIVE_OVER_CAPACITY")
  for q in range(6):
   details[q].update(selected=q in selected,teacher=route[q],reason=reasons[q])
  out=dict(route_teacher_id=route,selected=selected,matched=matched,occupancy=occupancy,reasons=reasons,teacher_count=len(teachers),assignment_requires_grad=False,adaptive=details)
  # Duplicate Batch rows see the same pre-step state; history advances only after committed optimizer update.
  self.pending.append((sid,out));return out
 def commit(self):
  groups={}
  for sid,row in self.pending:groups.setdefault(sid,[]).append(row)
  for sid,rows in groups.items():
   state=self.state.setdefault(sid,[dict(count=0,age=0,longest=0,eligible=0) for _ in range(6)])
   for q,h in enumerate(state):
    eligible=any(r["adaptive"][q]["eligible"] for r in rows);selected=any(q in r["selected"] for r in rows)
    if selected:h["count"]+=1
    if eligible:
     h["eligible"]+=1;h["age"]=0 if selected else h["age"]+1;h["longest"]=max(h["longest"],h["age"])
    # Safe/matched exposures are not starvation and reset consecutive eligible wait.
    else:h["age"]=0
  self.pending=[]
  return copy.deepcopy(self.state)
