"""Detached capacity slots preserve original Hungarian and safe unmatched Queries."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from query_minco.training.io import require
def assign(inner,tail,matches,hard_pass,teachers,step=0,capacity=2,preferred=None):
 require(inner.shape==(6,2,3) and tail.shape==(6,3) and hard_pass.shape==(6,),"PG3_ASSIGN_SHAPE")
 require(capacity>=1 and bool(torch.isfinite(inner).all()) and bool(torch.isfinite(tail).all()),"PG3_ASSIGN_FINITE")
 M=len(teachers);route=[-1]*6;occupancy=[0]*M;matched=[False]*6;reasons=["NOT_SELECTED"]*6
 require(len(matches["queries"])==len(matches["teachers"]),"PG3_MATCH")
 for q,m in zip(matches["queries"],matches["teachers"]):
  q,m=int(q),int(m);require(0<=q<6 and 0<=m<M and not matched[q],"PG3_HUNGARIAN_IDS")
  matched[q]=True;route[q]=m;occupancy[m]+=1;reasons[q]="ORIGINAL_HUNGARIAN"
 require(all(x<=capacity for x in occupancy),"PG3_CAPACITY_HUNGARIAN")
 bad=[q for q in range(6) if not matched[q] and not bool(hard_pass[q].detach())]
 for q in range(6):
  if not matched[q] and bool(hard_pass[q].detach()):reasons[q]="SAFE_UNMATCHED_PRESERVED"
 slots=[m for m,t in enumerate(teachers) if t["representable"] and t["certified"] for _ in range(capacity-occupancy[m])]
 if not M:
  return dict(route_teacher_id=route,selected=[],matched=matched,occupancy=occupancy,reasons=["NO_TEACHER_UNKNOWN"]*6,teacher_count=0,assignment_requires_grad=False)
 if bad:
  offset=int(step)%len(bad);bad=bad[offset:]+bad[:offset]
  if preferred is not None and preferred in bad:bad=[preferred]+[q for q in bad if q!=preferred]
 selected=bad[:len(slots)]
 if selected:
  D=[]
  for q in selected:
   D.append([float((inner[q].detach().double()-torch.as_tensor(teachers[m]["inner"],dtype=torch.float64,device=inner.device)).square().sum()+(tail[q].detach().double()-torch.as_tensor(teachers[m]["tail"][0],dtype=torch.float64,device=inner.device)).square().sum()) for m in slots])
  # Stable lexicographic tie breaking has no trainable derivative.
  matrix=np.asarray(D);matrix=matrix+np.arange(matrix.size).reshape(matrix.shape)*1e-12
  qq,ss=linear_sum_assignment(matrix)
  for i,j in zip(qq,ss):
   q,m=selected[int(i)],slots[int(j)];route[q]=m;occupancy[m]+=1;reasons[q]="BAD_UNMATCHED_CAPACITY_SLOT"
 for q in bad[len(slots):]:reasons[q]="CAPACITY_DEFERRED"
 require(all(x<=capacity for x in occupancy),"PG3_CAPACITY")
 require(all(not matched[q] and not bool(hard_pass[q]) for q in selected),"PG3_UNSAFE_ONLY")
 return dict(route_teacher_id=route,selected=selected,matched=matched,occupancy=occupancy,reasons=reasons,teacher_count=M,assignment_requires_grad=False)
