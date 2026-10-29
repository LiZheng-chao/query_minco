"""V2 losses, all Teacher/GT information stays training/offline only."""
import torch
from query_minco.training.pg1.objective import forward as old_forward
from query_minco.training.pg1.geometry import authenticated_task_gates
from query_minco.training.pg1.losses import polynomial_values
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,hard_labels,verify_label_binding
from query_minco.evaluation.goal import task_metrics
from query_minco.training.io import require
from . import losses,ranking
from .spec import FULL,COMPONENTS,FIXED,BASELINE,recipe,weights_for_method

def forward(model,samples,sidecar,method=FULL,context=None,weights=None):
 base=old_forward(model,samples,sidecar,"query_minco_pg1_task_complete_v1")
 if method==BASELINE:return base
 fields,gates=authenticated_task_gates(samples,sidecar)
 return from_base(base,samples,fields,gates,method,context,weights)

def from_base(base,samples,fields,gates,method=FULL,context=None,weights=None):
 cfg=recipe(method);raw=base["raw"];out=base["output"];ana=base["analytic"]
 p,v,w=base["positions"],base["velocities"],base["time_weights"]
 context=context if context is not None else TeacherGuidance()
 route=context.build(samples,fields,base["matches"],p,w)
 safe=losses.safety(p,v,w,fields,route,cfg)
 bounds=raw.new_tensor([f.row["flight_center_domain_m"] for f in fields])
 outside=((p.detach()<bounds[:,0,None,None])|(p.detach()>bounds[:,1,None,None])).any(-1).any(-1)
 blocked=safe["confirmed"]|outside
 goals=torch.stack([s.state[6:9] for s in samples]).to(raw)
 terminal_a=polynomial_values(out.trajectory.coeffs[:,:,-1],out.parameters.durations[:,:,-1,None],2).squeeze(-2)
 enabled=torch.tensor([g["goal_enabled"] for g in gates],dtype=torch.bool,device=raw.device)
 task,dense,gd=losses.goal(ana["endpoint"],ana["terminal_velocity"],terminal_a,goals,enabled,blocked,route)
 feasible,fd=losses.feasibility(ana,bounds,cfg);quality=losses.control(out.trajectory)
 labels=hard_labels(out,samples,fields);verify_label_binding(labels,out,samples,fields)
 goal_labels=[]
 for b,s in enumerate(samples):
  goal_labels.append([task_metrics(ana["endpoint"][b,q].detach().cpu().numpy(),ana["terminal_velocity"][b,q].detach().cpu().numpy(),s.state[6:9].numpy())["passed"] for q in range(6)])
 goal_labels=torch.tensor(goal_labels,dtype=torch.bool,device=raw.device)
 q=quality["terms"]
 quality_cost=q["acceleration"]+q["jerk"]+q["time"]
 utility=base["per_query"]["obstacle"].detach()+base["per_query"]["goal"].detach()+quality_cost.detach()
 rank,rd=ranking.losses(raw,labels["passed"],goal_labels,quality_cost,utility,cfg["ranking"])
 per=dict(clearance=safe["clearance"],retreat=safe["retreat"],corridor=safe["corridor"],braking=safe["braking"],
  original_task=task,dense_guidance=dense,**feasible,**q)
 per.update({k:t[:,None].expand(-1,6) for k,t in rank.items()})
 terms=dict(imitation=base["terms"]["imitation"],**{k:t.mean() for k,t in per.items()})
 require(set(terms)==set(COMPONENTS),"V2_COMPONENT_SET")
 effective=weights_for_method(weights,method) if weights is not None else {k:FIXED.get(k,1.) for k in COMPONENTS}
 total=sum(effective[k]*terms[k] for k in COMPONENTS)
 require(bool(torch.isfinite(total)),"V2_LOSS_FINITE")
 return dict(loss=total,terms=terms,per_query=per,raw=raw,output=out,base=base,fields=fields,samples=samples,
  route=route,safe=safe,goal_diag=gd,feasibility_diag=fd,quality=quality,labels=labels,goal_labels=goal_labels,
  matched=base["matched"],recipe=cfg,weights=effective,ranking_diag=rd)
