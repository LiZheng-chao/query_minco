"""Independent PG2 objective built on unchanged PG1 network/MINCO/Hungarian."""
import torch
from query_minco.training.pg1.objective import forward as pg1_forward
from query_minco.training.pg1.geometry import authenticated_task_gates
from query_minco.training.pg1.losses import polynomial_values
from query_minco.training.io import require
from .spec import FULL,recipe,validate_recipe
from .geometry import TeacherGuidance,hard_labels,verify_label_binding
from .losses import safety_loss,corridor_loss,guidance_loss,feasibility_loss,control_loss
from .ranking import ranking_loss

def forward(model,samples,sidecar,method=FULL,context=None,cfg=None):
 cfg=validate_recipe(recipe(method) if cfg is None else cfg)
 require(cfg["method_id"]==method,"PG2_CONFIG_ID")
 # This call preserves the original forward and matching arithmetic verbatim.
 base=pg1_forward(model,samples,sidecar,"query_minco_pg1_task_complete_v1")
 fields,gates=authenticated_task_gates(samples,sidecar)
 raw=base["raw"];out=base["output"];ana=base["analytic"]
 p,v,w=base["positions"],base["velocities"],base["time_weights"]
 guidance=context if context is not None else TeacherGuidance()
 route=guidance.build(samples,fields,base["matches"],p,w)
 safe=safety_loss(p,v,w,fields,cfg)
 corridor=corridor_loss(p,w,route,cfg["position_reference_m"])
 safe_total=cfg["clearance_scale"]*safe["clearance"]+cfg["retreat_scale"]*safe["retreat"]+cfg["corridor_scale"]*corridor
 goals=torch.stack([s.state[6:9] for s in samples]).to(raw)
 terminal_a=polynomial_values(out.trajectory.coeffs[:,:,-1],out.parameters.durations[:,:,-1,None],2).squeeze(-2)
 enabled=torch.tensor([g["goal_enabled"] for g in gates],device=raw.device,dtype=torch.bool)
 goal,goal_diag=guidance_loss(ana["endpoint"],ana["terminal_velocity"],terminal_a,goals,enabled,
  safe["severe"],route["target"][:,:,-1])
 bounds=raw.new_tensor([f.row["flight_center_domain_m"] for f in fields])
 gravity=torch.stack([s.state[9:12] for s in samples]).to(raw)*9.81
 feasible,feas_diag=feasibility_loss(ana,out.trajectory,goals,bounds,cfg,gravity)
 control,quality=control_loss(out.trajectory,cfg)
 labels=hard_labels(out,samples,fields);verify_label_binding(labels,out,samples,fields)
 progress=(ana["endpoint"]*(goals/goals.norm(dim=-1,keepdim=True))[:,None]).sum(-1)
 rank,rank_diag=ranking_loss(raw,labels,control,progress)
 rank=cfg["ranking_scale"]*rank_diag["rank"]+cfg["validity_scale"]*rank_diag["validity"]
 per=dict(safe=safe_total,guidance=goal,control=control,feasibility=feasible,ranking=rank[:,None].expand(-1,6))
 terms=dict(imitation=base["terms"]["imitation"],**{k:t.mean() for k,t in per.items()})
 total=raw.sum()*0
 for k,vv in terms.items():
  if cfg["scales"][k]:total=total+cfg["scales"][k]*vv
 require(bool(torch.isfinite(total)),"PG2_TOTAL_FINITE")
 return dict(loss=total,terms=terms,per_query=per,raw=raw,output=out,base=base,matched=base["matched"],
  matches=base["matches"],recipe=cfg,labels=labels,route=route,safe=safe,
  corridor=corridor,guidance_diag=goal_diag,feasibility_diag=feas_diag,quality=quality,ranking_diag=rank_diag)
