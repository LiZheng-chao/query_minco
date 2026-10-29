"""Read-only common adapters for PG3 experiments; no historical method modifications."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.io import read,artifact
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss
from query_minco.training.pg1.losses import analytic_samples,goal_task_loss
from query_minco.training.pg2_guided_v2.objective import from_base
from query_minco.training.pg2_guided_v2.spec import PREFIX as OLD
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,check_curve,scene_from_field
from query_minco.training.pg3_feasible_mode_v1.objective import augment
from query_minco.training.pg3_feasible_mode_v1.bottleneck import crossing_events
from query_minco.training.pilot_evaluation import _certified_empty
from query_minco.evaluation.signed_geometry import fine_geometry
from query_minco.evaluation.validation import evaluate,norm_max
from query_minco.evaluation.goal import task_metrics

ROOT=Path("/home/lizheng/query_minco_mainline")
OUT=ROOT/"research_outputs/pg3_feasible_mode_v1"
def model_from(init):
 model=QueryMincoPlanner();model.load_state_dict(torch.load(artifact(init["checkpoint"]),map_location="cpu",weights_only=True)["state_dict"],strict=True);return model
def dev_field(meta):
 scene=meta["scene"]
 return SimpleNamespace(row=dict(obstacles=scene["obstacles"],flight_center_domain_m=scene["flight_center_domain_m"],empty_certified=_certified_empty(scene)),boxes=torch.tensor([[b["min"],b["max"]] for b in scene["obstacles"]],dtype=torch.float32).reshape(-1,2,3))
def raw_base(raw,sample,field,weights,context=None):
 out=TrajectoryPostprocessor()(raw,sample.state[None])
 cost,_=cost_matrix(out,sample,0);im,qq,tt=match(cost,"B_set")
 matches=[dict(sample_id=sample.sample_id,queries=qq.tolist(),teachers=tt.tolist())]
 p,v,w=segment_samples(out.trajectory,64);ana=analytic_samples(out.trajectory,64)
 goal,_=goal_task_loss(ana["endpoint"],ana["terminal_velocity"],sample.state[6:9][None])
 obs,_=physical_loss(p,v,w,[field.boxes],geometry_bound=[True],radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
 mask=torch.zeros(1,6,dtype=torch.bool);mask[0,qq]=True
 base=dict(raw=raw,output=out,matches=matches,positions=p,velocities=v,time_weights=w,analytic=ana,matched=mask,terms=dict(imitation=im),per_query=dict(goal=goal,obstacle=obs))
 return from_base(base,[sample],[field],[dict(goal_enabled=True)],OLD+"safe_v1",context or TeacherGuidance(),weights)
def query_metrics(output,sample,field,q):
 cc=output.trajectory.coeffs[0,q].detach().double().numpy();dd=output.parameters.durations[0,q].detach().double().numpy()
 hard=check_curve(cc,dd,scene_from_field(field),output.parameters.head_pva[0,q].detach().numpy())
 scene=scene_from_field(field);g=fine_geometry(cc,dd,scene,points=4097) if scene["obstacles"] else None
 goal=task_metrics(evaluate(cc[-1],dd[-1]),evaluate(cc[-1],dd[-1],1),sample.state[6:9].numpy())
 labels=[]
 for box in scene["obstacles"]:
  x=(box["min"][0]+box["max"][0])/2;cross=crossing_events(cc,dd,x)
  if cross["selected"]:labels.append(dict(box=box.get("name"),x=x,position=cross["selected"]["position"],status=cross["status"]))
 from experiments.pg1_v2_r1.cpu0_readonly import numpy_integral
 energy={k:float(numpy_integral(cc[None,None].repeat(6,1),dd[None,None].repeat(6,1),order)[0,0]) for k,order in (("qa",2),("qj",3))}
 return dict(hard_pass=hard["passed"],hard_reasons=hard["reasons"],goal=goal,signed=g,acceleration_energy=energy["qa"],jerk_energy=energy["qj"],time_s=float(sum(dd)),vmax=hard["max_speed_mps"],amax=hard["max_acceleration_mps2"],jmax=hard["max_jerk_mps3"],crossings=labels)
