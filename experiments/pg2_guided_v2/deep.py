"""Frozen Rtail replay; fixed raw perturbations, NEVER optimizer updates."""
import os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from query_minco.training.run_spec import ROOT
from query_minco.training.io import read,artifact,sha,require
from query_minco.training.pg2_guided_v1.audit import write,resource_gate
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.pilot_evaluation import load_rows
from query_minco.training.objective import forward as network
from query_minco.training.pg1.losses import analytic_samples,goal_task_loss
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,check_curve
from query_minco.training.pg2_guided_v1.losses import safety_loss as old_safe,corridor_loss
from query_minco.training.pg2_guided_v1.spec import recipe as old_recipe
from query_minco.training.pg2_guided_v2.objective import from_base
from query_minco.training.pg2_guided_v2.losses import safety,control
from query_minco.training.pg2_guided_v2.spec import recipe
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.evaluation.signed_geometry import fine_geometry
from query_minco.evaluation.validation import evaluate,sample_curve
from query_minco.evaluation.goal import task_metrics

def run(cpu0,output):
 os.environ["CUDA_VISIBLE_DEVICES"]="";torch.set_num_threads(1)
 cpu0=Path(cpu0).resolve();output=Path(output).resolve()
 require(output.parent==ROOT/"research_outputs/pg2_guided_v2/audit","V2_DEEP_OUTPUT")
 output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 cal=read(cpu0/"CALIBRATION.json");bindings=read(cpu0/"audit/BINDINGS.json");weights=cal["weights"]
 origin=read(ROOT/"research_outputs/pg2_guided_v1/runs/A42_C1_PG2Frozen384_CPU0_run001/summary/DEEP_QUERY_REPLAY.json")
 cp=origin["checkpoint"];require(sha(cp["path"])==cp["sha256"],"V2_DEEP_CHECKPOINT")
 model=QueryMincoPlanner();blob=torch.load(cp["path"],map_location="cpu",weights_only=True);model.load_state_dict(blob["state_dict"],strict=True);model.eval()
 before=state_sha(model);require(before==cp["weights_sha256"],"V2_DEEP_WEIGHTS")
 selected={(r["sample_id"],r["query"]) for r in origin["rows"] if r["geometry"] and r["geometry"]["minimum_signed_surface_clearance_m"]<-.120001}
 require(len(selected)==14,"V2_DEEP14")
 context=TeacherGuidance();rows=[];directions=[]
 for sample,meta in load_rows(bindings["development"]):
  qs=[q for q in range(6) if (sample.sample_id,q) in selected]
  if not qs:continue
  resource_gate(started,300)
  scene=meta["scene"];f=SimpleNamespace(row=dict(obstacles=scene["obstacles"],flight_center_domain_m=scene["flight_center_domain_m"],empty_certified=not scene["obstacles"]),
   boxes=torch.tensor([[b["min"],b["max"]] for b in scene["obstacles"]],dtype=torch.float32).reshape(-1,2,3))
  im,matches,raw,out,costs=network(model,[sample],"B_set",torch.device("cpu"))
  p,v,w=segment_samples(out.trajectory,64);ana=analytic_samples(out.trajectory,64)
  oldgoal,_=goal_task_loss(ana["endpoint"],ana["terminal_velocity"],sample.state[6:9][None])
  obstacle,_=physical_loss(p,v,w,[f.boxes],geometry_bound=[True],radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.)
  mask=torch.zeros(1,6,dtype=torch.bool);mask[0,matches[0]["queries"]]=True
  base=dict(raw=raw,output=out,matches=matches,positions=p,velocities=v,time_weights=w,analytic=ana,matched=mask,
   terms=dict(imitation=im),per_query=dict(obstacle=obstacle,goal=oldgoal))
  z=from_base(base,[sample],[f],[dict(goal_enabled=True)],context=context,weights=weights)
  route=z["route"]
  def phi(r):
   oo=TrajectoryPostprocessor()(r,sample.state[None]);pp,vv,ww=segment_samples(oo.trajectory,64)
   ss=safety(pp,vv,ww,[f],route,recipe());old=old_safe(pp,vv,ww,[f],old_recipe())
   q=control(oo.trajectory)["terms"]
   return dict(v1_safe=old["clearance"]+old["retreat"]+corridor_loss(pp,ww,route),
    v2_safe=sum(weights[k]*ss[k] for k in ("clearance","retreat","corridor","braking")),
    v2_corridor=ss["corridor"],QA=q["acceleration"],QJ=q["jerk"],QT=q["time"]),oo
  terms,_=phi(raw)
  for query in qs:
   geo=fine_geometry(out.trajectory.coeffs[0,query].detach().numpy(),out.parameters.durations[0,query].detach().numpy(),scene,points=2049)
   require(not bool(mask[0,query]),"V2_DEEP_UNMATCHED")
   gs={k:torch.autograd.grad(t[0,query],raw,retain_graph=True)[0].detach() for k,t in terms.items()}
   dot=float(gs["v1_safe"].flatten().dot(gs["v2_safe"].flatten()));den=float(gs["v1_safe"].norm()*gs["v2_safe"].norm())
   rows.append(dict(id=sample.sample_id,query=query,matched=False,geometry=geo,teacher=z["route"]["assignments"][0][query],
    v1_v2_raw_gradient_cosine=dot/den if den else None,gradient_norms={k:float(g.norm()) for k,g in gs.items()},
    status=z["safe"]["records"][0][query],original_goal=float(z["goal_diag"]["original_unmodified"][0,query])))
   for name,g in gs.items():
    if not float(g.norm()):directions.append(dict(id=sample.sample_id,query=query,component=name,status="ZERO_GRADIENT"));continue
    perturbed=raw.detach()-.001*g/g.norm()
    try:
     newer,newout=phi(perturbed);cc=newout.trajectory.coeffs[0,query].detach().numpy();dd=newout.parameters.durations[0,query].detach().numpy()
     hard=check_curve(cc,dd,scene,newout.parameters.head_pva[0,query].detach().numpy())
     geom=fine_geometry(cc,dd,scene,points=2049)
     goal=task_metrics(evaluate(cc[-1],dd[-1]),evaluate(cc[-1],dd[-1],1),sample.state[6:9].numpy())
     qq=control(newout.trajectory)
     directions.append(dict(id=sample.sample_id,query=query,component=name,status="OFFLINE_RAW_DIRECTION",
      proxy_before=float(terms[name][0,query]),proxy_after=float(newer[name][0,query]),raw_step_l2=.001,
      clearance_before=geo["minimum_signed_surface_clearance_m"],clearance_after=geom["minimum_signed_surface_clearance_m"],
      hard_pass=hard["passed"],hard_reasons=hard["reasons"],goal=goal,time=float(dd.sum()),
      vmax=hard["max_speed_mps"],amax=hard["max_acceleration_mps2"],jmax=hard["max_jerk_mps3"],
      acceleration_energy=float(qq["integrals_si"]["acceleration"][0,query]),jerk_energy=float(qq["integrals_si"]["jerk"][0,query])))
    except Exception as exc:directions.append(dict(id=sample.sample_id,query=query,component=name,status="REJECTED",error=repr(exc)))
 require(state_sha(model)==before,"V2_DEEP_WEIGHT_MUTATION")
 write(output/"DEEP_RESULTS.json",dict(mode="OFFLINE_DIAGNOSTIC_NOT_TRAINING",checkpoint=cp,calibration_sha256=sha(cpu0/"CALIBRATION.json"),
  weights_before=before,weights_after=state_sha(model),rows=rows,directions=directions,optimizer_created=False,optimizer_steps=0,wall_s=time.monotonic()-started))
 print("V2_DEEP_DONE",len(rows),len(directions),flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--cpu0",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();run(a.cpu0,a.output)
