"""Differentiable physical-coordinate parameter supervision and conditional quality."""
import torch
import torch.nn.functional as F
from query_minco.training.io import require
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss
from .bottleneck import at_progress

def smooth(error,scale=1.):
 require(scale>0 and bool(torch.isfinite(error).all()),"PG3_SMOOTH")
 return F.smooth_l1_loss(error/scale,torch.zeros_like(error),reduction="none",beta=1.).mean(-1)

def mode_query(inner,tail,d,target,cfg):
 W=inner.new_tensor(target["inner"]).detach();PVA=tail.new_tensor(target["tail"]).detach()
 require(target["certified"] and target["representable"],"PG3_DIRECT_TARGET_UNTRUSTED")
 waypoint=smooth(inner-W,cfg["sigma_W"]).sum()
 result=cfg["alpha_W"]*waypoint+cfg["alpha_P"]*smooth(tail[0]-PVA[0],cfg["sigma_P"])+cfg["alpha_V"]*smooth(tail[1]-PVA[1],cfg["sigma_V"])
 if cfg["alpha_A"]:
  require(target["representability"]["acceleration"],"PG3_ACCEL_TARGET_RANGE")
  result=result+cfg["alpha_A"]*smooth(tail[2]-PVA[2],cfg["sigma_A"])
 if cfg["alpha_duration"]:
  require(target["representability"]["duration"],"PG3_DURATION_TARGET_RANGE")
  result=result+cfg["alpha_duration"]*smooth(d-d.new_tensor(target["durations"]).detach(),cfg["sigma_duration"])
 return result,waypoint/2

def conditional_control(trajectory,hard,goal,signed,cfg):
 require(hard.dtype==goal.dtype==torch.bool and hard.shape==goal.shape==trajectory.durations.shape[:2],"PG3_QUALITY_LABEL")
 quality=dynamic_quality_loss(trajectory)
 energies=quality["integrals_si"]
 require(0<=cfg["quality_gamma"]<1 and cfg["quality_tau"]>0,"PG3_QUALITY_GATE")
 soft=(~hard).to(trajectory.coeffs)*cfg["quality_gamma"]*torch.sigmoid((signed.detach()-cfg["quality_c_gate"])/cfg["quality_tau"])
 gate=(goal.to(trajectory.coeffs)*(hard.to(trajectory.coeffs)+soft)).detach()
 denom=gate.sum().clamp_min(1)
 qa=energies["acceleration"]/(cfg["a_ref"]**2*cfg["T_ref"])
 qj=energies["jerk"]/(cfg["j_ref"]**2*cfg["T_ref"])
 qt=trajectory.durations.sum(-1)/cfg["T_ref"]
 # Separate lateral quality term; not silently folded into the stated QA/QJ/QT formula.
 from query_minco.training.pg1.losses import analytic_samples
 ana=analytic_samples(trajectory,64)
 lat=torch.cross(ana["v"],ana["a"],dim=-1).square().sum(-1)/(ana["v"].square().sum(-1)+.1**2)
 lat=(lat/cfg["lateral_ref"]**2*ana["weights"]).sum(-1)/ana["weights"].sum(-1)
 per=dict(qa=qa,qj=qj,qt=qt,lateral=lat)
 return {k:(v*gate).sum()/denom for k,v in per.items()},dict(gate=gate,per_query=per,energies=energies,active=int((gate>0).sum()),label_semantics="OFFLINE_TRAINING_HARD_PASS_AND_GOAL_NOT_ONLINE_CERTIFICATE")

def diversity(trajectory,assignments,banks,fraction=.5):
 values=[];records=[]
 for b,assignment in enumerate(assignments):
  active=[q for q in assignment["selected"]]
  for i,q in enumerate(active):
   for r in active[i+1:]:
    m,n=assignment["route_teacher_id"][q],assignment["route_teacher_id"][r]
    if m==n:continue
    tm,tn=banks[b]["teachers"][m],banks[b]["teachers"][n]
    pairs=[(x,y) for x in tm["bottlenecks"] for y in tn["bottlenecks"] if x["box"]==y["box"] and x["side"] is not None and y["side"] is not None and x["side"]!=y["side"]]
    if not pairs:continue
    x,y=pairs[0];c=trajectory.coeffs[b];d=trajectory.durations[b]
    a=at_progress(c[q],d[q],[x["progress"]])[0];z=at_progress(c[r],d[r],[y["progress"]])[0]
    normal=a.new_tensor(x["normal"]).detach();delta=a-z;perp=delta-(delta*normal).sum()*normal
    target=(a.new_tensor(x["position"])-a.new_tensor(y["position"])).detach();target=target-(target*normal).sum()*normal
    separation=fraction*target.norm();values.append((separation-perp.norm()).clamp_min(0).square())
    records.append(dict(batch=b,q=q,r=r,teacher_pair=[m,n],source="DISTINCT_CERTIFIED_SECTION_FACES_NOT_HOMOTOPY",target_separation_m=float(separation)))
 zero=(trajectory.coeffs.sum()+trajectory.durations.sum())*0
 return torch.stack(values).mean() if values else zero,records
