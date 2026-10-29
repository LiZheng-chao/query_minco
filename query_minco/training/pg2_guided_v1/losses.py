"""Differentiable training proxies. None of these functions certify safety."""
import math
import torch
import torch.nn.functional as F
from query_minco.training.io import require
from query_minco.training.pg1.losses import finite,dynamic_limit_loss,center_domain_loss
from query_minco.training.pg1_v2_r1.losses import weighted_lse,dynamic_quality_loss
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.evaluation.goal_config import CONFIG
from .spec import validate_recipe

def sdf_normal(p,boxes):
 with torch.enable_grad():
  x=p.detach().requires_grad_(True)
  n=torch.autograd.grad(box_sdf(x,boxes).sum(),x)[0]
 return n.detach()

def safety_loss(p,v,w,fields,cfg):
 validate_recipe(cfg)
 finite(p,"PG2_P");finite(v,"PG2_V");finite(w,"PG2_MASS")
 require(p.ndim==4 and p.shape[1]==6 and p.shape[-1]==3 and v.shape==p.shape and w.shape==p.shape[:-1],"PG2_SAMPLE_SHAPE")
 require(bool((w>=0).all()) and bool((w.sum(-1)>0).all()) and len(fields)==len(p),"PG2_MASS_OR_GEOMETRY")
 require(all(x.dtype==p.dtype and x.device==p.device for x in [v,w]),"PG2_SAMPLE_TYPE")
 alpha=w/w.sum(-1,keepdim=True)
 losses=[];retreat=[];risks=[];diagnostics=[];severe=[];first_indices=[];target_rows=[]
 for b,f in enumerate(fields):
  boxes=f.boxes.to(p)
  require(type(f.row["empty_certified"]) is bool,"PG2_EMPTY_CERT_TYPE")
  if not len(boxes):
   require(f.row["empty_certified"] is True,"PG2_EMPTY_UNTRUSTED")
   z=p[b,...,0]*0;losses.append(z.sum(-1));retreat.append(z.sum(-1));risks.append(z)
   severe.append(torch.zeros(6,dtype=torch.bool,device=p.device));first_indices.append([-1]*6)
   target_rows.append(torch.zeros_like(p[b,:,0]));diagnostics.append(["CERTIFIED_EMPTY"]*6);continue
  require(not f.row["empty_certified"],"PG2_EMPTY_CONTRADICTION")
  sdf=box_sdf(p[b],boxes)
  n=sdf_normal(p[b],boxes)
  closing=(-(v[b]*n).sum(-1)).clamp_min(0)
  buffer=closing.square()/(2*cfg["braking_acceleration_mps2"]) if cfg["braking_buffer"] else closing*0
  target=.15+buffer
  risk=cfg["risk_eps_m"]*F.softplus((target-sdf)/cfg["risk_eps_m"])/.15
  losses.append(weighted_lse(risk[None],alpha[b:b+1],cfg["lse_temperature"])[0]);risks.append(risk)
  severe.append((sdf.detach()<.12).any(-1))
  with torch.no_grad():
   point_fail=sdf<.15
   chord=(p[b,:,1:]-p[b,:,:-1]).norm(dim=-1)
   gap=chord>(sdf[:,1:]-.15)+(sdf[:,:-1]-.15)
   fail=point_fail.clone();fail[:,1:]|=gap
  vals=[];reasons=[];indices=[];targets=[]
  for q in range(6):
   ix=torch.nonzero(fail[q],as_tuple=False).flatten()
   if not len(ix):
    vals.append(p[b,q].sum()*0);reasons.append("NO_DISCRETE_FAILURE");indices.append(-1);targets.append(p[b,q,0].detach()*0);continue
   i=int(ix[0]);indices.append(i)
   safe=torch.nonzero(sdf[q,:i].detach()>=.15+1e-6,as_tuple=False).flatten()
   if not len(safe):
    vals.append(p[b,q].sum()*0);reasons.append("NO_PREVIOUS_SAFE_ANCHOR");targets.append(p[b,q,0].detach()*0);continue
   j=int(safe[-1]);anchor=p[b,q,j].detach()
   require(float(box_sdf(anchor[None],boxes)[0])>=.15,"PG2_RETREAT_ANCHOR")
   vals.append(((p[b,q,i]-anchor)/cfg["position_reference_m"]).square().sum())
   targets.append(anchor);reasons.append("POINT_WITNESS" if bool(point_fail[q,i]) else "CHORD_UNPROVEN_PROXY_ONLY")
  retreat.append(torch.stack(vals));diagnostics.append(reasons);first_indices.append(indices);target_rows.append(torch.stack(targets))
 return dict(clearance=torch.stack(losses),retreat=torch.stack(retreat),point_risk=torch.stack(risks),
  severe=torch.stack(severe),first_indices=first_indices,retreat_targets=torch.stack(target_rows),
  reasons=diagnostics,certificate=False)

def corridor_loss(p,w,guidance,reference=1.):
 require(guidance["target"].shape==p.shape and guidance["radius"].shape==w.shape,"PG2_CORRIDOR_SHAPE")
 distance=(p-guidance["target"].detach()).norm(dim=-1)
 value=((distance-guidance["radius"].detach()).clamp_min(0)/reference).square()
 return (w/w.sum(-1,keepdim=True)*value).sum(-1)*guidance["available"]

def guidance_loss(endpoint,velocity,acceleration,goals,enabled,severe,route_target=None):
 for x in (endpoint,velocity,acceleration,goals):finite(x,"PG2_GUIDANCE_FINITE")
 require(endpoint.ndim==3 and endpoint.shape[1:]==(6,3) and velocity.shape==acceleration.shape==endpoint.shape and goals.shape==(len(endpoint),3),"PG2_GUIDANCE_SHAPE")
 require(enabled.dtype==severe.dtype==torch.bool and enabled.shape==(len(endpoint),) and severe.shape==endpoint.shape[:2],"PG2_GUIDANCE_MASK")
 distance=goals.norm(dim=-1)
 require(bool((distance>0).all()),"PG2_ZERO_GOAL")
 unit=goals/distance[:,None];near=goals.detach().double().norm(dim=-1)<=CONFIG["near_distance_threshold_m"]
 ref=distance.clamp_min(1.)[:,None]
 near_pos=((endpoint-goals[:,None])/ref[...,None]).square().sum(-1)
 near_v=velocity.square().sum(-1)/(3.*3.)
 near_a=acceleration.square().sum(-1)/(8.*8.)
 progress=(endpoint*unit[:,None]).sum(-1)
 # Smooth bounded progress reward; Far is never required to reach the global goal.
 far_progress=torch.exp(-progress/ref)
 vnorm=velocity.norm(dim=-1)
 far_direction=((1-(velocity*unit[:,None]).sum(-1)/vnorm.clamp_min(.05))*(vnorm>.05)).clamp_min(0)
 normal=torch.where(near[:,None],near_pos+near_v+near_a,far_progress+.1*far_direction)
 # A severe collision cannot receive a direct goal-pushing gradient.
 if route_target is None:repair=endpoint.sum(-1)*0
 else:repair=((endpoint-route_target.detach())/ref[...,None]).square().sum(-1)
 value=torch.where(severe.detach(),repair,normal)
 return value*enabled[:,None],dict(near=near,masked_direct_goal=severe.detach(),normal=normal,repair=repair)

def feasibility_loss(ana,trajectory,goals,bounds,cfg,gravity_B0=None):
 base,terms=dynamic_limit_loss(ana["v"],ana["a"],ana["jerk"],ana["weights"],cfg["dynamic_limits"])
 w=ana["weights"];v=ana["v"];a=ana["a"];speed=v.norm(dim=-1)
 with torch.no_grad():
  kappa=torch.cross(v,a,dim=-1).norm(dim=-1)/speed.clamp_min(.1).pow(3)
  cap=(cfg["lateral_acceleration_mps2"]/kappa.clamp_min(1e-6)).sqrt().clamp(max=cfg["dynamic_limits"][0])
 curve=(w*(speed/cap-1).clamp_min(0).square()).sum(-1)
 near=goals.detach().double().norm(dim=-1)<=CONFIG["near_distance_threshold_m"]
 terminal=(ana["terminal_velocity"].norm(dim=-1)/3.).square()*near[:,None]
 domain=center_domain_loss(ana["p"],w,bounds)
 flat=base*0
 diagnostics={}
 if cfg["flatness_enabled"]:
  require(gravity_B0 is not None and gravity_B0.shape==(len(v),3) and not gravity_B0.requires_grad and
   bool(torch.isfinite(gravity_B0).all()) and bool(((gravity_B0.norm(dim=-1)-9.81).abs()<1e-3).all()),"PG2_GRAVITY")
  force=a-gravity_B0[:,None,None,:];mag=force.norm(dim=-1).clamp_min(1e-5)
  up=-gravity_B0/9.81
  cosine=(force*up[:,None,None,:]).sum(-1)/mag
  tilt=(math.cos(cfg["tilt_max_rad"])-cosine).clamp_min(0).square()
  thrust=((cfg["thrust_acceleration_min"]-mag).clamp_min(0)/9.81).square()+((mag-cfg["thrust_acceleration_max"]).clamp_min(0)/9.81).square()
  # Zero-drag, fixed-yaw differential-flatness perpendicular body rate only.
  j=ana["jerk"];perp=j-(j*force).sum(-1,keepdim=True)/mag[...,None].square()*force
  omega=perp.norm(dim=-1)/mag
  body=(omega/cfg["body_rate_max_radps"]-1).clamp_min(0).square()
  flat=(w*(tilt+thrust+body)).sum(-1)
  diagnostics=dict(thrust_acceleration=mag,perpendicular_body_rate=omega)
 return base+cfg["curvature_scale"]*curve+cfg["terminal_stop_scale"]*terminal+cfg["domain_scale"]*domain+flat,dict(limit=base,curvature=curve,terminal_stop=terminal,domain=domain,flatness=flat,limit_terms=terms,**diagnostics)

def control_loss(trajectory,cfg):
 q=dynamic_quality_loss(trajectory)
 parts=q["terms"]
 total=sum(cfg["quality_scales"][k]*v for k,v in parts.items())
 return total,q
