"""Physical-time proxies; statuses describe evidence, never a flight certificate."""
import torch
import torch.nn.functional as F
from query_minco.training.io import require
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.training.pg1.losses import goal_task_loss,dynamic_limit_loss,center_domain_loss
from query_minco.training.pg1_v2_r1.losses import weighted_lse,dynamic_quality_loss
from query_minco.training.pg2_guided_v1.losses import sdf_normal,corridor_loss,guidance_loss as dense_v1
from .spec import validate

def buffer(closing,cfg):
 # Extra reaction-distance proxy, capped by horizon and spatial reserve.
 # Required full stopping distance is logged separately, never certified by cap.
 required=closing.square()/(2*cfg["braking_acceleration"])
 cap=torch.minimum(closing*cfg["braking_horizon_s"],torch.full_like(closing,cfg["braking_cap_m"]))
 bounded=cap*(-torch.expm1(-required/cap.clamp_min(1e-8)))
 return bounded,required,required.detach()>cap.detach()+1e-6

def safety(p,v,w,fields,route,cfg):
 validate(cfg);require(p.shape==v.shape and p.shape[:-1]==w.shape and p.ndim==4 and p.shape[1]==6 and p.shape[-1]==3,"V2_SAMPLE_SHAPE")
 require(all(bool(torch.isfinite(x).all()) for x in (p,v,w)) and bool((w>=0).all()) and bool((w.sum(-1)>0).all()) and len(fields)==len(p),"V2_SAMPLE_FINITE")
 alpha=w/w.sum(-1,keepdim=True);cc=[];bb=[];rr=[];flags=[];records=[];signed=[];pointsets=[];unknownsets=[]
 for b,f in enumerate(fields):
  boxes=f.boxes.to(p);require(type(f.row["empty_certified"]) is bool,"V2_EMPTY_CERT")
  zero=p[b,...,0].sum(-1)*0
  if not len(boxes):
   require(f.row["empty_certified"],"V2_UNKNOWN_EMPTY")
   cc.append(zero);bb.append(zero);rr.append(zero)
   flags.append(torch.zeros(6,dtype=torch.bool,device=p.device));pointsets.append(torch.zeros_like(w[b],dtype=torch.bool));unknownsets.append(torch.zeros(6,dtype=torch.bool,device=p.device))
   signed.append(None);records.append([dict(status="NO_DISCRETE_FAILURE",retreat="CERTIFIED_EMPTY") for _ in range(6)]);continue
  require(not f.row["empty_certified"],"V2_EMPTY_CONTRADICTION")
  sdf=box_sdf(p[b],boxes);n=sdf_normal(p[b],boxes)
  closing=(-(v[b]*n).sum(-1)).clamp_min(0)
  reserve,required,capped=buffer(closing,cfg)
  r0=cfg["eps_m"]*F.softplus((.15-sdf)/cfg["eps_m"])/.15
  r1=cfg["eps_m"]*F.softplus((.15+reserve-sdf)/cfg["eps_m"])/.15
  l0=weighted_lse(r0[None],alpha[b:b+1],cfg["tau"])[0]
  l1=weighted_lse(r1[None],alpha[b:b+1],cfg["tau"])[0]
  cc.append(l0);bb.append(l1-l0)
  with torch.no_grad():
   points=sdf<.15
   chord=(p[b,:,1:]-p[b,:,:-1]).norm(dim=-1)>(sdf[:,1:]-.15)+(sdf[:,:-1]-.15)
   anypoint=points.any(-1);unknown=chord.any(-1)&~anypoint
  flags.append(anypoint);pointsets.append(points);unknownsets.append(unknown)
  signed.append(sdf-.12);values=[];meta=[]
  for q in range(6):
   common=dict(required_buffer_max_m=float(required[q].max().detach()),used_buffer_max_m=float(reserve[q].max().detach()),cap_exceeded_nodes=int(capped[q].sum()))
   if not bool(anypoint[q]):
    values.append(zero[q]);meta.append(dict(common,status="CHORD_UNPROVEN" if bool(unknown[q]) else "NO_DISCRETE_FAILURE",retreat="UNKNOWN_NO_RETREAT" if bool(unknown[q]) else "NOT_TRIGGERED"));continue
   i=int(torch.nonzero(points[q],as_tuple=False)[0])
   anchors=torch.nonzero(sdf[q,:i].detach()>=.150001,as_tuple=False).flatten()
   reason="START_INVALID" if i==0 else ("NEAR_START_INVALID" if i==1 else "NO_SAFE_ANCHOR")
   if not len(anchors):
    values.append(zero[q]);meta.append(dict(common,status="CONFIRMED_VIOLATION",retreat=reason,first_node=i));continue
   anchor=p[b,q,int(anchors[-1])].detach()
   require(float(box_sdf(anchor[None],boxes)[0])>=.15,"V2_ANCHOR")
   values.append((p[b,q,i]-anchor).square().sum())
   meta.append(dict(common,status="CONFIRMED_VIOLATION",retreat="PENETRATION_WITH_ANCHOR" if float(sdf[q,i].detach())<.12 else "MARGIN_WITH_ANCHOR",first_node=i,anchor=anchor.tolist()))
  rr.append(torch.stack(values));records.append(meta)
 confirmed=torch.stack(flags);unknown=torch.stack(unknownsets)
 corridor=corridor_loss(p,w,route)*(confirmed.to(p.dtype)+cfg["chord_corridor_scale"]*unknown)
 # Hard-pass routes are not pulled into one Teacher tube.
 return dict(clearance=torch.stack(cc),braking=torch.stack(bb),retreat=torch.stack(rr),corridor=corridor,
  confirmed=confirmed,chord_unproven=unknown,records=records,signed_samples=signed,certificate=False)

def goal(endpoint,velocity,acceleration,goals,enabled,blocked,route):
 original,old=goal_task_loss(endpoint,velocity,goals,enabled)
 dense,diag=dense_v1(endpoint,velocity,acceleration,goals,enabled,blocked,route["target"][:,:,-1])
 # Original numerical task penalties stay intact. Unsafe direct attraction is
 # detached until repaired; route guidance supplies the repair derivative.
 task=torch.where(blocked.detach(),original.detach()+endpoint.sum(-1)*0,original)
 return task,dense,dict(original_unmodified=original,original_gradient_masked=blocked.detach(),old=old,dense=diag)

def lateral(velocity,acceleration,weights,limit=4.,epsilon=.1):
 require(limit>0 and epsilon>0 and velocity.shape==acceleration.shape and velocity.shape[:-1]==weights.shape,"V2_LATERAL")
 square=torch.cross(velocity,acceleration,dim=-1).square().sum(-1)/(velocity.square().sum(-1)+epsilon**2)
 value=(square/(limit**2)-1).clamp_min(0).square()
 return ((weights/weights.sum(-1,keepdim=True))*value).sum(-1),square

def feasibility(ana,bounds,cfg):
 limit,components=dynamic_limit_loss(ana["v"],ana["a"],ana["jerk"],ana["weights"],(3.,8.,100.))
 lat,lat_sq=lateral(ana["v"],ana["a"],ana["weights"],cfg["lateral_acceleration"],cfg["low_speed_epsilon"])
 domain=center_domain_loss(ana["p"],ana["weights"],bounds)
 return dict(dynamic_limit=limit,domain=domain,lateral=lat),dict(original_dynamic_terms=components,lateral_squared=lat_sq)

def control(trajectory):
 # Four-node positive Gauss quadrature is exact for quintic a^2 and jerk^2.
 return dynamic_quality_loss(trajectory)

def flatness_diagnostic(ana,trajectory,goals,bounds,gravity_B0,*,frame,assumptions):
 """Optional diagnostic only; no v2 training recipe enables this component."""
 require(frame=="fixed B0 FLU" and assumptions==("zero_drag","fixed_yaw","specific_thrust","measured_gravity"),"V2_FLATNESS_ASSUMPTIONS")
 from query_minco.training.pg2_guided_v1.losses import feasibility_loss
 from query_minco.training.pg2_guided_v1.spec import recipe as v1_recipe
 cfg=v1_recipe();cfg["flatness_enabled"]=True
 _,diag=feasibility_loss(ana,trajectory,goals,bounds,cfg,gravity_B0)
 return diag["flatness"],dict(diag,scope="DIAGNOSTIC_NOT_CONTROLLER_VALIDATION")
