"""Training-only Teacher free-ball guidance and original hard-label producer.

A ball around a verified Teacher point is not a corridor certificate for the
predicted curve. All predicted trajectories retain the ORIGINAL hard validator.
"""
import hashlib
import numpy as np
import torch
from query_minco.training.io import require,digest,sha
from query_minco.training.run_spec import ROOT
from query_minco.training.pg1.losses import polynomial_values
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.evaluation.validation import validate_teacher,evaluate

def tensor_digest(*values):
 h=hashlib.sha256()
 for x in values:
  a=x.detach().cpu().contiguous().numpy()
  h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
 return h.hexdigest()

def scene_from_field(field):
 return dict(obstacles=field.row["obstacles"],flight_center_domain_m=field.row["flight_center_domain_m"])

def check_curve(c,d,scene,head=None):
 c=np.asarray(c,dtype=np.float64);d=np.asarray(d,dtype=np.float64)
 require(c.shape==(3,3,6) and d.shape==(3,) and np.isfinite(c).all() and np.isfinite(d).all() and (d>0).all(),"PG2_CURVE")
 if head is None: head=np.stack([evaluate(c[0],0.,k) for k in range(3)])
 tail=np.stack([evaluate(c[-1],d[-1],k) for k in range(3)])
 inner=np.stack([evaluate(c[i],d[i]) for i in range(2)])
 return validate_teacher(c,d,head,tail,inner,scene)

class TeacherGuidance:
 """Caches only immutable Teacher+state+geometry content, not predicted proofs."""
 def __init__(self): self.cache={}
 def build(self,samples,fields,matches,p,weights):
  targets=[];radii=[];assignments=[];records=[]
  u=torch.linspace(0,1,64,dtype=p.dtype,device=p.device)
  for b,(sample,field,match) in enumerate(zip(samples,fields,matches)):
   scene=scene_from_field(field)
   key=digest(dict(tensors=tensor_digest(sample.teacher_coeffs,sample.teacher_durations,sample.state),scene=scene))
   if key not in self.cache:
    cc=sample.teacher_coeffs.detach().double().cpu().numpy()
    dd=sample.teacher_durations.detach().double().cpu().numpy()
    head=np.stack([np.zeros(3),sample.state[:3].numpy(),sample.state[3:6].numpy()])
    checks=[check_curve(c,d,scene,head) for c,d in zip(cc,dd)]
    require(all(x["passed"] for x in checks),"PG2_TEACHER_NOT_HARD_FEASIBLE_"+sample.sample_id)
    self.cache[key]=checks
   checks=self.cache[key];m=len(checks)
   if not m:
    targets.append(torch.zeros_like(p[b]));radii.append(torch.zeros_like(weights[b]))
    assignments.append([-1]*6);records.append(dict(sample_id=sample.sample_id,status="NO_TEACHER_UNKNOWN_NOT_INFEASIBLE",teachers=0))
    continue
   c=sample.teacher_coeffs.to(p);d=sample.teacher_durations.to(p)
   path=polynomial_values(c,d[...,None]*u).flatten(1,2).detach()
   cost=((p[b].detach()[:,None]-path[None])**2*weights[b].detach()[:,None,:,None]).sum((-1,-2))
   chosen=[None]*6;counts=[0]*m
   for q,t in zip(match["queries"],match["teachers"]): chosen[q]=t;counts[t]+=1
   for q in range(6):
    if chosen[q] is None:
     t=min(range(m),key=lambda t:(counts[t],float(cost[q,t]),t));chosen[q]=t;counts[t]+=1
   target=path[chosen].detach()
   boxes=field.boxes.to(p)
   clearance=box_sdf(target,boxes)-.15 if len(boxes) else torch.full_like(weights[b],float("inf"))
   bounds=p.new_tensor(field.row["flight_center_domain_m"])
   domain=torch.minimum(target-bounds[0],bounds[1]-target).amin(-1)
   radius=torch.minimum(clearance,domain).clamp(0,.2)
   radius=(radius-1e-5).clamp_min(0).detach()
   targets.append(target);radii.append(radius);assignments.append(chosen)
   records.append(dict(sample_id=sample.sample_id,status="VERIFIED_TEACHER_POINT_BALLS_NOT_CURVE_CERTIFICATE",
    teachers=m,teacher_usage=counts,teacher_checks_passed=len(checks),cache_identity=key,
    minimum_ball_radius_m=float(radius.min()),zero_radius_points=int((radius==0).sum())))
  return dict(target=torch.stack(targets),radius=torch.stack(radii),
   assignments=assignments,available=torch.tensor([[a>=0 for a in row] for row in assignments],device=p.device),
   records=records,certificate=False)

def hard_labels(out,samples,fields):
 """Fresh labels from the original continuous hard validator, never a query claim."""
 c=out.trajectory.coeffs.detach().double().cpu().numpy()
 d=out.parameters.durations.detach().double().cpu().numpy()
 head=out.parameters.head_pva.detach().double().cpu().numpy()
 checks=[];passed=[];clear=[]
 for b,field in enumerate(fields):
  row=[check_curve(c[b,q],d[b,q],scene_from_field(field),head[b,q]) for q in range(6)]
  checks.append(row);passed.append([x["passed"] for x in row])
  clear.append([min(x.get("surface_clearance_lower_bound_m",-1.),1.) if x["passed"] else -1. for x in row])
 binding=dict(trajectory_sha256=tensor_digest(out.trajectory.coeffs,out.parameters.durations,out.parameters.head_pva),
  state_sha256=tensor_digest(torch.stack([s.state for s in samples])),
  scenes_sha256=digest([scene_from_field(f) for f in fields]),
  validator_sha256=sha(ROOT/"query_minco/evaluation/validation.py"))
 return dict(passed=torch.tensor(passed,dtype=torch.bool,device=out.trajectory.coeffs.device),
  clearance=out.trajectory.coeffs.new_tensor(clear),checks=checks,binding=binding,
  origin="ORIGINAL_CONTINUOUS_HARD_VALIDATOR_TRAINING_ONLY",certificate=False)

def verify_label_binding(labels,out,samples,fields):
 require(labels["origin"]=="ORIGINAL_CONTINUOUS_HARD_VALIDATOR_TRAINING_ONLY","PG2_LABEL_ORIGIN")
 expected=dict(trajectory_sha256=tensor_digest(out.trajectory.coeffs,out.parameters.durations,out.parameters.head_pva),
  state_sha256=tensor_digest(torch.stack([s.state for s in samples])),
  scenes_sha256=digest([scene_from_field(f) for f in fields]),
  validator_sha256=sha(ROOT/"query_minco/evaluation/validation.py"))
 require(labels["binding"]==expected,"PG2_STALE_LABEL_BINDING")
