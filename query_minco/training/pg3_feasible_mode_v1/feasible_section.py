"""Teacher-side convex section regions; time-aligned proxy, never trajectory certificate."""
import numpy as np
import torch
from query_minco.training.io import require,digest
from .corridor import validate_scene
from .bottleneck import at_progress,bottleneck_loss
def configuration():
 return dict(schema="pg3_feasible_section_region_v1",radius=.12,margin=.03,pad=1e-6,sigma_m=1.,dimension=2,
  reduction="mean_four_unit_axis_faces_then_valid_sections_then_eligible_queries_then_batch",
  time_alignment="UNCHANGED_TEACHER_PROGRESS_TIMES_PREDICTED_TOTAL",invalid="ZERO_REGION_LOSS_KEEP_EXISTING_MODE_AND_CORRIDOR",projection=False)
def halfspace_loss(u,A,b,sigma=1.):
 require(u.ndim==2 and u.shape[1] in (1,2) and A.ndim==2 and A.shape[1]==u.shape[1] and b.shape==(len(A),) and len(A)>0 and sigma==1.,"REGION_LOSS_SHAPE")
 require(all(bool(torch.isfinite(x).all()) for x in (u,A,b)),"REGION_LOSS_NONFINITE")
 A=A.detach().to(u);b=b.detach().to(u)
 require(bool(torch.isclose(A.norm(dim=-1),torch.ones(len(A),device=u.device,dtype=u.dtype),atol=1e-6).all()),"REGION_UNIT_PLANES")
 residual=(u@A.T-b)/sigma
 return residual.clamp_min(0).square().mean(-1)
def faces_for(box,point,pad):
 result=[]
 for axis in range(3):
  if point[axis]<=box["min"][axis]-.15-pad:result.append((float(box["min"][axis]-.15-pad-point[axis]),axis,1.))
  if point[axis]>=box["max"][axis]+.15+pad:result.append((float(point[axis]-box["max"][axis]-.15-pad),axis,-1.))
 return result
def build_region(teacher,label,scene,empty_certified=False):
 bounds=validate_scene(scene,empty_certified);cfg=configuration()
 if teacher.get("certified") is not True:return None,"TEACHER_NOT_CERTIFIED"
 if teacher.get("representable") is not True:return None,"TEACHER_UNREPRESENTABLE"
 if not label.get("valid") or label.get("side") is None:return None,"NO_CERTIFIED_TEACHER_SIDE"
 point=np.asarray(label["position"],float);plane=float(label["plane_x"]);bi=int(label["box"]);side=label["side"]
 require(point.shape==(3,) and np.isfinite(point).all() and np.isfinite(plane) and abs(point[0]-plane)<1e-7,"REGION_TEACHER_PLANE")
 require(0<=bi<len(scene["obstacles"]) and side[0] in (1,2) and side[1] in (-1,1),"REGION_LABEL")
 if not (bounds[0]<=point).all() or not (point<=bounds[1]).all():return None,"TEACHER_OUTSIDE_CENTER_DOMAIN"
 lower=bounds[0,1:].copy();upper=bounds[1,1:].copy();selected=[]
 for i,box in enumerate(scene["obstacles"]):
  candidates=faces_for(box,point,cfg["pad"])
  if i==bi:candidates=[x for x in candidates if x[1]==side[0] and x[2]==-side[1]]
  if not candidates:return None,"NO_CONSERVATIVE_AABB_FACE_FOR_OBSTACLE_"+str(i)
  _,axis,sign=max(candidates,key=lambda x:(x[0],-x[1],x[2]))
  limit=box["min"][axis]-.15-cfg["pad"] if sign==1 else -box["max"][axis]-.15-cfg["pad"]
  if axis>0:
   if sign==1:upper[axis-1]=min(upper[axis-1],limit)
   else:lower[axis-1]=max(lower[axis-1],-limit)
  selected.append(dict(box=i,axis=axis,sign=sign,limit=float(limit)))
 if not (lower<upper).all():return None,"EMPTY_REGION_INTERSECTION"
 A=[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]];b=[upper[0],-lower[0],upper[1],-lower[1]]
 proof=dict(schema=cfg["schema"],A=A,b=[float(x) for x in b],lower=lower.tolist(),upper=upper.tolist(),plane_x=plane,
  radius=.12,margin=.03,pad=cfg["pad"],faces=selected,scene_digest=digest(scene),landmark_digest=digest(label),
  teacher_identity=teacher["cache_identity"],teacher_id=teacher["teacher"],mode_side=side,teacher_point=point.tolist())
 require(verify_region(proof,teacher,label,scene,empty_certified),"REGION_INTERNAL_CERTIFICATION")
 return proof,"CERTIFIED_SECTION_REGION_ONLY"
def verify_region(proof,teacher,label,scene,empty_certified=False):
 try:
  bounds=validate_scene(scene,empty_certified)
  if teacher.get("certified") is not True or teacher.get("representable") is not True:return False
  if proof["scene_digest"]!=digest(scene) or proof["landmark_digest"]!=digest(label) or proof["teacher_identity"]!=teacher["cache_identity"] or proof["teacher_id"]!=teacher["teacher"]:return False
  if proof["radius"]!=.12 or proof["margin"]!=.03 or proof["pad"]!=1e-6 or proof["mode_side"]!=label["side"]:return False
  A=np.asarray(proof["A"],float);b=np.asarray(proof["b"],float);lo=np.asarray(proof["lower"],float);hi=np.asarray(proof["upper"],float)
  if not np.array_equal(A,[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]) or not np.isfinite(b).all() or not (lo<hi).all():return False
  if not np.array_equal(b,[hi[0],-lo[0],hi[1],-lo[1]]) or not (bounds[0,1:]<=lo).all() or not (hi<=bounds[1,1:]).all():return False
  p=np.asarray(label["position"],float)
  if proof["plane_x"]!=label["plane_x"] or not np.array_equal(proof["teacher_point"],p) or not (A@p[1:]<=b+1e-10).all():return False
  corners=np.array([[proof["plane_x"],y,z] for y in (lo[0],hi[0]) for z in (lo[1],hi[1])])
  if not (bounds[0]<=corners).all() or not (corners<=bounds[1]).all():return False
  if sorted(f["box"] for f in proof["faces"])!=list(range(len(scene["obstacles"]))):return False
  for f in proof["faces"]:
   axis=f["axis"];sign=f["sign"];box=scene["obstacles"][f["box"]]
   if axis not in (0,1,2) or sign not in (-1.,1.):return False
   limit=box["min"][axis]-.15-1e-6 if sign==1 else -box["max"][axis]-.15-1e-6
   if f["limit"]!=limit or not (corners[:,axis]*sign<=limit+1e-12).all():return False
   if f["box"]==label["box"] and (axis!=label["side"][0] or sign!=-label["side"][1]):return False
  return True
 except (KeyError,TypeError,IndexError,ValueError):return False
class RegionBottleneck:
 def __init__(self):self.cfg=configuration();self.cache={}
 def __call__(self,c,d,teacher,bank):
  point,old=bottleneck_loss(c,d,teacher["bottlenecks"])
  key=digest(dict(teacher=teacher["cache_identity"],id=teacher["teacher"],scene=bank["scene"],labels=teacher["bottlenecks"],cfg=self.cfg))
  if key not in self.cache:
   self.cache[key]=[build_region(teacher,x,bank["scene"],bank["empty_certified"]) for x in teacher["bottlenecks"] if x["valid"]]
  values=[];records=[]
  labels=[x for x in teacher["bottlenecks"] if x["valid"]]
  for label,(proof,status) in zip(labels,self.cache[key]):
   if proof is None:records.append(dict(status=status,valid=False));continue
   require(verify_region(proof,teacher,label,bank["scene"],bank["empty_certified"]),"REGION_STALE_PROOF")
   p=at_progress(c,d,[label["progress"]]);A=p.new_tensor(proof["A"]).detach();b=p.new_tensor(proof["b"]).detach()
   v=halfspace_loss(p[:,1:],A,b,self.cfg["sigma_m"])[0];values.append(v)
   records.append(dict(status=status,valid=True,proof=proof,prediction_yz=p[0,1:].detach().tolist(),max_violation_m=float((p[:,1:]@A.T-b).detach().max()),contained=bool((p[:,1:]@A.T<=b).detach().all())))
  loss=torch.stack(values).mean() if values else (c.sum()+d.sum())*0
  return loss,dict(status="REGION_TIME_ALIGNED_PROXY_NOT_FULL_TRAJECTORY_CERTIFICATE",count=len(values),region_records=records,
   original_teacher_point_loss=float(point.detach()),lateral_error_m=old.get("lateral_error_m",[]),axial_error_m=old.get("axial_error_m",[]),
   valid_regions=len(values),total_sections=len(records),normalization="mean4faces/validsections; sigma1m",invalid_fallback="NO_REGION_LABEL_KEEP_MODE_CORRIDOR")
