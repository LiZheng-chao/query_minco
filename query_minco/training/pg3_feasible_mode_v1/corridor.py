"""Conservative AABB face-separated polytopes; differentiable Bernstein containment."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from query_minco.training.io import require,digest

def bernstein(c,T):
 require(c.shape[-2:]==(3,6) and c.shape[:-2]==T.shape,"PG3_BERNSTEIN_SHAPE")
 require(bool(torch.isfinite(c).all()) and bool(torch.isfinite(T).all()) and bool((T>0).all()),"PG3_BERNSTEIN_FINITE")
 powers=T[...,None]**torch.arange(6,device=c.device,dtype=c.dtype)
 normalized=c*powers[...,None,:]
 rows=[]
 for k in range(6):
  rows.append(sum(normalized[...,i]*math.comb(k,i)/math.comb(5,i) for i in range(k+1)))
 return torch.stack(rows,dim=-2)

def split(B,u=.5):
 require(B.shape[-2:]==(6,3) and 0<=u<=1,"PG3_CASTELJAU")
 left=[B[...,0,:]];right=[B[...,-1,:]];work=B
 for _ in range(5):
  work=(1-u)*work[...,:-1,:]+u*work[...,1:,:]
  left.append(work[...,0,:]);right.append(work[...,-1,:])
 return torch.stack(left,-2),torch.stack(right[::-1],-2)

def restrict(B,lo,hi):
 require(0<=lo<hi<=1,"PG3_SUBINTERVAL")
 left=split(B,hi)[0] if hi<1 else B
 return split(left,lo/hi)[1] if lo>0 else left

def validate_scene(scene,empty_certified=False):
 obs=scene["obstacles"];bounds=np.asarray(scene["flight_center_domain_m"],float)
 require(bounds.shape==(2,3) and np.isfinite(bounds).all() and (bounds[0]<bounds[1]).all(),"PG3_DOMAIN")
 require(bool(obs) or empty_certified,"PG3_UNTRUSTED_EMPTY")
 for box in obs:
  lo=np.asarray(box["min"],float);hi=np.asarray(box["max"],float)
  require(lo.shape==hi.shape==(3,) and np.isfinite([lo,hi]).all() and (lo<hi).all(),"PG3_INVALID_AABB")
 return bounds

def polytope(B,scene,empty_certified=False,pad=1e-6):
 bounds=validate_scene(scene,empty_certified)
 cp=np.asarray(B,float);require(cp.shape==(6,3) and np.isfinite(cp).all(),"PG3_CONTROL_POINTS")
 A=[];b=[];faces=[]
 for axis in range(3):
  n=np.eye(3)[axis];A.extend([n.tolist(),(-n).tolist()]);b.extend([float(bounds[1,axis]),float(-bounds[0,axis])])
 if np.max(np.asarray(A)@cp.T-np.asarray(b)[:,None])>0:return None
 for bi,box in enumerate(scene["obstacles"]):
  candidates=[]
  for axis in range(3):
   for sign in (1.,-1.):
    n=np.eye(3)[axis]*sign
    limit=float(box["min"][axis]-.15-pad) if sign==1 else float(-box["max"][axis]-.15-pad)
    slack=limit-float((cp@n).max())
    if slack>=0:candidates.append((slack,axis,sign,n,limit))
  if not candidates:return None
  _,axis,sign,n,limit=max(candidates,key=lambda x:(x[0],-x[1],x[2]))
  faces.append(dict(box=bi,axis=axis,sign=sign,plane=len(A)));A.append(n.tolist());b.append(limit)
 proof=dict(A=A,b=b,faces=faces,scene_digest=digest(scene),radius=.12,margin=.03,pad=pad)
 require(verify_polytope(proof,scene,empty_certified),"PG3_INTERNAL_POLYTOPE_PROOF")
 return proof

def verify_polytope(proof,scene,empty_certified=False):
 bounds=validate_scene(scene,empty_certified)
 A=np.asarray(proof["A"],float);b=np.asarray(proof["b"],float)
 if A.ndim!=2 or A.shape[1]!=3 or b.shape!=(len(A),) or len(A)<6:return False
 if not np.isfinite(A).all() or not np.isfinite(b).all() or not np.allclose(np.linalg.norm(A,axis=1),1.,rtol=0,atol=1e-12):return False
 if proof.get("scene_digest")!=digest(scene) or proof.get("radius")!=.12 or proof.get("margin")!=.03 or proof.get("pad",0)<1e-6:return False
 for axis in range(3):
  if not np.array_equal(A[2*axis],np.eye(3)[axis]) or not np.array_equal(A[2*axis+1],-np.eye(3)[axis]):return False
  if b[2*axis]!=bounds[1,axis] or b[2*axis+1]!=-bounds[0,axis]:return False
 if sorted(f["box"] for f in proof["faces"])!=list(range(len(scene["obstacles"]))):return False
 for face in proof["faces"]:
  i=face["plane"];axis=face["axis"];sign=face["sign"]
  if not (0<=i<len(A) and axis in (0,1,2) and sign in (-1.,1.)):return False
  n=np.eye(3)[axis]*sign
  box=scene["obstacles"][face["box"]]
  limit=box["min"][axis]-.15-proof["pad"] if sign==1 else -box["max"][axis]-.15-proof["pad"]
  if not np.array_equal(A[i],n) or b[i]>limit:return False
 return True

def build_corridor(c,d,scene,empty_certified=False,max_depth=4):
 validate_scene(scene,empty_certified);leaves=[];missing=[]
 with torch.no_grad():
  B=bernstein(torch.as_tensor(c,dtype=torch.float64),torch.as_tensor(d,dtype=torch.float64))
  def visit(cp,segment,lo,hi,depth):
   proof=polytope(cp.numpy(),scene,empty_certified)
   if proof is not None:
    leaves.append(dict(segment=segment,lo=lo,hi=hi,proof=proof));return
   if depth==max_depth:missing.append(dict(segment=segment,lo=lo,hi=hi,reason="CORRIDOR_UNAVAILABLE_CONSERVATIVE_AABB_SEPARATION"));return
   a,b=split(cp);mid=(lo+hi)/2
   visit(a,segment,lo,mid,depth+1);visit(b,segment,mid,hi,depth+1)
  for segment in range(3):visit(B[segment],segment,0.,1.,0)
 return dict(leaves=leaves,missing=missing,complete=not missing,status="CERTIFIED_LOCAL_POLYTOPES" if not missing else "CORRIDOR_PARTIAL_OR_UNAVAILABLE")

def penalty(c,d,corridor,scene,empty_certified=False,sigma=.15):
 zero=(c.sum()+d.sum())*0
 require(sigma>0,"PG3_PLANE_SCALE")
 if not corridor["leaves"]:return zero,dict(status="CORRIDOR_UNAVAILABLE",complete=False,contained=False,terms=0)
 B=bernstein(c,d);values=[];contained=True;max_violation=0.
 for leaf in corridor["leaves"]:
  proof=leaf["proof"];require(verify_polytope(proof,scene,empty_certified),"PG3_STALE_OR_INVALID_POLYTOPE")
  sub=restrict(B[leaf["segment"]],leaf["lo"],leaf["hi"])
  A=sub.new_tensor(proof["A"]).detach();b=sub.new_tensor(proof["b"]).detach()
  residual=sub@A.T-b
  values.append(F.smooth_l1_loss(residual.clamp_min(0)/sigma,torch.zeros_like(residual),reduction="none",beta=1.).reshape(-1))
  contained=contained and bool((residual.detach()<=0).all())
  max_violation=max(max_violation,float(residual.detach().max()))
 loss=torch.cat(values).mean()
 return loss,dict(status=corridor["status"],complete=corridor["complete"],contained=bool(corridor["complete"] and contained),terms=sum(v.numel() for v in values),max_violation_m=max_violation,meaning="CONVEX_HULL_CONTAINMENT_SUFFICIENT_CONDITION_ORIGINAL_HARD_CHECK_STILL_REQUIRED")
