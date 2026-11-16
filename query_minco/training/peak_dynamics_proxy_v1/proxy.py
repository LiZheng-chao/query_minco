"""Versioned peak-aware dynamics diagnostic. Original objectives/Hard are untouched."""
import math
import torch
COMPONENT_ID='subdivided_bernstein_peak_upper_v1'
def _controls(coeffs,durations,order):
 n=5-order
 powers=torch.stack([coeffs[...,order+i]*math.factorial(order+i)/math.factorial(i)*durations[...,None].pow(i) for i in range(n+1)],-2)
 rows=[]
 for j in range(n+1):rows.append(sum(powers[...,i,:]*math.comb(j,i)/math.comb(n,i) for i in range(j+1)))
 return torch.stack(rows,-2)
def subdivide(c):
 levels=[c]
 while levels[-1].shape[-2]>1:levels.append((levels[-1][...,:-1,:]+levels[-1][...,1:,:])*.5)
 left=torch.stack([x[...,0,:] for x in levels],-2);right=torch.stack([x[...,-1,:] for x in levels[::-1]],-2)
 return torch.stack([left,right],-3).flatten(-4,-3)
def peak_proxy(coeffs,durations,depth=6,accumulation=torch.float64,epsilon_ratio=1e-9):
 if coeffs.ndim!=5 or coeffs.shape[1] not in (1,6) or coeffs.shape[-3:]!=(3,3,6) or durations.shape!=coeffs.shape[:3]:raise ValueError('PEAK_SHAPE')
 if coeffs.device.type!='cpu' or coeffs.dtype not in (torch.float32,torch.float64) or durations.dtype!=coeffs.dtype or not torch.isfinite(coeffs).all() or not torch.isfinite(durations).all():raise ValueError('PEAK_FINITE_TYPE')
 if durations.requires_grad or not torch.allclose(durations,torch.full_like(durations,2/3),atol=1e-7,rtol=0):raise ValueError('FIXED2_DURATION_REQUIRED')
 if not math.isfinite(epsilon_ratio) or epsilon_ratio<=0:raise ValueError('PEAK_EPSILON')
 if type(depth) is not int or not 0<=depth<=8 or accumulation not in (torch.float32,torch.float64):raise ValueError('PEAK_CONFIG')
 c=coeffs.to(accumulation);d=durations.to(accumulation);terms={};ratios={};bounds={}
 for order,name,limit in [(1,'speed',3.),(2,'acceleration',8.),(3,'jerk',100.)]:
  b=_controls(c,d,order).unsqueeze(-3)
  for _ in range(depth):b=subdivide(b)
  normalized=(b/limit).square().sum(-1).add(epsilon_ratio**2).sqrt()
  upper=normalized.amax(dim=(-3,-2,-1))
  h=d/(2**depth);degree=5-order
  derivative=degree*(b[...,1:,:]-b[...,:-1,:])/h[...,None,None,None]
  gap=(h[...,None,None]*derivative.norm(dim=-1)/limit).amax(dim=(-3,-2,-1))+epsilon_ratio
  ratios[name]=upper;terms['F_peak_'+name]=(upper-1).clamp_min(0).square();bounds[name]=gap
 return dict(F_peak=sum(terms.values()),terms=terms,upper_ratios=ratios,error_bounds_ratio=bounds,
  component_id=COMPONENT_ID,accumulation=str(accumulation),depth=depth)
def diagnostic_objective(original_result,depth=6):
 p=peak_proxy(original_result.trajectory.coeffs,original_result.trajectory.durations,depth)
 # Phase-A mean normalization,not a change to domain,collision,goal/quality terms or weights.
 loss=original_result.loss-original_result.metrics['F_dyn'].mean()+p['F_peak'].mean()
 return loss,p
