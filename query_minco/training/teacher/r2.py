"""R2 search only; no historical main(), paths or dataset dispatch."""
import numpy as np,torch
from scipy.optimize import minimize
from .reference64 import solve64
from .c1a import seeds,solve,inspect_candidate,differentiable_cost
from query_minco.evaluation.goal import expert_tail

def search(state,scene,mode,folder,cfg):
 s=np.asarray(state,np.float32);head=np.stack([np.zeros(3,np.float32),s[:3],s[3:6]]);tail=expert_tail(s)
 ip,dt=seeds(s,scene,mode);h=torch.from_numpy(head);t=torch.from_numpy(tail)
 with torch.no_grad():initial=solve(h,t,torch.from_numpy(ip),torch.from_numpy(dt)).numpy()
 attempts=[];history=[]
 for i in range(len(ip)):
  teacher,receipt=inspect_candidate(initial[i],dt[i],head,tail,ip[i],s,scene,mode,"seed",0,i);attempts.append(receipt)
  if teacher is not None:
   with (folder/(mode+"_search.npz")).open("xb") as f:np.savez_compressed(f,initial_coeffs=initial,initial_durations=dt,initial_inner=ip,final_coeffs=np.empty((0,3,3,6)),final_durations=np.empty((0,3)),final_inner=np.empty((0,2,3)))
   return teacher,{"mode":mode,"attempts":attempts,"iterations":[],"scipy_called":False}
 n=len(ip);total=dt.astype(float).sum(-1);z=(total-.8)/1.4
 x0=np.c_[ip.reshape(n,6),np.log(z/(1-z)),np.log((dt-.12)/(total[:,None]-.36))].astype(float)
 cx=scene["crossing_x_m"];bounds=[]
 for _ in range(n):
  yl,yh=(-1.5,1.5);zl,zh=(-1.1,1.1)
  if mode=="left":yl=.46
  if mode=="right":yh=-.46
  if mode=="up":zl=.46
  if mode=="down":zh=-.46
  bounds += [(.4,cx-.15),(yl,yh),(zl,zh),(cx+.15,float(tail[0,0])-.20),(yl,yh),(zl,zh),(-4,8),(-4,4),(-4,4),(-4,4)]
 base=torch.tensor(ip,dtype=torch.float64)
 def unpack(x):
  v=x.reshape(n,10);inner=v[:,:6].reshape(n,2,3);T=.8+1.4*v[:,6].sigmoid();duration=.12+(T-.36)[:,None]*v[:,7:10].softmax(-1)
  return inner,duration
 eval_count=[0]
 def objective(x):
  v=torch.tensor(x,dtype=torch.float64,requires_grad=True);inner,duration=unpack(v);coeff=solve64(h,t,inner,duration)
  costs,_=differentiable_cost(coeff,duration,inner,base,scene);loss=costs.mean();g=torch.autograd.grad(loss,v)[0]
  if not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(loss)):raise RuntimeError("nonfinite offline numerical search")
  eval_count[0]+=1
  return float(loss.detach()),g.detach().numpy()
 def callback(x):
  value,grad=objective(x);history.append({"iteration":len(history)+1,"objective":value,"trajectory_variable_gradient_norm":float(np.linalg.norm(grad)),"variables":x.tolist()})
 result=minimize(objective,x0.ravel(),jac=True,method="L-BFGS-B",bounds=bounds,callback=callback,
  options={"maxiter":cfg["fallback_maxiter"],"maxfun":cfg["fallback_maxfun"],"ftol":cfg["fallback_ftol"],"gtol":cfg["fallback_gtol"],"maxls":cfg["fallback_maxls"]})
 with torch.no_grad():
  inner,duration=unpack(torch.from_numpy(result.x));coeff=solve64(h,t,inner,duration);cc=coeff.numpy();dd=duration.numpy();ii=inner.numpy()
 with (folder/(mode+"_search.npz")).open("xb") as f:np.savez_compressed(f,initial_coeffs=initial,initial_durations=dt,initial_inner=ip,final_coeffs=cc,final_durations=dd,final_inner=ii,scipy_x=result.x)
 chosen=None
 for i in range(n):
  candidate,receipt=inspect_candidate(cc[i],dd[i],head,tail,ii[i],s,scene,mode,"scipy_offline",int(result.nit),i);attempts.append(receipt)
  if chosen is None and candidate is not None:chosen=candidate
 return chosen,{"mode":mode,"attempts":attempts,"iterations":history,"scipy_called":True,"scipy_success":bool(result.success),"scipy_message":str(result.message),
  "scipy_nit":int(result.nit),"scipy_nfev":int(result.nfev),"objective_evaluations_including_callback":eval_count[0],"network_parameters_optimized":False}
