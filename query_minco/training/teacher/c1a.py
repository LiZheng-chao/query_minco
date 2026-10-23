"""Offline multi-start MINCO experts; optimizer variables are waypoints/time, never network weights."""
import hashlib,math
import numpy as np,torch
from query_minco._vendor.minco.minco import MincoS3NU
from query_minco.trajectory.postprocessor_legacy import TrajectoryPostprocessor
from query_minco.evaluation.validation import validate_teacher,evaluate,sample_curve
from query_minco.evaluation.signed_geometry import fine_geometry
from .config import CONFIG
from query_minco.evaluation.goal import expert_tail,task_metrics
def solve(head,tail,inner,durations):
 n=inner.shape[0];solver=MincoS3NU(piece_num=3,device=inner.device)
 solver.set_parameters(head[None].expand(n,-1,-1),tail[None].expand(n,-1,-1),inner,durations)
 return solver.coeffs.reshape(n,3,6,3).permute(0,1,3,2).contiguous()
def polynomial_samples(coeffs,durations,order=0,count=49):
 ts=durations[...,None]*torch.linspace(0,1,count,device=coeffs.device,dtype=coeffs.dtype)
 factors=coeffs.new_tensor([math.factorial(k)/math.factorial(k-order) for k in range(order,6)])
 powers=ts[...,None]**torch.arange(6-order,device=coeffs.device)
 return torch.einsum("npak,npsk->npsa",coeffs[...,order:]*factors,powers)
def differentiable_cost(coeffs,durations,inner,seed,scene):
 p=polynomial_samples(coeffs,durations);v=polynomial_samples(coeffs,durations,1);a=polynomial_samples(coeffs,durations,2);j=polynomial_samples(coeffs,durations,3)
 center=p.new_tensor([(np.array(b["min"])+np.array(b["max"]))/2 for b in scene["obstacles"]])
 half=p.new_tensor([(np.array(b["max"])-np.array(b["min"]))/2 for b in scene["obstacles"]])
 q=(p[...,None,:]-center).abs()-half
 signed=q.clamp_min(0).norm(dim=-1)+q.max(-1).values.clamp_max(0)
 clearance=signed.min(-1).values
 penetration=torch.relu(.175-clearance)
 vmax=v.norm(dim=-1).amax((1,2));amax=a.norm(dim=-1).amax((1,2));jmax=j.norm(dim=-1).amax((1,2))
 loss=4000*(penetration.square().mean((1,2))+penetration.amax((1,2)).square())+80*torch.relu(vmax-2.95).square()+8*torch.relu(amax-7.8).square()+.02*torch.relu(jmax-95.).square()
 loss=loss+.02*(inner-seed).square().mean((1,2))+.00001*j.square().sum(-1).mean((1,2))+.002*durations.sum(-1).square()
 return loss,{"min_obstacle_distance":clearance.amin((1,2)).detach(),"speed":vmax.detach(),"acceleration":amax.detach(),"jerk":jmax.detach()}
def modes_for(kind):
 return {"free":["direct"],"four_way":["left","right","up","down"],"left_only":["left"],"right_only":["right"],"up_only":["up"],"down_only":["down"],
  "narrow3d":["portal"],"multi3d":["left","right","up","down"]}[kind]
def seeds(state,scene,mode):
 tail=expert_tail(state);cx=scene["crossing_x_m"];seeds=[];times=[]
 for amp in CONFIG["seed_amplitudes_m"]:
  for total in CONFIG["seed_total_times_s"]:
   xs=np.array([cx-.40,cx+.40],np.float32);inner=xs[:,None]/tail[0,0]*tail[0]
   inner[:,0]=xs
   if mode in ["left","right"]:inner[:,1]+=(1 if mode=="left" else -1)*amp
   if mode in ["up","down"]:inner[:,2]+=(1 if mode=="up" else -1)*amp
   if mode=="portal":inner[:,1]=.84;inner[:,2]=.46
   headpos=np.zeros(3,np.float32);lengths=np.linalg.norm(np.diff(np.vstack([headpos,inner,tail[0]]),axis=0),axis=1)+.15
   durations=.12+(total-.36)*lengths/lengths.sum()
   seeds.append(inner);times.append(durations)
 return np.array(seeds,np.float32),np.array(times,np.float32)
def inverse_physical(inner,tail,durations):
 points=np.vstack([inner,tail[0]]).astype(np.float64);lower=np.array([0.,-4.,-2.]);upper=np.array([10.,4.,2.])
 r=np.empty(19,np.float64);scaled=2*(points-lower)/(upper-lower)-1
 if (np.abs(scaled)>=1).any():raise ValueError("waypoint not in invertible decoder interior")
 r[:9]=np.arctanh(scaled).ravel();r[9:12]=np.arctanh(np.asarray(tail[1],float)/np.array([6.,4.,3.]))
 r[12:15]=np.arctanh(np.asarray(tail[2],float)/np.array([6.,6.,4.]))
 total=float(np.sum(np.asarray(durations,np.float64)));s=(total-.8)/1.4
 if not 0<s<1 or np.min(durations)<=.12:raise ValueError("time inverse out of interior")
 r[15]=math.log(s/(1-s));weights=(np.asarray(durations,float)-.12)/(total-.36);r[16:19]=np.log(weights)
 return r.astype(np.float32)
def canonical_teacher(state,scene,inner,tail,durations,mode):
 raw19=inverse_physical(inner,tail,durations)
 # P2A API has a fixed six-query axis. Other five zero rows are math-only neutral probes,
 # NOT expert copies or network predictions and are never stored as teachers.
 raw=torch.zeros(1,6,21);raw[0,0,:19]=torch.from_numpy(raw19)
 with torch.no_grad():out=TrajectoryPostprocessor()(raw,torch.from_numpy(np.asarray(state,np.float32))[None],num_samples=513)
 par=out.parameters;c=out.trajectory.coeffs[0,0].numpy();d=par.durations[0,0].numpy()
 head=par.head_pva[0,0].numpy();end=par.tail_pva[0,0].numpy();ip=par.inner_pts[0,0].numpy()
 check=validate_teacher(c,d,head,end,ip,scene);task=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),np.asarray(state)[6:9])
 if not check["passed"] or not task["passed"]:return None,{"verification":check,"task":task,"raw_roundtrip":True}
 geo=fine_geometry(c,d,scene,CONFIG["signed_witness_points_per_piece"])
 if geo["minimum_signed_surface_clearance_m"]<.03-1e-6:raise AssertionError("independent signed geometry contradicts accepted teacher")
 crossing=None
 # Full sampled crossing position for transparent mode reporting; continuous collision certificate remains authoritative.
 _,curve=sample_curve(c,d,1025);i=int(np.argmin(np.abs(curve[:,0]-scene["crossing_x_m"])));crossing=curve[i].tolist()
 actual="direct"
 if scene["kind"]!="free":
  y,z=curve[i,1],curve[i,2]
  actual=("left" if y>=0 else "right") if abs(y)>=abs(z) else ("up" if z>=0 else "down")
 teacher={"coeffs":c,"durations":d,"head_pva":head,"tail_pva":end,"inner_pts":ip,"raw_physical_19":raw19,
  "p":out.samples.p[0,0].numpy(),"v":out.samples.v[0,0].numpy(),"a":out.samples.a[0,0].numpy(),"jerk":out.samples.jerk[0,0].numpy(),"times":out.samples.times[0,0].numpy(),
  "seed_mode":mode,"actual_mode":actual,"crossing_point_B0_m":crossing,"verification":check,"task":task,"geometry":geo,"raw_roundtrip":True,
  "source":"offline GT multi-start waypoints/time; unchanged MincoS3NU + unchanged P2A full canonical roundtrip"}
 return teacher,{"verification":check,"task":task,"geometry":geo,"raw_roundtrip":True}
def inspect_candidate(c,d,head,tail,inner,state,scene,mode,phase,iteration,idx):
 check=validate_teacher(c,d,head,tail,inner,scene)
 task=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),np.asarray(state)[6:9])
 receipt={"mode":mode,"phase":phase,"iteration":iteration,"seed_index":idx,"passed_before_roundtrip":bool(check["passed"] and task["passed"]),
  "independent_verifier":check,"task":task,"dense_body_collision_evidence":check["dense_surface_clearance_m"]<0,
  "dense_margin_evidence":0<=check["dense_surface_clearance_m"]<.03,
  "coefficient_sha256":hashlib.sha256(c.tobytes()+d.tobytes()).hexdigest()}
 if check["passed"] and task["passed"]:
  t,roundtrip=canonical_teacher(state,scene,inner,tail,d,mode);receipt["canonical_check"]=roundtrip;receipt["accepted"]=t is not None
  return t,receipt
 receipt["accepted"]=False;return None,receipt
def search_mode(state,scene,mode):
 state=np.asarray(state,np.float32);head=np.stack([np.zeros(3,np.float32),state[:3],state[3:6]]);tail=expert_tail(state)
 seed,dt=seeds(state,scene,mode);ht=torch.from_numpy(head);tt=torch.from_numpy(tail)
 with torch.no_grad():coeff=solve(ht,tt,torch.from_numpy(seed),torch.from_numpy(dt)).numpy()
 attempts=[];history=[]
 for i in range(len(seed)):
  teacher,receipt=inspect_candidate(coeff[i],dt[i],head,tail,seed[i],state,scene,mode,"seed",0,i);attempts.append(receipt)
  if teacher is not None:return teacher,attempts,history
 inner=torch.nn.Parameter(torch.from_numpy(seed.copy()));total0=dt.sum(1);z=(total0-.8)/1.4
 total_logit=torch.nn.Parameter(torch.logit(torch.from_numpy(z)));weights=(dt-.12)/(total0[:,None]-.36);fraction_logit=torch.nn.Parameter(torch.from_numpy(np.log(weights)))
 parameters=[inner,total_logit,fraction_logit];optimizer=torch.optim.Adam(parameters,lr=CONFIG["optimizer_lr"]);base=torch.from_numpy(seed)
 for step in range(1,CONFIG["max_optimization_steps"]+1):
  total=.8+1.4*total_logit.sigmoid();durations=.12+(total-.36)[:,None]*fraction_logit.softmax(-1)
  coeff=solve(ht,tt,inner,durations);loss,stats=differentiable_cost(coeff,durations,inner,base,scene)
  optimizer.zero_grad(set_to_none=True);loss.mean().backward()
  finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
  if not finite:
   history.append({"step":step,"error":"nonfinite numerical trajectory gradient","network_parameters_in_optimizer":False});break
  optimizer.step()
  with torch.no_grad():
   cx=scene["crossing_x_m"];inner[:,0,0].clamp_(.4,cx-.15);inner[:,1,0].clamp_(cx+.15,float(tail[0,0])-.20)
   inner[:,:,1].clamp_(-1.5,1.5);inner[:,:,2].clamp_(-1.1,1.1)
   if mode in ["left","right"]:inner[:,:,1].clamp_(min=.46) if mode=="left" else inner[:,:,1].clamp_(max=-.46)
   if mode in ["up","down"]:inner[:,:,2].clamp_(min=.46) if mode=="up" else inner[:,:,2].clamp_(max=-.46)
   if mode=="portal":inner[:,:,1].clamp_(.65,1.03);inner[:,:,2].clamp_(.27,.65)
   total_logit.clamp_(-4,8);fraction_logit.clamp_(-4,4)
  history.append({"step":step,"mean_cost":float(loss.detach().mean()),"per_candidate_cost":loss.detach().tolist(),"finite_trajectory_gradients":True,
   "waypoint_gradient_norm":float(inner.grad.norm()),"time_gradient_norm":float(total_logit.grad.norm()),"network_parameters_in_optimizer":False,
   **{k:v.tolist() for k,v in stats.items()}})
  if step%20==0 or step==CONFIG["max_optimization_steps"]:
   with torch.no_grad():
    total=.8+1.4*total_logit.sigmoid();durations=.12+(total-.36)[:,None]*fraction_logit.softmax(-1)
    cc=solve(ht,tt,inner,durations).numpy();dd=durations.numpy();ip=inner.detach().numpy()
   for i in range(len(seed)):
    teacher,receipt=inspect_candidate(cc[i],dd[i],head,tail,ip[i],state,scene,mode,"optimized",step,i);attempts.append(receipt)
    if teacher is not None:return teacher,attempts,history
 return None,attempts,history
def common_time_distance(a,b):
 end=min(float(a["durations"].astype(float).sum()),float(b["durations"].astype(float).sum()));times=np.linspace(0,end,257)
 def at(t):
  ds=t["durations"].astype(float);ends=np.cumsum(ds);starts=np.r_[0,ends[:-1]];idx=np.minimum(np.searchsorted(ends,times,side="right"),2)
  return np.stack([evaluate(t["coeffs"][i],s-starts[i]) for s,i in zip(times,idx)])
 return float(np.sqrt(np.mean(np.sum((at(a)-at(b))**2,axis=1))))
def generate_scene(spec):
 teachers=[];receipts=[];logs=[];dedup=[]
 for mode in modes_for(spec["kind"]):
  try:t,attempts,history=search_mode(spec["state"],spec["scene"],mode)
  except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
   receipts.append({"mode":mode,"error":repr(e),"classification":"numerical_generation_failure_not_physical_infeasibility"});continue
  receipts.extend(attempts);logs.append({"mode":mode,"history":history})
  if t is not None:
   distances=[common_time_distance(t,other) for other in teachers]
   match=next((i for i,d in enumerate(distances) if d<CONFIG["dedup_common_physical_time_RMS_m"]),None)
   if match is None:teachers.append(t)
   else:dedup.append({"mode":mode,"representative":match,"common_physical_time_distances_m":distances})
 return teachers,{"attempts":receipts,"optimization":logs,"deduplicated":dedup,"accepted_count":len(teachers),"no_expert_means":"unresolved unless separate analytic proof"}
