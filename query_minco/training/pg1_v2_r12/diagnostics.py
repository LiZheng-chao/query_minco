"""CPU0 and pre-step math shared by future pilot; NEVER creates optimizer."""
import time,math
import numpy as np
import torch
from query_minco.training.io import require
from query_minco.training.pg1.audit import numpy_samples,numpy_terms
from query_minco.training.audit_rsps import numpy_physical
from query_minco.training.pg1_v2_r1.losses import weighted_upper_tail
from query_minco.training.losses.set_physical_risk import box_sdf
from .objective import forward
def flat(gs,pp):
 return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g,p in zip(gs,pp)])
def cosine(a,b):
 n=float(a.norm()*b.norm())
 return float(a.dot(b))/n if n else None
def ratio_guard(im,physical,streak):
 require(math.isfinite(im) and math.isfinite(physical) and im>=0 and physical>=0,'R12_GRAD_NORM_FINITE')
 ratio=physical/im if im else (None if physical==0 else 1e30)
 streak=streak+1 if ratio is not None and ratio>10 else 0
 require(streak<2,'TWO_STEP_GRADIENT_RATIO_GT10')
 return ratio,streak
def numpy_tail(positions,weights,boxes):
 result=[]
 for p,w,bb in zip(positions,weights,boxes):
  if not len(bb):result.append(np.zeros(6));continue
  qq=np.abs(p[...,None,:]-(bb[:,0]+bb[:,1])/2)-(bb[:,1]-bb[:,0])/2
  sdf=(np.linalg.norm(np.maximum(qq,0),axis=-1)+np.minimum(qq.max(-1),0)).min(-1)
  risk=.05*np.logaddexp(0,-(sdf-.12-.03)/.05)
  mass=w/w.sum(-1,keepdims=True)
  # Independent Python greedy fractional mass, not Torch sort/cumsum reuse.
  row=[]
  for v,a in zip(risk,mass):
   remaining=.1;total=0.
   for i in sorted(range(len(v)),key=lambda j:(-float(v[j]),j)):
    take=min(remaining,float(a[i]));total+=take*float(v[i]);remaining-=take
    if remaining<=1e-15:break
   row.append(total/.1)
  result.append(row)
 return np.asarray(result)
def prepare_gradients(model,samples,geo,method,streak=0,groups=False):
 start=time.monotonic();model.zero_grad(set_to_none=True)
 z=forward(model,samples,geo,method);raw=z['raw'];pp=list(model.parameters());cfg=z['recipe']
 require(raw.dtype==z['loss'].dtype==z['output'].trajectory.coeffs.dtype==torch.float32,'R12_FP32')
 terms=z['terms'];rg={};pg={}
 for k,v in terms.items():
  rg[k]=torch.autograd.grad(v,raw,retain_graph=True)[0].detach()
  pg[k]=flat(torch.autograd.grad(v,pp,retain_graph=True),pp)
  require(torch.isfinite(rg[k]).all() and torch.isfinite(pg[k]).all() and not torch.count_nonzero(rg[k][...,19:21]),'R12_FINITE_COMPONENT')
 physical=pg['obstacle']+sum(cfg[k+'_scale']*pg[k] for k in ('goal','dynamic','domain'))
 if cfg['tail_scale']:physical=physical+cfg['tail_scale']*pg['tail']
 ratio,streak=ratio_guard(float(pg['imitation'].norm()),float(physical.norm()),streak)
 grad=torch.autograd.grad(z['loss'],pp,retain_graph=True)
 require(all(g is not None and bool(torch.isfinite(g).all()) for g in grad),'R12_ALL_PARAMS_CONNECTED_FINITE')
 for p,g in zip(pp,grad):p.grad=g.detach().clone()
 total=flat(grad,pp);require(torch.isfinite(total).all(),'R12_TOTAL_FINITE')
 gc,gd=torch.autograd.grad(z['loss'],[z['output'].trajectory.coeffs,z['output'].parameters.durations],retain_graph=True)
 require(gc is not None and gd is not None and torch.isfinite(gc).all() and torch.isfinite(gd).all(),'R12_TOTAL_COEFF_DURATION_FINITE')
 gt=torch.autograd.grad(terms['tail'],[z['output'].trajectory.coeffs,z['output'].parameters.durations],retain_graph=True)
 require(all(torch.isfinite(x).all() for x in gt),'R12_TAIL_COEFF_DURATION')
 rawtotal=torch.autograd.grad(z['loss'],raw,retain_graph=True)[0]
 require(torch.isfinite(rawtotal).all() and not torch.count_nonzero(rawtotal[...,19:21]),'R12_SCORE_ROWS')
 for name,p in model.named_parameters():
  if name.startswith('trajectory_head.') and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),'R12_HEAD_ROWS')
 fields=geo.load(samples)
 c=z['output'].trajectory.coeffs.detach().numpy();d=z['output'].parameters.durations.detach().numpy()
 ana=numpy_samples(c,d)
 per=numpy_terms(ana,np.stack([s.state[6:9].numpy() for s in samples]),z['goal_enabled'].numpy(),
   [f.row['flight_center_domain_m'] for f in fields],cfg['dynamic_limits'])
 per['obstacle']=numpy_physical(z['positions'].detach().numpy(),z['velocities'].detach().numpy(),z['time_weights'].detach().numpy(),[f.boxes.numpy() for f in fields],cfg)
 per['tail']=numpy_tail(ana['p'],ana['weights'],[f.boxes.numpy() for f in fields])
 for k,x in per.items():require(np.allclose(x,z['per_query'][k].detach().numpy(),rtol=8e-4,atol=8e-4),'R12_NUMPY_'+k)
 imloss=np.mean([float(cst[m['queries'],m['teachers']].detach().mean()) for cst,m in zip(z['costs'],z['matches'])])
 require(np.isclose(imloss,float(terms['imitation'].detach()),rtol=8e-4,atol=8e-4),'R12_NUMPY_IMITATION')
 total_np=imloss+float(per['obstacle'].mean())+sum(cfg[k+'_scale']*float(per[k].mean()) for k in ('goal','dynamic','domain'))
 if cfg['tail_scale']:total_np=total_np+cfg['tail_scale']*float(per['tail'].mean())
 with torch.no_grad():
  from query_minco.trajectory import TrajectoryPostprocessor
  rebuilt=TrajectoryPostprocessor()(raw.detach().cpu(),torch.stack([x.state for x in samples]),num_samples=32)
  require(np.allclose(rebuilt.trajectory.coeffs.numpy(),c,rtol=2e-4,atol=2e-4),'EXEC_RAW_MINCO')
 require(np.isclose(total_np,float(z['loss'].detach()),rtol=8e-4,atol=8e-4),'R12_NUMPY_TOTAL')
 group_records={}
 if groups:
  mm=z['matched']
  for k in terms:
   gm=flat(torch.autograd.grad(raw,pp,grad_outputs=rg[k]*mm[...,None],retain_graph=True,allow_unused=True),pp)
   gu=flat(torch.autograd.grad(raw,pp,grad_outputs=rg[k]*(~mm)[...,None],retain_graph=True,allow_unused=True),pp)
   require(torch.allclose(gm+gu,pg[k],rtol=3e-4,atol=2e-6),'R12_GROUP_SUM')
   group_records[k]=dict(matched_parameter_norm=float(gm.norm()),unmatched_parameter_norm=float(gu.norm()),
     matched_raw_norm=float(rg[k][mm].norm()),unmatched_raw_norm=float(rg[k][~mm].norm()))
 record=dict(method=method,loss=float(z['loss'].detach()),losses={k:float(v.detach()) for k,v in terms.items()},
  matches=z['matches'],M=[len(s.teacher_coeffs) for s in samples],ids=[s.sample_id for s in samples],
  physical_imitation_gradient_ratio=ratio,ratio_alarm_streak=streak,
  global_grad_norm_before_clip=float(total.norm()),numpy_reconstruction=True,raw19_21_zero=True,
  score_rows_zero=True,goal_masked_count=int((~z['goal_enabled']).sum()),
  parameter_component_norms={k:float(g.norm()) for k,g in pg.items()},
  component_cosines={k:dict(imitation=cosine(g,pg['imitation']),obstacle=cosine(g,pg['obstacle'])) for k,g in pg.items()},
  parameter_groups=group_records,
  raw_time15_19={k:[float(g[...,j].norm()) for j in range(15,19)] for k,g in rg.items()},
  tail_coeff_grad_norm=float(gt[0].norm()),tail_duration_grad_norm=float(gt[1].norm()),
  matched_queries=int(z['matched'].sum()),unmatched_queries=int((~z['matched']).sum()),
  duration_gradient_max=float(gd.abs().max()),all_original_numeric_guards=True,
  pre_step_wall_s=time.monotonic()-start)
 arrays={k:z[k].detach().numpy() for k in ['raw','positions','velocities','time_weights']}
 arrays.update(coeffs=c,durations=d,matched=z['matched'].numpy(),raw_grad=rawtotal.detach().numpy())
 for k,g in rg.items():arrays['grad_'+k]=g.numpy()
 for k,v in z['per_query'].items():arrays['per_query_'+k]=v.detach().numpy()
 for k,v in z['dynamic_terms'].items():arrays['dynamic_limit_'+k]=v.detach().numpy()
 for b,cost in enumerate(z['costs']):arrays['cost_%d'%b]=cost.detach().numpy()
 require(all(np.isfinite(v).all() for v in arrays.values()),'R12_ARRAY_FINITE')
 return record,arrays,z,pg
