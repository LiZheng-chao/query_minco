"""Shared network/MINCO/matching, independent PG1 objective. CPU zero updates only."""
import time
import numpy as np
import torch
from query_minco.training import objective as reference
from query_minco.training.datasets.inputs import forward_inputs
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from query_minco.training.losses.base_imitation import NoTeacherError
from query_minco.training.losses.set_physical_risk import segment_samples,physical_loss,reduce_set
from query_minco.training.run_spec import source_files
from query_minco.training.io import require,digest
from query_minco.inference.identity import state_sha
from query_minco.trajectory import TrajectoryPostprocessor
from .geometry import authenticated_task_gates
from .losses import analytic_samples,goal_task_loss,dynamic_limit_loss,center_domain_loss
from .spec import recipe

def forward(model,samples,sidecar,method):
 device=next(model.parameters()).device
 cfg=recipe(method);fields,gates=authenticated_task_gates(samples,sidecar)
 if all(len(s.teacher_coeffs)>0 for s in samples):
  imitation,matches,raw,out,costs=reference.forward(model,samples,'B_set',device)
 else:
  inputs=forward_inputs(samples,'normal',device);raw=model(**inputs)
  if raw.requires_grad:raw.retain_grad()
  out=TrajectoryPostprocessor()(raw,inputs['state'],num_samples=32)
  losses=[];matches=[];costs=[]
  for b,s in enumerate(samples):
   if not len(s.teacher_coeffs):
    if gates[b]['goal_enabled']:raise NoTeacherError('PG1_M0_UNKNOWN_IS_NOT_A_NEGATIVE_LABEL')
    losses.append(raw[b].sum()*0);costs.append(raw.new_empty(6,0));q=[];t=[]
   else:
    c,_=cost_matrix(out,s,b);l,qq,tt=match(c,'B_set');losses.append(l);costs.append(c);q=qq.tolist();t=tt.tolist()
   matches.append(dict(sample_id=s.sample_id,queries=q,teachers=t))
  imitation=torch.stack(losses).mean()
 mask=torch.zeros(raw.shape[:2],dtype=torch.bool,device=raw.device)
 for b,m in enumerate(matches):mask[b,m['queries']]=True
 p,v,w=segment_samples(out.trajectory,64) # Preserve exact E2 obstacle arithmetic.
 obs,_=physical_loss(p,v,w,[f.boxes.to(device) for f in fields],geometry_bound=[True]*len(fields),
  radius=cfg['radius'],margin=cfg['margin'],eps=cfg['eps'],beta=cfg['beta'],v_ref=cfg['v_ref'],w_max=cfg['w_max'])
 obstacle=reduce_set(obs,mask,1.,1.) if bool((mask.sum(-1)>0).all()) else (obs.sum(-1)/6).mean()
 samples4=analytic_samples(out.trajectory,64)
 enabled=torch.tensor([g['goal_enabled'] for g in gates],dtype=torch.bool,device=raw.device)
 goal,goal_diag=goal_task_loss(samples4['endpoint'],samples4['terminal_velocity'],torch.stack([s.state[6:9] for s in samples]).to(device),enabled)
 dyn,dyn_terms=dynamic_limit_loss(samples4['v'],samples4['a'],samples4['jerk'],samples4['weights'],cfg['dynamic_limits'])
 domain=center_domain_loss(samples4['p'],samples4['weights'],raw.new_tensor([f.row['flight_center_domain_m'] for f in fields]))
 terms=dict(imitation=imitation,obstacle=obstacle,goal=goal.mean(),dynamic=dyn.mean(),domain=domain.mean())
 total=imitation+obstacle
 for name,key in [('goal','goal_scale'),('dynamic','dynamic_scale'),('domain','domain_scale')]:
  if cfg[key]:total=total+cfg[key]*terms[name] # Exact paired-E2 graph when scales are0.
 return dict(loss=total,terms=terms,per_query=dict(obstacle=obs,goal=goal,dynamic=dyn,domain=domain),
  dynamic_terms=dyn_terms,naive_goal=goal_diag['unmasked'].mean(),gates=gates,goal_enabled=enabled,matched=mask,
  matches=matches,raw=raw,output=out,costs=costs,analytic=samples4,positions=p,velocities=v,time_weights=w,recipe=cfg)

def conflict(a,b,eligible):
 na=a.norm(dim=-1);nb=b.norm(dim=-1);valid=(na>1e-12)&(nb>1e-12)&eligible
 count=int(valid.sum());bad=int((((a*b).sum(-1)<0)&valid).sum())
 return dict(conflicts=bad,nonzero_pairs=count,rate=bad/count if count else None)

def cpu_zero_update(model,samples,sidecar,method):
 require(next(model.parameters()).device.type=='cpu','PG1_CPU_ONLY')
 started=time.perf_counter();cpu=time.thread_time();before=state_sha(model);code_before=source_files();model.zero_grad(set_to_none=True)
 z=forward(model,samples,sidecar,method);raw=z['raw'];out=z['output']
 grads={k:torch.autograd.grad(v,raw,retain_graph=True)[0] for k,v in z['terms'].items()}
 naive_grad=torch.autograd.grad(z['naive_goal'],raw,retain_graph=True)[0]
 raw.grad=None;out.parameters.durations.retain_grad();out.parameters.durations.grad=None
 z['loss'].backward()
 require(raw.grad is not None and torch.isfinite(raw.grad).all() and not torch.count_nonzero(raw.grad[...,19:21]),'PG1_RAW_GRADIENT')
 require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'PG1_MODEL_GRADIENT')
 require(out.parameters.durations.grad is not None and bool(torch.isfinite(out.parameters.durations.grad).all()),'PG1_DURATION_GRADIENT')
 for name,p in model.named_parameters():
  if name.startswith('trajectory_head.') and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),'PG1_HEAD_ROWS')
 require(state_sha(model)==before,'PG1_UPDATE_FORBIDDEN')
 require(source_files()==code_before,'PG1_SOURCE_DRIFT')
 arrays={k:t.detach().cpu().numpy() for k,t in dict(raw=raw,raw_grad=raw.grad,coeffs=out.trajectory.coeffs,
  durations=out.parameters.durations,duration_grad=out.parameters.durations.grad,matched=z['matched'],
  positions=z['positions'],velocities=z['velocities'],time_weights=z['time_weights']).items()}
 for k,t in z['analytic'].items():arrays['analytic_'+k]=t.detach().numpy()
 for k,t in z['per_query'].items():arrays['per_query_'+k]=t.detach().numpy()
 for k,t in grads.items():arrays['grad_'+k]=t.detach().numpy()
 arrays['grad_naive_goal_offline_only']=naive_grad.detach().numpy()
 for b,c in enumerate(z['costs']):arrays['cost_%d'%b]=c.detach().numpy()
 require(all(np.isfinite(a).all() for a in arrays.values()),'PG1_ARRAY_FINITE')
 enabled=z['goal_enabled'][:,None].expand(-1,6);mask=z['matched']
 strata={}
 for count in sorted({len(s.teacher_coeffs) for s in samples}):
  rows=torch.tensor([len(s.teacher_coeffs)==count for s in samples],dtype=torch.bool)
  mm=mask[rows];aa=grads['goal'][rows];bb=grads['obstacle'][rows]
  denom=aa.norm(dim=-1)*bb.norm(dim=-1);valid=denom>1e-12
  strata['M%d'%count]=dict(samples=int(rows.sum()),
   per_query_losses={k:dict(all=float(t[rows].mean()),
    matched=float(t[rows][mm].mean()) if bool(mm.any()) else None,
    unmatched=float(t[rows][~mm].mean()) if bool((~mm).any()) else None) for k,t in z['per_query'].items()},
   gradient_norms={k:dict(matched=float(g[rows][mm].norm()),unmatched=float(g[rows][~mm].norm())) for k,g in grads.items()},
   goal_obstacle_cosine_mean=float(((aa*bb).sum(-1)[valid]/denom[valid]).mean()) if bool(valid.any()) else None)
 record=dict(schema='pg1_cpu_record_v1',mode='CPU_ZERO_UPDATE',method_id=method,recipe=z['recipe'],recipe_sha256=digest(z['recipe']),
  ids=[s.sample_id for s in samples],matches=z['matches'],source_files=source_files(),weights_before=before,weights_after=state_sha(model),
  optimizer_created=False,optimizer_updates=0,total=float(z['loss'].detach()),losses={k:float(t.detach()) for k,t in z['terms'].items()},
  gates=z['gates'],feasibility_mask=z['goal_enabled'].tolist(),feasibility_mask_semantics='true=goal supervised, not proof of reachability',
  goal_loss_masked_count=int((~z['goal_enabled']).sum()),goal_query_masked_count=int((~enabled).sum()),goal_denominator=raw.shape[0]*6,
  naive_goal_offline_only=float(z['naive_goal'].detach()),
  goal_obstacle_conflicts=conflict(grads['goal'],grads['obstacle'],enabled),
  naive_goal_obstacle_conflicts_offline_only=conflict(naive_grad,grads['obstacle'],torch.ones_like(enabled)),
  gradient_norms={k:dict(total=float(g.norm()),matched=float(g[mask].norm()),unmatched=float(g[~mask].norm()),per_query=g.norm(dim=-1).tolist()) for k,g in grads.items()},
  teacher_count_strata=strata,raw19_21_zero=True,time_summary=dict(min_total=float(z['analytic']['total_time'].min()),max_total=float(z['analytic']['total_time'].max()),
  near_upper_count=int((z['analytic']['total_time']>2.19).sum())),
  wall_ms=(time.perf_counter()-started)*1000,thread_cpu_ms=(time.thread_time()-cpu)*1000,
  limitations='sampled proxies, not execution certificate; synthetic records are not real-data qualification')
 return dict(record=record,arrays=arrays)
