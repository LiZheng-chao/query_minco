"""Independent NumPy loss/polynomial reconstruction; original hard verifier reused."""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from query_minco.training.io import require,digest
from query_minco.training.run_spec import source_files
from query_minco.training.audit_rsps import numpy_physical
from query_minco.training.losses.imitation import cost_matrix
from query_minco.evaluation.goal import task_metrics
from query_minco.evaluation.goal_config import CONFIG
from query_minco.evaluation.validation import validate_teacher,evaluate
from query_minco.trajectory import TrajectoryPostprocessor
from .geometry import authenticated_task_gates
from .spec import recipe

def numpy_samples(c,dt,n=64):
    batch=[]
    for b in range(len(c)):
        queries=[]
        for q in range(6):
            values=[]
            for order in range(4):
                pieces=[]
                for cf,T in zip(c[b,q],dt[b,q]):
                    times=np.linspace(0,float(T),n)
                    dc=np.stack([cf[:,k].astype(float)*math.factorial(k)/math.factorial(k-order) for k in range(order,6)],-1)
                    pieces.append(np.stack([np.polynomial.polynomial.polyval(times,axis) for axis in dc],-1))
                values.append(np.concatenate(pieces))
            trap=np.ones(n);trap[[0,-1]]=.5
            weights=np.concatenate([float(T)*trap/(n-1)/float(dt[b,q].sum(dtype=float)) for T in dt[b,q]])
            end=evaluate(c[b,q,-1],float(dt[b,q,-1]))
            vel=evaluate(c[b,q,-1],float(dt[b,q,-1]),1)
            queries.append(dict(zip(('p','v','a','jerk','weights','endpoint','terminal_velocity'),values+[weights,end,vel])))
        batch.append(queries)
    return {k:np.array([[x[k] for x in row] for row in batch]) for k in batch[0][0]}

def numpy_terms(values,goals,enabled,bounds,limits=(3,8,100)):
    p=values['endpoint'];v=values['terminal_velocity'];goal=[]
    for b,g in enumerate(np.asarray(goals,float)):
        d=np.linalg.norm(g);require(d>0,'PG1_NUMPY_GOAL')
        u=g/d;remaining=np.linalg.norm(p[b]-g,axis=-1)
        if d<=CONFIG['near_distance_threshold_m']:
            z=(np.maximum(remaining/.15-1,0)**2+np.maximum(np.linalg.norm(v[b],axis=-1)/.25-1,0)**2)
        else:
            progress=p[b]@u;reduction=d-remaining;cos=progress/np.maximum(np.linalg.norm(p[b],axis=-1),1e-12)
            side=np.linalg.norm(p[b]-progress[:,None]*u,axis=-1)
            z=np.maximum(1-reduction,0)**2+np.maximum(1-progress,0)**2+np.maximum((.5-cos)/.5,0)**2+np.maximum((side-.8)/.8,0)**2
        goal.append(z*enabled[b])
    w=values['weights'];w=w/w.sum(-1,keepdims=True)
    dynamic=sum((w*np.maximum(np.linalg.norm(values[k],axis=-1)/limit-1,0)**2).sum(-1) for k,limit in zip(('v','a','jerk'),limits))
    bounds=np.asarray(bounds,float);lo=bounds[:,0,None,None];hi=bounds[:,1,None,None]
    violations=np.maximum((lo-values['p'])/(hi-lo),0)**2+np.maximum((values['p']-hi)/(hi-lo),0)**2
    domain=(w*violations.sum(-1)).sum(-1)
    return dict(goal=np.array(goal),dynamic=dynamic,domain=domain)

def strict_diagnostics(value,empty,path='',null_paths=None):
    """Export only legitimate positive infinite empty-clearance sentinels."""
    if null_paths is None:null_paths=[]
    if isinstance(value,dict):
        return {k:strict_diagnostics(v,empty,path+'.'+k,null_paths) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [strict_diagnostics(v,empty,path+'[%d]'%i,null_paths) for i,v in enumerate(value)]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):
        allowed=path.split('.')[-1] in ('surface_clearance_lower_bound_m','dense_surface_clearance_m')
        require(empty and allowed and value>0,'PG1_NONFINITE_EXPORT')
        null_paths.append(path);return None
    return value

def _norms(g,mask):
    return dict(total=float(np.linalg.norm(g)),matched=float(np.linalg.norm(g[mask])),unmatched=float(np.linalg.norm(g[~mask])),
                per_query=np.linalg.norm(g,axis=-1).tolist())
def _conflict(a,b,enabled):
    valid=(np.linalg.norm(a,axis=-1)>1e-12)&(np.linalg.norm(b,axis=-1)>1e-12)&enabled
    n=int(valid.sum());bad=int((((a*b).sum(-1)<0)&valid).sum())
    return dict(conflicts=bad,nonzero_pairs=n,rate=bad/n if n else None)
def _global_path(c,d,times):
    ends=np.cumsum(d);starts=np.r_[0,ends[:-1]]
    index=np.minimum(np.searchsorted(ends,times,side='right'),2)
    return np.stack([evaluate(c[i],t-starts[i]) for i,t in zip(index,times)])

def audit_cpu(record,a,samples,sidecar):
    require(record['schema']=='pg1_cpu_record_v1' and record['mode']=='CPU_ZERO_UPDATE' and
            record['optimizer_updates']==0 and record['optimizer_created'] is False,'PG1_AUDIT_MODE')
    cfg=recipe(record['method_id']);require(record['recipe']==cfg and record['recipe_sha256']==digest(cfg),'PG1_AUDIT_RECIPE')
    require(record['source_files']==source_files() and record['weights_before']==record['weights_after'] and record['raw19_21_zero'] is True,'PG1_AUDIT_SOURCE_WEIGHTS')
    require(record['ids']==[s.sample_id for s in samples] and all(np.isfinite(x).all() for x in a.values()),'PG1_AUDIT_INPUT')
    fields,gates=authenticated_task_gates(samples,sidecar)
    require(record['gates']==gates,'PG1_AUDIT_PROOF')
    enabled=np.array([g['goal_enabled'] for g in gates]);B=len(samples)
    require(record['feasibility_mask']==enabled.tolist() and record['goal_loss_masked_count']==int((~enabled).sum()) and
            record['goal_query_masked_count']==int((~enabled).sum())*6 and record['goal_denominator']==B*6,'PG1_AUDIT_MASK')
    require(a['raw'].shape==(B,6,21) and a['raw'].dtype==np.float32,'PG1_AUDIT_RAW')
    with torch.no_grad():
        out=TrajectoryPostprocessor()(torch.from_numpy(a['raw']),torch.stack([s.state for s in samples]),num_samples=32)
        require(np.array_equal(out.trajectory.coeffs.numpy(),a['coeffs']) and np.array_equal(out.parameters.durations.numpy(),a['durations']),'PG1_AUDIT_MINCO')
        mask=np.zeros((B,6),bool);im=[]
        for b,s in enumerate(samples):
            if len(s.teacher_coeffs):
                c,_=cost_matrix(out,s,b);require(np.array_equal(c.numpy(),a['cost_%d'%b]),'PG1_AUDIT_COST')
                q,t=linear_sum_assignment(c.numpy());im.append(float(c[q,t].mean()));mask[b,q]=True
            else:
                require(not enabled[b] and a['cost_%d'%b].shape==(6,0),'PG1_AUDIT_M0_PROOF')
                q=np.array([],int);t=q;im.append(0.)
            require(record['matches'][b]['queries']==q.tolist() and record['matches'][b]['teachers']==t.tolist(),'PG1_AUDIT_MATCH')
    require(np.array_equal(mask,a['matched']),'PG1_AUDIT_MATCH_MASK')
    values=numpy_samples(a['coeffs'],a['durations'])
    for key in ('p','v','a','jerk','weights','endpoint','terminal_velocity'):
        require(np.allclose(values[key],a['analytic_'+key],rtol=5e-4,atol=5e-4),'PG1_NUMPY_POLY_'+key)
    # Obstacle reconstruction uses the original sampling arrays, themselves checked against NumPy coefficients.
    for key,source in [('positions','p'),('velocities','v'),('time_weights','weights')]:
        require(np.allclose(a[key],values[source],rtol=5e-4,atol=5e-4),'PG1_AUDIT_OLD_SAMPLER_'+key)
    per=numpy_terms(values,np.stack([s.state[6:9].numpy() for s in samples]),enabled,
                    [f.row['flight_center_domain_m'] for f in fields],cfg['dynamic_limits'])
    per['obstacle']=numpy_physical(a['positions'],a['velocities'],a['time_weights'],[f.boxes.numpy() for f in fields],cfg)
    for k,x in per.items():require(np.allclose(x,a['per_query_'+k],rtol=8e-4,atol=8e-4),'PG1_NUMPY_LOSS_'+k)
    losses={k:float(v.mean()) for k,v in per.items()};losses['imitation']=float(np.mean(im))
    total=losses['imitation']+losses['obstacle']+sum(cfg[k+'_scale']*losses[k] for k in ('goal','dynamic','domain'))
    for k,v in losses.items():require(np.isclose(v,record['losses'][k],rtol=8e-4,atol=8e-4),'PG1_AUDIT_SCALAR_'+k)
    require(np.isclose(total,record['total'],rtol=8e-4,atol=8e-4),'PG1_AUDIT_TOTAL')
    expected=a['grad_imitation']+a['grad_obstacle']
    for k in ('goal','dynamic','domain'):expected=expected+cfg[k+'_scale']*a['grad_'+k]
    # Separate backward passes accumulate FP32 MINCO adjoints in a different order.
    # Record their residual; validate every gradient by exact independent graph replay instead.
    sum_residual=float(np.max(np.abs(expected-a['raw_grad'])))
    require(not np.count_nonzero(a['raw_grad'][...,19:21]) and
            not np.count_nonzero(a['grad_imitation'][~mask]),'PG1_AUDIT_GRADIENT')
    from .objective import forward
    class RawReplay(torch.nn.Module):
        def __init__(self):
            super().__init__();self.raw=torch.nn.Parameter(torch.from_numpy(a['raw'].copy()))
        def forward(self,**unused):return self.raw
    replay=RawReplay();rebuilt=forward(replay,samples,sidecar,record['method_id'])
    for key,term in rebuilt['terms'].items():
        part=torch.autograd.grad(term,rebuilt['raw'],retain_graph=True)[0].detach().numpy()
        require(np.array_equal(part,a['grad_'+key]),'PG1_AUDIT_REPLAY_COMPONENT_'+key)
    naive=torch.autograd.grad(rebuilt['naive_goal'],rebuilt['raw'],retain_graph=True)[0].detach().numpy()
    require(np.array_equal(naive,a['grad_naive_goal_offline_only']),'PG1_AUDIT_NAIVE_GRADIENT')
    gradient=torch.autograd.grad(rebuilt['loss'],rebuilt['raw'])[0].detach().numpy()
    require(np.array_equal(gradient,a['raw_grad']),'PG1_AUDIT_REPLAY_GRADIENT')
    require(a['duration_grad'].shape==(B,6,3),'PG1_AUDIT_TIME_GRADIENT')
    for k in record['losses']:
        g=a['grad_'+k];require(not np.count_nonzero(g[...,19:21]),'PG1_AUDIT_COMPONENT_SCORE')
        actual=_norms(g,mask)
        for part in actual:require(np.allclose(actual[part],record['gradient_norms'][k][part],rtol=1e-5,atol=1e-7),'PG1_AUDIT_GRADIENT_LOG')
    counts=sorted({len(s.teacher_coeffs) for s in samples})
    require(set(record['teacher_count_strata'])=={'M%d'%n for n in counts},'PG1_AUDIT_STRATA')
    for count in counts:
        rows=np.array([len(s.teacher_coeffs)==count for s in samples]);mm=mask[rows];rr=record['teacher_count_strata']['M%d'%count]
        require(rr['samples']==int(rows.sum()),'PG1_AUDIT_STRATUM_COUNT')
        for key in ('obstacle','goal','dynamic','domain'):
            x=a['per_query_'+key][rows];actual={'all':float(x.mean()),'matched':float(x[mm].mean()) if mm.any() else None,
                'unmatched':float(x[~mm].mean()) if (~mm).any() else None}
            for part,v in actual.items():
                logged=rr['per_query_losses'][key][part]
                require(logged is None if v is None else logged is not None and np.isclose(v,logged,rtol=1e-5,atol=1e-7),'PG1_AUDIT_STRATUM_LOSS')
        for key in record['losses']:
            x=a['grad_'+key][rows]
            for part,selection in [('matched',mm),('unmatched',~mm)]:
                require(np.isclose(np.linalg.norm(x[selection]),rr['gradient_norms'][key][part],rtol=1e-5,atol=1e-7),'PG1_AUDIT_STRATUM_GRADIENT')
    require(record['goal_obstacle_conflicts']==_conflict(a['grad_goal'],a['grad_obstacle'],np.broadcast_to(enabled[:,None],(B,6))),'PG1_AUDIT_CONFLICT')
    hard=[];rms=[];nulls=[];goals=[]
    for b,s in enumerate(samples):
        scene=dict(obstacles=fields[b].row['obstacles'],flight_center_domain_m=fields[b].row['flight_center_domain_m'])
        checks=[];gm=[]
        for q in range(6):
            cert=validate_teacher(a['coeffs'][b,q],a['durations'][b,q],out.parameters.head_pva[b,q].numpy(),
                 out.parameters.tail_pva[b,q].numpy(),out.parameters.inner_pts[b,q].numpy(),scene)
            checks.append(strict_diagnostics(cert,gates[b]['certified_empty'],'hard[%d][%d]'%(b,q),nulls))
            gm.append(task_metrics(values['endpoint'][b,q],values['terminal_velocity'][b,q],s.state[6:9].numpy()))
        hard.append(checks);goals.append(gm)
        times=np.linspace(0,float(a['durations'][b].sum(-1).min()),257)
        paths=[_global_path(a['coeffs'][b,q],a['durations'][b,q],times) for q in range(6)]
        rms.append([float(np.sqrt(np.mean(np.sum((paths[i]-paths[j])**2,-1)))) for i in range(6) for j in range(i+1,6)])
    physical=[[c['passed'] for c in row] for row in hard];goal_pass=[[g['passed'] for g in row] for row in goals]
    return dict(schema='pg1_cpu_audit_v1',passed=True,losses=losses,total=total,hard=hard,goal_metrics=goals,
        per_sample_physical_queries=np.sum(physical,axis=1).tolist(),
        complete_oracle=[bool(any(p and g for p,g in zip(pp,gg))) for pp,gg in zip(physical,goal_pass)],
        pairwise_common_physical_time_RMS_m=rms,selector_at1=None,selector_reason='NO_CALIBRATED_SELECTOR',
        feasible_mode_count=None,mode_reason='NO_FROZEN_MODE_CLASSIFIER_BOUND; RMS is not homotopy',
        empty_clearance_null_paths=nulls,gradient_sum_FP32_max_residual=sum_residual,
        gradient_audit='exact full/component graph replay plus norms/proof; FP32 sum residual diagnostic; independent FD separate',
        limits='original Teacher3/8/100; F3 not relaxed; sampled proxies not safety proof')

def audit_parameter_gradients(model,samples,sidecar,method):
    """Fresh-network full backward and component norms; no optimizer/update."""
    import copy
    from .objective import forward
    from query_minco.inference.identity import state_sha
    before=state_sha(model);replica=copy.deepcopy(model);replica.zero_grad(set_to_none=True)
    rebuilt=forward(replica,samples,sidecar,method);params=list(replica.parameters())
    component={name:torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(term,params,retain_graph=True)]) for name,term in rebuilt['terms'].items()}
    rebuilt['loss'].backward()
    arrays={};names=[];finite=True
    for index,((name,original),(other_name,other)) in enumerate(zip(model.named_parameters(),replica.named_parameters())):
        require(name==other_name and original.grad is not None and other.grad is not None and torch.equal(original.grad,other.grad),'PG1_PARAMETER_GRADIENT_REPLAY '+name)
        require(torch.isfinite(original.grad).all(),'PG1_PARAMETER_GRADIENT_FINITE')
        if name.startswith('trajectory_head.') and original.shape[0]==21:
            require(not torch.count_nonzero(original.grad[19:21]),'PG1_PARAMETER_SCORE_ROWS')
        key='parameter_grad_%03d'%index;arrays[key]=original.grad.detach().numpy().copy()
        names.append(dict(key=key,name=name,shape=list(original.shape),norm=float(original.grad.norm())))
    cfg=recipe(method)
    phys=component['obstacle']+sum(cfg[k+'_scale']*component[k] for k in ('goal','dynamic','domain'))
    denom=float(component['imitation'].norm());ratio=float(phys.norm())/denom if denom else None
    require(state_sha(model)==before==state_sha(replica),'PG1_PARAMETER_REPLAY_UPDATED_WEIGHTS')
    return dict(passed=True,names=names,total_norm=float(np.sqrt(sum(np.square(x.astype(float)).sum() for x in arrays.values()))),
        component_norms={k:float(v.norm()) for k,v in component.items()},weighted_non_imitation_norm=float(phys.norm()),
        weighted_non_imitation_over_imitation=ratio,ratio_denominator=denom,finite=True,score_rows_zero=True,
        validation='full-network independent backward bit-identical per parameter; component parameter gradients measured on replay'),arrays
