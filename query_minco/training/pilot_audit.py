"""Pilot content audit: independent scalar reconstruction, identities and fixed budget."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .io import read,require,sha,artifact,digest
from .pilot_spec import load_spec,qualify,lr_at
from .audit_rsps import numpy_physical
from query_minco.trajectory import TrajectoryPostprocessor
from .losses.imitation import cost_matrix
from .losses.set_physical_risk import segment_samples
from .checkpoints import load_checkpoint
from .recipe import optimizer
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha

def audit_batch(s,record,a,samples,geo,index):
    require(record['ids']==geo.orders[42].batch(index)[0]==[x.sample_id for x in samples],'AUDIT_ORDER')
    require(record['spec_sha256']==digest(s) and record['training_method_id']==s['training_method_id'] and
        record['recipe_sha256']==s['recipe_sha256'],'AUDIT_METHOD')
    require(all(np.isfinite(x).all() for x in a.values()),'AUDIT_FINITE')
    require(a['raw'].dtype==np.float32 and a['raw'].shape==(8,6,21),'AUDIT_RAW')
    g=a['raw_grad'];gi=a['imitation_raw_grad'];gp=a['physical_raw_grad'];mask=np.zeros((8,6),bool);im=[]
    for b,x in enumerate(samples):
        c=a['cost_%d'%b];require(c.shape==(6,len(x.teacher_coeffs)),'AUDIT_COST_SHAPE')
        q,t=linear_sum_assignment(c);logged=record['matches'][b]
        require(logged['queries']==q.tolist() and logged['teachers']==t.tolist() and logged['sample_id']==x.sample_id,'AUDIT_MATCH')
        im.append(c[q,t].mean());mask[b,q]=True
    require(np.array_equal(mask,a['mask']) and not np.count_nonzero(gi[~mask]),'AUDIT_IMITATION_GRADIENT')
    require(not np.count_nonzero(g[...,19:21]) and np.allclose(g,gi+gp,rtol=1e-4,atol=2e-6),'AUDIT_RAW_GRADIENT')
    if s['objective_arm']=='E0':
        require(not np.count_nonzero(g[~mask]) and not np.count_nonzero(gp),'E0_UNMATCHED_DIRECT_GRADIENT')
        phys=0.
    else:
        fields=geo.load(samples)
        pq=numpy_physical(a['positions'],a['velocities'],a['time_weights'],[x.boxes.numpy() for x in fields],s['recipe'])
        require(np.allclose(pq,a['per_query'],rtol=2e-5,atol=2e-7),'AUDIT_NUMPY_PHYSICAL')
        weights=np.where(mask,1.,s['recipe']['unmatched_strength'])
        phys=np.mean((pq*weights).sum(-1)/weights.sum(-1))
    require(np.isclose(record['imitation'],np.mean(im),rtol=2e-5,atol=2e-7) and
        np.isclose(record['physical'],phys,rtol=2e-5,atol=2e-7) and
        np.isclose(record['loss'],np.mean(im)+phys,rtol=2e-5,atol=2e-7),'AUDIT_TOTAL_LOSS')
    # Raw->MINCO->cost independently reconstructed on CPU. GPU tolerance is numerical, not a safety threshold.
    with torch.no_grad():
        out=TrajectoryPostprocessor()(torch.from_numpy(a['raw']),torch.stack([x.state for x in samples]),num_samples=32)
        require(np.allclose(out.trajectory.coeffs.numpy(),a['coeffs'],rtol=2e-4,atol=2e-4),'AUDIT_RAW_MINCO')
        pp,vv,aa=segment_samples(out.trajectory,64)
        for key,val in [('positions',pp),('velocities',vv),('time_weights',aa)]:
            require(np.allclose(val.numpy(),a[key],rtol=2e-4,atol=2e-4),'AUDIT_TRAJECTORY_'+key)
        for b,x in enumerate(samples):
            cost,_=cost_matrix(out,x,b)
            require(np.allclose(cost.numpy(),a['cost_%d'%b],rtol=2e-4,atol=2e-5),'AUDIT_RAW_IMITATION')
    ratio=float(np.linalg.norm(gp)/max(np.linalg.norm(gi),1e-30))
    cosine=float(np.sum(gp*gi)/max(np.linalg.norm(gp)*np.linalg.norm(gi),1e-30))
    require(np.isclose(record['physical_imitation_gradient_ratio'],ratio,rtol=1e-5,atol=1e-8) and
            np.isclose(record['gradient_cosine'],cosine,rtol=1e-5,atol=1e-8),'AUDIT_GRADIENT_DIAGNOSTICS')
    require(a['duration_grad'].shape==(8,6,3),'AUDIT_DURATION')
    require(record['solve_valid']==[[True]*6]*8 and record['raw19_21_zero'] is True,'AUDIT_FINITE_GATES')
    require(np.isclose(record['matched_physical_gradient_norm'],np.linalg.norm(gp[mask]),rtol=1e-5,atol=1e-8) and
        np.isclose(record['unmatched_physical_gradient_norm'],np.linalg.norm(gp[~mask]),rtol=1e-5,atol=1e-8),'AUDIT_GROUP_GRADIENTS')
    return dict(passed=True,imitation=float(np.mean(im)),physical=float(phys),total=float(np.mean(im)+phys))

def audit_run(directory,candidate=False):
    from pathlib import Path
    run=Path(directory);s=read(run/'RUN_SPEC.json');ds,order,_=qualify(s)
    require(not any((run/x).exists() for x in ['ERROR.json','STOP.json','LAUNCH_ERROR.json']),'FAILED_RUN')
    require(read(run/'PROCESS_EXIT.json')['returncode']==0 and read(run/'ISOLATION.json')['passed'],'ISOLATION_EXIT')
    require(read(run/'PROTECTION_BEFORE.json')['files']==read(run/'PROTECTION_AFTER.json')['files'],'PROTECTION_DRIFT')
    require(read(run/'WORKER_RESULT.json')['completed_optimizer_updates']==32,'UPDATE_BUDGET')
    require(len(list(run.glob('batch_*.npz')))==32 and len(list(run.glob('update_*.json')))==32,'EXACT_RECORD_BUDGET')
    for step in range(1,33):
        rec=read(run/('batch_%04d.json'%step));p=run/('batch_%04d.npz'%step)
        require(rec['arrays_sha256']==sha(p) and rec['step']==step and rec['lr']==lr_at(step-1) and rec['roles']==order.batch(step-1)[1],'STEP_SHA_LR')
        require(read(run/('update_%04d.json'%step))['completed_optimizer_updates']==step,'UPDATE_CURSOR')
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        audit_batch(s,rec,a,ds.load(order.batch(step-1)[0]),ds.geometry,step-1)
    require(sorted(x.name for x in run.glob('checkpoint_step*.pt'))==['checkpoint_step%04d.pt'%i for i in [0,8,16,32]],'CHECKPOINT_SET')
    initial_rows=None
    for step in [0,8,16,32]:
        p=run/('checkpoint_step%04d.pt'%step);receipt=read(str(p)+'.json');blob=torch.load(str(p),map_location='cpu',weights_only=True)
        require(receipt['sha256']==sha(p) and receipt['complete'] and blob['spec']==s and
            blob['completed_optimizer_updates']==blob['order_cursor']==step and blob['scheduler']=='pilot_spec.lr_at_v1','CHECKPOINT_CONTENT')
        require(all(int(x['step'])==step and all(torch.isfinite(v).all() for v in x.values() if isinstance(v,torch.Tensor))
                    for x in blob['optimizer_state_dict']['state'].values()),'OPTIMIZER_STEP_COUNT')
        for group in blob['optimizer_state_dict']['param_groups']:
            require(group['lr']==lr_at(max(0,step-1)) and group['betas']==(.9,.999) and
                    group['eps']==1e-8 and group['weight_decay']==0.,'OPTIMIZER_RECIPE')
        model=QueryMincoPlanner();opt=optimizer(model)
        load_checkpoint(p,model,opt,s)
        require(not opt.state if step==0 else len(opt.state)==len(list(model.parameters())),'OPTIMIZER_STATE_COMPLETE')
        rows={k:v[19:21] for k,v in model.state_dict().items() if k.startswith('trajectory_head.') and v.shape[0]==21}
        if initial_rows is None:initial_rows=rows
        require(all(torch.equal(v,initial_rows[k]) for k,v in rows.items()),'CHECKPOINT_HEAD_ROWS')
        ev=read(run/('evaluation_step%04d.json'%step))
        from .pilot_evaluation import evaluate_model
        reconstructed=evaluate_model(model,s['development'])
        require(ev==reconstructed,'DEVELOPMENT_EVALUATION_RECONSTRUCTION')
        if step==32:require(read(run/'WORKER_RESULT.json')['final_weights_sha256']==state_sha(model),'FINAL_MODEL_SHA')
    if not candidate:
        seal=read(run/'RUN_SEAL.json')['files']
        require({str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='RUN_SEAL.json'}==seal,'RUN_SEAL')
    return dict(passed=True,updates=32,profile='rsps_pilot32_v1',formal_training=False)
