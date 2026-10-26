"""Bounded Pilot32 worker using the shared isolation/checkpoint/optimizer lifecycle."""
import os,sys,time,traceback
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from .io import require,read,write,artifact,digest,sha
from .initialization import configure,head_rows,rows_unchanged
from .recipe import optimizer
from .checkpoints import save_checkpoint
from . import objective
from .datasets.geometry_sidecar import validate_semantics
from .pilot_spec import recipe,lr_at,qualify,authorize
from .run_spec import verify_code
from .losses.set_physical_risk import segment_samples,physical_loss,reduce_set,box_sdf

def initialize(s):
    configure(s)
    if s['device']=='cuda':
        require(torch.cuda.device_count()==1,'ONE_VISIBLE_GPU_REQUIRED')
        free,total=torch.cuda.mem_get_info()
        require(free>=s['resources']['min_free_gpu_mib']*2**20,'GPU_FREE_MEMORY')
        torch.cuda.set_per_process_memory_fraction(min(s['resources']['max_gpu_fraction'],s['resources']['max_gpu_mib']*2**20/total))
        torch.cuda.reset_peak_memory_stats()
    blob=torch.load(str(artifact(s['initialization']['artifact'])),map_location='cpu',weights_only=True)
    m=QueryMincoPlanner();m.load_state_dict(blob['state_dict'],strict=True)
    require(state_sha(m)==s['initialization']['weights_sha256'],'PILOT_INITIAL_WEIGHTS')
    return m.to(s['device']).train()

def resource(s):
    if s['device']=='cpu':return dict(device='cpu',peak_allocated_bytes=0)
    p=torch.cuda.get_device_properties(0);peak=torch.cuda.max_memory_allocated()
    require(max(peak,torch.cuda.max_memory_reserved())<=min(s['resources']['max_gpu_mib']*2**20,p.total_memory*s['resources']['max_gpu_fraction']),'GPU_MEMORY_BUDGET')
    return dict(device='cuda:0',name=p.name,total_memory=p.total_memory,peak_allocated_bytes=peak,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda=torch.version.cuda)

def batch(model,samples,geo,s):
    verify_code(s['code'])
    model.zero_grad(set_to_none=True);cfg=s['recipe'];started=time.perf_counter();before=state_sha(model)
    fields=geo.load(samples)
    require(len(fields)==len(samples),'GEOMETRY_BATCH')
    for field in fields:
        validate_semantics(field.row)  # Provenance is authenticated by geo.load, not this flag.
        require(field.boxes.shape==(len(field.row['obstacles']),2,3),'GEOMETRY_BOX_BINDING')
    certified_empty=[field.row['empty_certified'] for field in fields]
    imit,matches,raw,out,costs=objective.forward(model,samples,'B_set',s['device'])
    p,v,a=segment_samples(out.trajectory,64)
    mask=torch.zeros(raw.shape[:2],dtype=torch.bool,device=raw.device)
    for b,x in enumerate(matches):mask[b,x['queries']]=True
    boxes=[f.boxes.to(raw.device) for f in fields]
    if cfg['arm']=='E0':
        pq=torch.zeros_like(raw[:,:,0]);phys=raw.sum()*0;loss=imit
        gp=torch.zeros_like(raw)
    else:
        pq,_=physical_loss(p,v,a,boxes,geometry_bound=[True]*len(fields),radius=.12,margin=.03,
            eps=.05,beta=cfg['beta'],v_ref=1.,w_max=3.)
        phys=reduce_set(pq,mask,1.,cfg['unmatched_strength']);loss=imit+phys
        gp=torch.autograd.grad(phys,raw,retain_graph=True)[0]
    gi=torch.autograd.grad(imit,raw,retain_graph=True)[0]
    raw.grad=None;out.parameters.durations.retain_grad();out.parameters.durations.grad=None
    loss.backward()
    require(torch.isfinite(loss) and out.trajectory.solve_valid.all(),'PILOT_LOSS_MINCO')
    require(all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters()),'PILOT_MODEL_GRADIENT')
    g=raw.grad
    require(g is not None and torch.isfinite(g).all() and not torch.count_nonzero(g[...,19:21]),'UNSUPERVISED_RAW_GRADIENT')
    require(not torch.count_nonzero(gi[~mask]),'E0_UNMATCHED_DIRECT_GRADIENT')
    if cfg['arm']=='E0':require(not torch.count_nonzero(g[~mask]),'E0_UNMATCHED_DIRECT_GRADIENT')
    dg=out.parameters.durations.grad;require(dg is not None and torch.isfinite(dg).all(),'DURATION_GRADIENT')
    for n,x in model.named_parameters():
        if n.startswith('trajectory_head.') and x.shape[0]==21:require(not torch.count_nonzero(x.grad[19:21]),'HEAD_GRADIENT')
    unsafe=torch.stack([torch.zeros(p.shape[1],dtype=torch.bool,device=p.device) if certified_empty[b]
                        else (box_sdf(p[b],field).min(-1).values<.12)
                        for b,field in enumerate(boxes)])
    if cfg['arm']!='E0':require(not bool((unsafe & (gp.norm(dim=-1)==0)).any()),'UNSAFE_ZERO_PHYSICAL_GRADIENT')
    ratio=float(gp.norm()/gi.norm().clamp_min(1e-30))
    cosine=float((gp*gi).sum()/(gp.norm()*gi.norm()).clamp_min(1e-30))
    arrays={k:x.detach().cpu().numpy() for k,x in dict(raw=raw,coeffs=out.trajectory.coeffs,
        durations=out.parameters.durations,positions=p,velocities=v,time_weights=a,per_query=pq,
        mask=mask,raw_grad=g,physical_raw_grad=gp,imitation_raw_grad=gi,duration_grad=dg).items()}
    arrays.update({'cost_%d'%b:c.detach().cpu().numpy() for b,c in enumerate(costs)})
    record=dict(ids=[x.sample_id for x in samples],certified_empty=certified_empty,
        sampled_collision=unsafe.detach().cpu().tolist(),matches=matches,loss=float(loss.detach()),
        imitation=float(imit.detach()),physical=float(phys.detach()),recipe_sha256=s['recipe_sha256'],
        training_method_id=s['training_method_id'],spec_sha256=digest(s),model_sha_before=before,
        physical_imitation_gradient_ratio=ratio,gradient_cosine=cosine,
        matched_physical_gradient_norm=float(gp[mask].norm()),unmatched_physical_gradient_norm=float(gp[~mask].norm()),
        raw19_21_zero=True,solve_valid=out.trajectory.solve_valid.tolist(),
        matched_counts=mask.sum(-1).tolist(),elapsed_s=time.perf_counter()-started,
        resources=resource(s))
    require(np.isfinite([record[k] for k in ['loss','imitation','physical','physical_imitation_gradient_ratio','gradient_cosine']]).all(),'NONFINITE_RECORD')
    require(state_sha(model)==before,'UNEXPECTED_FORWARD_UPDATE')
    return record,arrays

def persist_batch(run,index,record,arrays):
    p=run/('batch_%04d.npz'%index)
    with p.open('xb') as f:np.savez(f,**arrays)
    record['arrays_sha256']=sha(p)
    write(run/('batch_%04d.json'%index),record)

def preflight(s,run):
    ds,order,_=qualify(s);model=initialize(s);rows=head_rows(model);before=state_sha(model);ids,roles=order.batch(0)
    first,arrays=batch(model,ds.load(ids),ds.geometry,s)
    # Repeat from the same parameters and RNG seed. No optimizer is constructed.
    configure(s)
    second,repeat=batch(model,ds.load(ids),ds.geometry,s)
    require(all(np.array_equal(v,repeat[k]) for k,v in arrays.items()),'DETERMINISTIC_BATCH_REPEAT')
    require(state_sha(model)==before and rows_unchanged(model,rows),'PREFLIGHT_PARAMETERS')
    first.update(step=0,roles=roles,model_sha_after=before,lr=lr_at(0))
    persist_batch(run,0,first,arrays)
    from .pilot_audit import audit_batch
    audit=audit_batch(s,first,arrays,ds.load(ids),ds.geometry,0)
    write(run/'PREFLIGHT.json',dict(passed=True,spec_sha256=digest(s),optimizer_updates=0,
        optimizer_created=False,forward_backward_calls=2,device=s['device'],gradient_gate=dict(passed=True),
        deterministic_repeat=True,initialized_weights_sha256=before,batch_audit=audit,
        environment=read(run/'ENVIRONMENT.json'),resources=resource(s)))

def execute(s,run,authorization,preflight_path):
    # Only a human can supply the CUDA authorization after actual preflight review.
    authorize(s,authorization)
    receipt=read(preflight_path)
    require(receipt['passed'] and receipt['device']=='cuda' and receipt['spec_sha256']==digest(s),'CUDA_PREFLIGHT_REQUIRED')
    require(receipt['environment']['torch']==torch.__version__ and receipt['environment']['executable']==sys.executable,'PREFLIGHT_RUNTIME_DRIFT')
    ds,order,_=qualify(s);m=None;opt=None;completed=0;step_started=False;last_arrays={}
    try:
        m=initialize(s)
        actual=resource(s)
        require(all(actual[k]==receipt['resources'][k] for k in ['name','total_memory','cuda']),'GPU_RUNTIME_DRIFT')
        rows=head_rows(m);opt=optimizer(m);require(not opt.state,'FRESH_OPTIMIZER')
        from .pilot_evaluation import evaluate_model,metric_alarm
        from .pilot_audit import audit_batch
        for group in opt.param_groups:group['lr']=lr_at(0)
        base=evaluate_model(m,s['development']);write(run/'evaluation_step0000.json',base)
        save_checkpoint(run,m,opt,s,0,scheduler_id='pilot_spec.lr_at_v1')
        alarms=[];high_ratio=[]
        for index in range(32):
            ids,roles=order.batch(index)
            record,arrays=batch(m,ds.load(ids),ds.geometry,s);last_arrays=arrays
            high_ratio.append(record['physical_imitation_gradient_ratio']>10)
            require(not(len(high_ratio)>=2 and all(high_ratio[-2:])),'GRADIENT_RATIO_TWO_STEPS')
            audit_batch(s,record,arrays,ds.load(ids),ds.geometry,index)
            for group in opt.param_groups:group['lr']=lr_at(index)
            norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),10));require(np.isfinite(norm),'CLIP_FINITE')
            step_started=True
            opt.step()  # USER-EXECUTED ONLY: never reached by preflight or CPU test.
            completed=index+1;step_started=False
            require(rows_unchanged(m,rows) and all(torch.isfinite(x).all() for x in m.parameters()),'POST_STEP_PARAMETERS')
            record.update(step=completed,lr=lr_at(index),roles=roles,global_gradient_norm_before_clip=norm,
                          model_sha_after=state_sha(m),resources=resource(s))
            persist_batch(run,completed,record,arrays)
            write(run/('update_%04d.json'%completed),dict(completed_optimizer_updates=completed,model_sha=state_sha(m)))
            if completed in [8,16,32]:
                save_checkpoint(run,m,opt,s,completed,scheduler_id='pilot_spec.lr_at_v1')
                ev=evaluate_model(m,s['development']);write(run/('evaluation_step%04d.json'%completed),ev)
                metric_alarm(alarms,ev,base,read(s['development']['path']))
        write(run/'WORKER_RESULT.json',dict(completed_optimizer_updates=completed,status='WORKER_FINISHED_AWAITING_AUDIT',
            formal_evaluations=0,profile='rsps_pilot32_v1',final_weights_sha256=state_sha(m)))
    except BaseException as e:
        snapshot_failure(run,e,s,completed,step_started,locals().get('ids'))
        if m is not None and opt is not None:
            try:
                save_checkpoint(run,m,opt,s,completed,name='FAILED_STATE_NOT_A_PRESET_CHECKPOINT.pt',scheduler_id='pilot_spec.lr_at_v1')
            except BaseException as save_error:
                write(run/'FAILURE_SAVE_ERROR.json',dict(error=repr(save_error)))
        raise

def snapshot_failure(run,error,s,completed=0,partial=False,ids=None):
    """Preserve the actual failed batch when available; never overwrite a run."""
    write(run/'ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),
        completed_optimizer_updates=completed,possible_partial_update=partial,
        ids=ids,spec_sha256=digest(s),no_resume=True))
    captured={};tb=error.__traceback__
    try:
        while tb is not None:
            frame=tb.tb_frame
            if frame.f_code.co_name in ['batch','forward']:
                raw=frame.f_locals.get('raw')
                if isinstance(raw,torch.Tensor):captured['raw']=raw.detach().cpu().numpy()
                out=frame.f_locals.get('out',frame.f_locals.get('output'))
                if out is not None and hasattr(out,'trajectory'):
                    captured['coeffs']=out.trajectory.coeffs.detach().cpu().numpy()
                    captured['durations']=out.trajectory.durations.detach().cpu().numpy()
            tb=tb.tb_next
        if captured:
            path=run/'FAILURE_ARRAYS.npz'
            with path.open('xb') as f:np.savez(f,**captured)
            write(run/'FAILURE_ARRAYS_SHA.json',dict(sha256=sha(path),keys=list(captured)))
    except BaseException as snapshot_error:
        write(run/'FAILURE_SNAPSHOT_ERROR.json',dict(error=repr(snapshot_error)))
