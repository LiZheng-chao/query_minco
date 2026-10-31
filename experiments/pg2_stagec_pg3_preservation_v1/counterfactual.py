"""Fixed-budget TRAIN-only frozen counterfactual; never publishes a checkpoint."""
import copy,gc,json,time,traceback,resource,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_next_exact_v1.minimal import MinimalEngine
from query_minco.training.pg3_next_exact_v1.proxy import Parameters,proxy,semantic_sha,canonical_gradient_witness
from query_minco.training.pg3_next_exact_v1.projection import project,numerical_attempt
from query_minco.training.pg3_preservation_v1.mechanism import local_direction,witness,retained,superior
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,step_guard_v21
from query_minco.training.pg3_v2_no_corridor.executor_v21 import wrap_pg2
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,scene_from_field
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from experiments.pg3_next_exact_v1.counterfactual import restore_monitor,fixed_mode
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg3_feasible_mode_v1.fulltrain128_evaluation import signature
from experiments.pg3_v2_no_corridor.ablation_v21 import load_context
ROOT=Path('/home/lizheng/query_minco_mainline')
BASE=ROOT/'research_outputs/pg2_stagec_pg3_preservation_v1'
QUAL=ROOT/'research_outputs/pg3_minimal_system_v1/qualification001'
RUN=QUAL/'runs/A42_MINIMAL_MINPAIR128_run001'

def context(folder):
    from query_minco.training.run_spec import source_files
    spec=read(RUN/'RUN_SPEC.json');binding=read(artifact(spec['bindings']))
    code=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=source_files(),allow_uncommitted_sources=True)
    # CPU-only disposable diagnostics. Preserve pre-existing untracked files; no formal qualification claim.
    binding['code']=code
    reference=ROOT/'query_minco/methods/reference.json';require(sha(reference)==binding['historical_reference']['sha256'],'REFERENCE_CONTENT_CHANGED')
    binding['historical_reference']=dict(path=str(reference),sha256=sha(reference));write(folder/'BINDINGS.json',binding)
    ref=dict(path=str((folder/'BINDINGS.json').resolve()),sha256=sha(folder/'BINDINGS.json'))
    prep,ctx=load_context(ref);write(folder/'DATA_QUALIFICATION.json',ctx[3]);write(folder/'SOURCE.json',dict(code=code,tracked_and_untracked_status=subprocess.check_output(['git','status','--porcelain'],text=True),formal_qualification=False))
    cfg=read(artifact(prep['configuration']));return spec,ctx,cfg

def sigs(z,scenes):
    tr=z['output'].trajectory;d=z['output'].parameters.durations
    return [[signature(tr.coeffs[b,q].detach().numpy(),d[b,q].detach().numpy(),scenes[b]) if bool(z['labels']['passed'][b,q] and z['goal_labels'][b,q]) else None for q in range(6)] for b in range(len(scenes))]

def proposal_from_gradient(engine,before,monitor,samples,geo,vector,lr_factor=1.,keep_weight=0.,pg2_only=False):
    restore_monitor(engine,before,monitor)
    z=engine.objective(engine.model,samples,geo,engine.spec['pg2_weights'],TeacherGuidance(),engine.completed)
    if keep_weight:
        value,diag=engine.reference.loss(z['pg2'],samples,z['pg2']['fields'])
        z['terms']['keep']=value;z['weights']['keep']=keep_weight;z['loss']=z['loss']+keep_weight*value
    if pg2_only:
        z['loss']=z['pg2']['loss'];z['terms']['mode']=z['terms']['mode']*0;z['weights']['mode']=0.
    audit,_,total=audit_v21(engine.model,z,False)
    step_guard_v21(audit,engine.config if hasattr(engine,'config') else engine.cfg,streak=monitor['monitor']['streak'],keep_previously_active=monitor['monitor']['keep_active'])
    offset=0
    for p in engine.model.parameters():p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    # Reuse the original executor's actual LR contract, never scale a gradient as an Adam surrogate.
    from query_minco.training.pg3_v2_no_corridor.executor_v21 import lr_at
    for group in engine.opt.param_groups:group['lr']=lr_at(engine.completed)*lr_factor
    torch.nn.utils.clip_grad_norm_(engine.model.parameters(),10.)
    engine.opt.step()
    # Original executor also freezes score rows including AdamW decay.
    with torch.no_grad():
        for n,p in engine.model.named_parameters():
            if n.startswith('trajectory_head.') and p.shape[0]==21:p[19:21]=before['model'][n][19:21]
    d=Parameters(engine.model).values()-vector
    engine.exposure.rollback(engine.completed)
    return d,dict(audit=audit,lr_factor=lr_factor,keep_weight=keep_weight,actual_adamw_step=True,pg2_only=pg2_only)

def one(step,spec,ctx,cfg,protocol,folder):
    folder.mkdir(exist_ok=False);ds,geo,init,_,train,dev=ctx
    path0=RUN/('commits/step%04d'%(step-1));path1=RUN/('commits/step%04d'%step)
    before=torch.load(path0/'checkpoint.pt',map_location='cpu',weights_only=True);after=torch.load(path1/'checkpoint.pt',map_location='cpu',weights_only=True);monitor=read(path0/'MONITOR.json');expected=read(path1/'UPDATE.json')
    archived={str(p):sha(p) for p in [path0/'checkpoint.pt',path1/'checkpoint.pt']}
    e=MinimalEngine(model_from(init),spec,cfg);restore_monitor(e,before,monitor);ids,_=geo.orders[42].batch(step-1);samples=ds.load(ids)
    require(all(s.sample_id in ds.rows for s in samples) and not ({s.sample_id for s in samples}&{item[0].sample_id for item in dev}),'TRAIN_ONLY')
    pack=Parameters(e.model);v=pack.values();rec=e.step(samples,geo);adam=pack.values()-v
    replay=dict(model_exact=state_sha(e.model)==after['model_sha256'],optimizer_exact=semantic_sha(e.opt.state_dict())==semantic_sha(after['optimizer']),exposure_exact=digest(e.exposure.state_dict())==digest(after['exposure']),gradient_witness_exact=digest(canonical_gradient_witness(rec))==digest(canonical_gradient_witness(expected)))
    write(folder/'ADAM_REPLAY.json',replay);require(all(replay.values()),'FROZEN_REPLAY_MISMATCH')
    restore_monitor(e,before,monitor);pack=Parameters(e.model)
    z=e.objective(e.model,samples,geo,spec['pg2_weights'],TeacherGuidance(),step-1);audit,vec,total=audit_v21(e.model,z,False);step_guard_v21(audit,cfg,streak=monitor['monitor']['streak'],keep_previously_active=monitor['monitor']['keep_active'])
    old_state=copy.deepcopy(e.model.state_dict());passed=z['labels']['passed'].detach().numpy();goals=z['goal_labels'].detach().numpy();scenes=[scene_from_field(f) for f in z['pg2']['fields']];before_sigs=sigs(z,scenes);groups=witness(passed,goals,before_sigs);selected=sorted({int(q) for a in z['assignments'] for q in a['selected']});before_mode=fixed_mode(z,z,samples,e.objective)
    module_norms=audit['components']['mode']['parameter_groups'];write(folder/'BEFORE.json',dict(ids=ids,passed=passed.tolist(),goal=goals.tolist(),matched=z['matched'].tolist(),signatures=before_sigs,witness=groups,selected_rows=selected,mode_parameter_groups=module_norms,audit=audit,model=before['model_sha256'],source_checkpoints=archived))
    e.exposure.rollback(step-1)
    methods={}
    def trial(delta):
        e.model.load_state_dict(old_state);pack.set_disposable(v+torch.as_tensor(delta))
        with torch.no_grad():
            zz=wrap_pg2(pg2_forward(e.model,samples,geo,'query_minco_pg2_guided_v2_safe_v1',TeacherGuidance(),spec['pg2_weights']))
            p=zz['labels']['passed'].numpy();g=zz['goal_labels'].numpy();sg=sigs(zz,scenes);valid=bool(zz['output'].trajectory.solve_valid.all())
            mode=fixed_mode(zz,z,samples,e.objective)
        return dict(all_numerical_valid=valid,protected_pass=bool(p[passed].all()),protected_goal_pass=bool(g[passed&goals].all()),mode_witness_retained=retained(groups,p,g,sg),pre_safe_pass_to_fail=int((passed&~p).sum()),pre_fail_to_pass=int((~passed&p).sum()),complete_repairs=int((~(passed&goals)&p&g).sum()),matched_safe_degraded=int((passed&z['matched'].numpy()&~p).sum()),oracle_after=int((p&g).any(-1).sum()),known_modes_after=[len({x for x in row if x is not None}) for row in sg],relative_update_norm=float(np.linalg.norm(delta)/adam.norm()),Mode_loss_delta=mode-before_mode,passed=p.tolist(),goals=g.tolist(),signatures=sg)
    def method(name,direction_fn,alphas,gate):
        start=time.monotonic();cpu=time.process_time();trials=[];extra={}
        try:
            direction,extra=direction_fn()
            require(bool(torch.isfinite(torch.as_tensor(direction)).all()) and float(np.linalg.norm(direction))>0,'NO_FINITE_NONZERO_DIRECTION')
            for alpha in alphas:
                require(time.monotonic()-start<30 and time.process_time()-cpu<30,'METHOD_BUDGET_EXCEEDED')
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4096*1024,'RSS_BUDGET')
                t=time.monotonic();r=numerical_attempt(trial,np.asarray(direction)*alpha);r.update(alpha=alpha,wall_s=time.monotonic()-t)
                r['accepted']=bool(r['all_numerical_valid'] and (r.get('protected_pass') and r.get('protected_goal_pass') if gate=='all_query' else r.get('mode_witness_retained')))
                trials.append(r)
                require(time.monotonic()-start<=30 and time.process_time()-cpu<=30,'METHOD_BUDGET_EXCEEDED')
                if r['accepted']:break
            chosen=next((r for r in trials if r['accepted']),None);result=dict(status='ACCEPTED' if chosen else 'REJECTED',chosen=chosen,trials=trials)
        except Exception as ex:result=dict(status='BLOCKED',reason=str(ex),traceback=traceback.format_exc(),chosen=None,trials=trials)
        result.update(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,extra=extra,gate=gate,trial_cap=6,wall_cpu_ceiling_s=30)
        methods[name]=result;write(folder/(name+'.json'),result);print('CF',step,name,result['status'],flush=True)
    raw=lambda:(adam.numpy(),{})
    alphas=protocol['algorithm']['alphas'] if 'algorithm' in protocol else [1,.5,.25,.125,.0625,.03125]
    method('ADAM_DIRECT',raw,[1.],'mode')
    method('STEPBACK_SAME_WITNESS',raw,alphas,'mode')
    method('STEPBACK_ALL_QUERY',raw,alphas,'all_query')
    def local():
        d,meta=local_direction(pack,adam,selected);return d.numpy(),meta
    method('QUERY_LOCAL_NORM_MATCHED',local,alphas,'mode')
    # All optimizer controls restore identical real BEFORE Adam moments and state.
    def optim(factor=1.,keep=0.,pg2=False):
        d,meta=proposal_from_gradient(e,before,monitor,samples,geo,v,factor,keep,pg2);return d.numpy(),meta
    method('SMALL_LR_ACTUAL_ADAM',lambda:optim(.25),[1.],'mode')
    method('OLD_KEEP_ACTUAL_ADAM',lambda:optim(1.,.012369432948913293),alphas,'mode')
    method('PG2_ONLY_SAME_MOMENTS_ATTRIBUTION',lambda:optim(1.,0.,True),[1.],'mode')
    def qp():
        restore_monitor(e,before,monitor);zz=e.objective(e.model,samples,geo,spec['pg2_weights'],TeacherGuidance(),step-1);pack2=Parameters(e.model);G=[]
        for b,q in zip(*np.nonzero(passed)):
            value,info=proxy(zz['output'].trajectory.coeffs[b,q],zz['output'].parameters.durations[b,q],scenes[b]['obstacles']);g=pack2.gradient(value)
            require(bool(torch.isfinite(g).all()) and float(g.norm())>1e-12,'UNRELIABLE_PROXY');G.append(g.numpy())
        e.exposure.rollback(step-1);d,meta=project(adam.numpy(),np.stack(G) if G else np.zeros((0,len(adam))));return d,meta
    method('EXISTING_QP_SAME_WITNESS',qp,alphas,'mode')
    restore_monitor(e,before,monitor)
    require(state_sha(e.model)==before['model_sha256'] and semantic_sha(e.opt.state_dict())==semantic_sha(before['optimizer']) and digest(e.exposure.state_dict())==digest(before['exposure']),'DISPOSABLE_POLLUTION')
    require(all(sha(Path(p))==h for p,h in archived.items()),'ARCHIVE_MUTATION')
    result=dict(step=step,status='VERIFIED',replay=replay,methods=methods,local_extra_repairs_no_extra_safe_loss=superior(methods['QUERY_LOCAL_NORM_MATCHED']['chosen'],methods['STEPBACK_SAME_WITNESS']['chosen']),isolation='PASS',formal_updates=0)
    write(folder/'RESULT.json',result);return result

def run():
    out=BASE/(sys.argv[1] if len(sys.argv)>1 else 'COUNTERFACTUAL001');out.mkdir(exist_ok=False);spec,ctx,cfg=context(out);protocol=read(BASE/'PREREGISTRATION.json');rows=[]
    # Fixed registered ten chronological TRAIN event steps; no Dev case selection.
    for step in [1,9,13,26,28,30,31,45,92,118]:
        folder=out/('step%04d'%step);rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
        try:rows.append(one(step,spec,ctx,cfg,protocol,folder))
        except Exception as ex:
            folder.mkdir(exist_ok=True);r=dict(step=step,status='BLOCKED',reason=str(ex),traceback=traceback.format_exc());write(folder/'BLOCKED.json',r);rows.append(r);print('CF_BLOCKED',step,str(ex),flush=True)
        finally:torch.set_rng_state(rng);np.random.set_state(np_rng);random.setstate(py_rng);gc.collect()
    write(out/'RESULT.json',dict(cases=rows,completed=sum(r['status']=='VERIFIED' for r in rows),formal_updates=0,decision='NO_GO_FOR_FORMAL_TRAINING',reason='Single-seed correlated frozen pilot cannot qualify a method; per-case hard evidence only.',development_used_for_protection=False))
if __name__=='__main__':run()
