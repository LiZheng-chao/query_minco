"""Disposable true-Adam replay and fixed-budget parameter-space alternatives."""
import copy,json,time,gc,random
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,require
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_v2_no_corridor.executor_v21 import Engine
from query_minco.training.pg3_v2_no_corridor.ablation_v21 import AblationEngine
from query_minco.training.pg3_v2_no_corridor.revision import load_v21_checkpoint
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,step_guard_v21
from query_minco.training.pg3_next_exact_v1.proxy import Parameters,proxy,semantic_sha,canonical_gradient_witness
from query_minco.training.pg3_next_exact_v1.projection import project,ALPHAS,numerical_attempt
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg3_v2_no_corridor.executor_v21 import wrap_pg2
from query_minco.training.pg3_feasible_mode_v1.losses import mode_query
from query_minco.evaluation.signed_geometry import fine_geometry
from query_minco.training.pg2_guided_v1.geometry import scene_from_field
from experiments.pg3_feasible_mode_v1.common import model_from
from .common import runpath,csvwrite
from experiments.pg3_v2_no_corridor.formal_v21 import resources
ID='pg3_next_disposable_adam_cone_counterfactual_v1'
CASES=tuple((arm,step) for arm in ('A','M','G','K') for step in (1,92,118))

def restore_monitor(engine,before,monitor):
    engine.model.load_state_dict(before['model']);engine.opt.load_state_dict(copy.deepcopy(before['optimizer']));engine.opt.zero_grad(set_to_none=True);engine.exposure.load_state_dict(before['exposure']);engine.completed=monitor['completed'];engine.monitor=copy.deepcopy(monitor['monitor']);torch.set_rng_state(before['torch_rng'])
    engine.validate_state()
def fixed_mode(z,before_z,samples,obj):
    if not before_z['weights']['mode']:return None
    zero=z['raw'].sum()*0;values=[];p=z['output'].parameters
    for b,a in enumerate(before_z['assignments']):
        bank=obj.bank.get(samples[b],before_z['pg2']['fields'][b]);vals=[mode_query(p.inner_pts[b,q],p.tail_pva[b,q],p.durations[b,q],bank['teachers'][a['route_teacher_id'][q]],obj.bank.cfg)[0] for q in a['selected']]
        values.append(torch.stack(vals).mean() if vals else zero)
    return float(torch.stack(values).mean().detach())
def one_case(folder,arm,step,context,config):
    folder.mkdir(exist_ok=False);ds,geo,init,qualified,train,dev=context;run=runpath(arm);s=read(run/'RUN_SPEC.json');path0=run/('commits/step%04d'%(step-1));path1=run/('commits/step%04d'%step)
    before=torch.load(path0/'checkpoint.pt',map_location='cpu',weights_only=True);after=torch.load(path1/'checkpoint.pt',map_location='cpu',weights_only=True);monitor=read(path0/'MONITOR.json');expected=read(path1/'UPDATE.json');archive_sha={str(p):sha(p) for p in (path0/'checkpoint.pt',path1/'checkpoint.pt')}
    rng0=torch.get_rng_state().clone();numpy0=np.random.get_state();python0=random.getstate();start=time.monotonic()
    engine=(AblationEngine if arm in ('M','G') else Engine)(model_from(init),s,config)
    restore_monitor(engine,before,monitor);ids,_=geo.orders[42].batch(step-1);samples=ds.load(ids)
    initial_model_sha=state_sha(engine.model);initial_optimizer_sha=semantic_sha(before['optimizer']);initial_exposure_sha=digest(before['exposure']);packed=Parameters(engine.model);vector_before=packed.values().clone()
    replay=engine.step(samples,geo);actual_optimizer_sha=semantic_sha(engine.opt.state_dict())
    expected_optimizer_sha=semantic_sha(after['optimizer']);model_exact=state_sha(engine.model)==after['model_sha256'];optimizer_exact=actual_optimizer_sha==expected_optimizer_sha
    witness_expected=digest(canonical_gradient_witness(expected));witness_actual=digest(canonical_gradient_witness(replay));witness_exact=witness_expected==witness_actual
    adam=packed.values()-vector_before
    expected_delta=torch.cat([(after['model'][n].double()-before['model'][n].double())[m] for (n,p),m in zip(packed.named,packed.parts)])
    replay_check=dict(model_exact=model_exact,optimizer_exact=optimizer_exact,exposure_exact=digest(engine.exposure.state_dict())==digest(after['exposure']),delta_digest_exact=semantic_sha(adam)==semantic_sha(expected_delta),gradient_evidence_digest_exact=witness_exact,gradient_evidence_sha256=witness_actual,historical_gradient_evidence_sha256=witness_expected,gradient_evidence_semantics='historically stored full parameter Gram + actual norm + ratios + scalar loss; historical raw gradient-vector SHA was not logged, and is NOT claimed',gradient_vector_digest_available=False,optimizer_sha256=actual_optimizer_sha,model_sha256=state_sha(engine.model),delta_sha256=semantic_sha(adam),optimizer_steps=int(step),source_checkpoint=str(path0/'checkpoint.pt'))
    write(folder/'ADAM_REPLAY.json',replay_check)
    require(all(replay_check[k] for k in ('model_exact','optimizer_exact','exposure_exact','delta_digest_exact','gradient_evidence_digest_exact')),'BLOCKED_REPLAY_MISMATCH')
    # Restore disposable state exactly BEFORE proposal; subsequent alternatives never optimizer.step or exposure commit.
    restore_monitor(engine,before,monitor);packed=Parameters(engine.model);require(semantic_sha(engine.opt.state_dict())==initial_optimizer_sha,'NEXT_OPTIMIZER_RESTORE')
    z=engine.objective(engine.model,samples,geo,s['pg2_weights'],TeacherGuidance(),step-1) if engine.objective else wrap_pg2(pg2_forward(engine.model,samples,geo,s['method_id'],TeacherGuidance(),s['pg2_weights']))
    audit,_,_=audit_v21(engine.model,z,False)
    if engine.objective:step_guard_v21(audit,config,streak=monitor['monitor']['streak'],keep_previously_active=monitor['monitor']['keep_active'])
    else:require(audit['ratios']['R_PG2']<=10,'NEXT_A_GRADIENT_GATE')
    baseline_mode=fixed_mode(z,z,samples,engine.objective) if engine.objective else None;baseline_geom=float(z['terms']['geom'].detach()) if z['weights']['geom'] else None
    protected=z['labels']['passed'].detach().clone();goal_before=z['goal_labels'].detach().clone();numeric_before=z['output'].trajectory.solve_valid.detach();fields=z['pg2']['fields'];scenes=[scene_from_field(f) for f in fields]
    constraints=[];metadata=[];initial_clear={};G=[]
    for b,sample in enumerate(samples):
        for q in range(6):
            if not bool(protected[b,q]):continue
            value,info=proxy(z['output'].trajectory.coeffs[b,q],z['output'].parameters.durations[b,q],scenes[b]['obstacles']);g=packed.gradient(value)
            require(bool(torch.isfinite(g).all()) and float(g.norm())>1e-12,'NEXT_UNRELIABLE_PROTECTED_PROXY')
            vmax=z['labels']['checks'][b][q].get('max_speed_mps',3.);info.update(batch=b,sample_id=sample.sample_id,query=q,gradient_norm=float(g.norm()),gradient_sha256=semantic_sha(g),sampling_error_bound_m=3.*info['maximum_time_gap_s']/2,proxy_value_actual_fp32_m=float(value.detach()),proxy_float32_vs64_error_m=abs(float(value.detach())-info['h_m']))
            metadata.append(info);G.append(g.numpy());initial_clear[b,q]=info['h_m']+.03
    G=np.stack(G) if G else np.zeros((0,len(adam)));require(len(G)==int(protected.sum()),'NEXT_UNPROTECTED_SAFE_CANDIDATE')
    projected,qp=project(adam.numpy(),G);write(folder/'PROTECTION.json',dict(parameter_order=packed.order,protected=metadata,QP=qp,selection='before continuous Hard PASS only; no GT score logit',time_method='4097 nodes per piece at fixed physical seconds + exact exterior squared-distance stationary/face-partition roots',certificate=False))
    with (folder/'DIRECTIONS.npz').open('xb') as f:np.savez_compressed(f,d_Adam=adam.numpy(),d_projected=projected,gram=G@G.T,original_directional_change=G@adam.numpy(),projected_directional_change=G@projected)
    # Remove pending transaction on a disposable objective only; archived exposure stays untouched.
    if engine.objective:engine.exposure.rollback(step-1)
    before_state=copy.deepcopy(engine.model.state_dict());base_obj=engine.objective
    def trial(delta):
        resources(folder,start,s);tick=time.perf_counter();engine.model.load_state_dict(before_state);packed.set_disposable(vector_before+torch.as_tensor(delta));require(all(bool(torch.isfinite(p).all()) for p in engine.model.parameters()),'NEXT_TRIAL_NONFINITE_PARAMETERS')
        torch.set_rng_state(before['torch_rng']);pred_start=time.perf_counter()
        try:
            zz=base_obj(engine.model,samples,geo,s['pg2_weights'],TeacherGuidance(),step-1) if base_obj else wrap_pg2(pg2_forward(engine.model,samples,geo,s['method_id'],TeacherGuidance(),s['pg2_weights']))
            passed=zz['labels']['passed'].detach();goals=zz['goal_labels'].detach();valid=zz['output'].trajectory.solve_valid.detach();changes=[]
            for b,q in initial_clear:
                tr=zz['output'].trajectory;clear=fine_geometry(tr.coeffs[b,q].detach().numpy(),tr.durations[b,q].detach().numpy(),scenes[b])['minimum_signed_surface_clearance_m'];changes.append(clear-initial_clear[b,q])
            mode=fixed_mode(zz,z,samples,base_obj) if base_obj else None
            geom=float(zz['terms']['geom'].detach()) if zz['weights']['geom'] else None
            return dict(protected_pass=bool(passed[protected].all()),protected_goal_pass=bool(goals[protected].all()),all_numerical_valid=bool(valid.all()),pre_safe_pass_to_fail=int((protected&~passed).sum()),pre_fail_to_pass=int((~protected&passed).sum()),goal_after=int(goals.sum()),oracle_after=int((passed&goals).any(-1).sum()),protected_min_clearance_change_m=min(changes) if changes else None,relative_update_norm=float(np.linalg.norm(delta)/np.linalg.norm(adam.numpy())),cosine_to_Adam=float(np.dot(delta,adam.numpy())/(np.linalg.norm(delta)*np.linalg.norm(adam.numpy()))) if np.linalg.norm(delta)>0 else None,Mode_loss_delta=None if mode is None else mode-baseline_mode,Geom_loss_delta=None if geom is None else geom-baseline_geom,Mode_loss_before_fixed_queries=baseline_mode,validator_runtime_s=time.perf_counter()-pred_start,total_runtime_s=time.perf_counter()-tick,original_pre_step_gradient_gate='PASS',formal_optimizer_state_equivalent=False)
        finally:
            if base_obj:engine.exposure.rollback(step-1)
    results={}
    for method,direction in (('ADAM_DIRECT',adam.numpy()),('STEP_BACKTRACK',adam.numpy()),('QUERY_QP_BACKTRACK',projected)):
        trials=[]
        # Equal actual evaluation count: six independent copies per method. Direct Adam repeats alpha1.
        for alpha in ((1.,)*6 if method=='ADAM_DIRECT' else ALPHAS):
            row=numerical_attempt(trial,alpha*direction);row.update(alpha=alpha)
            if not row['all_numerical_valid']:row['relative_update_norm']=float(np.linalg.norm(alpha*direction)/np.linalg.norm(adam.numpy()))
            trials.append(row)
        accepted=next((r for r in trials if r['protected_pass'] and r['protected_goal_pass'] and r['all_numerical_valid']),None)
        if method=='ADAM_DIRECT':accepted=trials[0] if trials[0]['protected_pass'] and trials[0]['protected_goal_pass'] and trials[0]['all_numerical_valid'] else None
        results[method]=dict(status='ACCEPTED' if accepted else 'REJECTED',chosen_alpha=None if accepted is None else accepted['alpha'],chosen=accepted,trials=trials,evaluation_count=6,progress_not_zero=bool(accepted and accepted['relative_update_norm']>1e-6),solver_runtime_s=qp['runtime_s'] if method=='QUERY_QP_BACKTRACK' else 0.)
        print('CF_RESULT',arm,step,method,results[method]['status'],results[method]['chosen_alpha'],flush=True)
    engine.model.load_state_dict(before_state);require(state_sha(engine.model)==initial_model_sha and semantic_sha(engine.opt.state_dict())==initial_optimizer_sha and digest(engine.exposure.state_dict())==initial_exposure_sha,'NEXT_DISPOSABLE_STATE_POLLUTION')
    require(all(sha(Path(p))==h for p,h in archive_sha.items()),'NEXT_ARCHIVE_MUTATED')
    torch.set_rng_state(rng0);np.random.set_state(numpy0);random.setstate(python0)
    require(torch.equal(torch.get_rng_state(),rng0),'NEXT_RNG_NOT_RESTORED')
    result=dict(status='VERIFIED',arm=arm,step=step,ids=ids,protected_count=int(protected.sum()),original_pre_safe_pass_to_fail=expected['metrics']['pass_to_fail'],baseline_mode=baseline_mode,baseline_geom=baseline_geom,methods=results,keep_counterfactual='NOT_COMPARABLE: historical K vs G changes objective/state; no same-state same-loss Keep alternative is claimed',QP=qp,isolation='PASS',no_formal_checkpoint=True,no_original_exposure_or_spec_modified=True,wall_s=time.monotonic()-start)
    require(result['wall_s']<1800,'NEXT_COUNTERFACTUAL_RESOURCE_WALL')
    write(folder/'RESULT.json',result);return result

def run(out,context,config):
    folder=out/'COUNTERFACTUAL';folder.mkdir(exist_ok=False);write(folder/'PREREGISTRATION.json',dict(id=ID,cases=[dict(arm=a,step=s) for a,s in CASES],training_only=True,alphas=ALPHAS,epsilon=0,trial_evaluations_per_method=6,requires_actual_adam=True,formal_training=False))
    results=[];blocked=[];isolation=[]
    for arm,step in CASES:
        case=folder/(arm+'_step%04d'%step);archive_paths=[runpath(arm)/('commits/step%04d/checkpoint.pt'%i) for i in (step-1,step)];archive_before={str(p):sha(p) for p in archive_paths};case_rng=torch.get_rng_state().clone();case_np=np.random.get_state();case_py=random.getstate()
        try:results.append(one_case(case,arm,step,context,config))
        except Exception as e:
            import traceback
            write(case/'BLOCKED.json',dict(status='BLOCKED',reason=str(e),traceback=traceback.format_exc()));blocked.append(dict(arm=arm,step=step,reason=str(e)));print('CF_BLOCKED',arm,step,str(e)[:300],flush=True)
        torch.set_rng_state(case_rng);np.random.set_state(case_np);random.setstate(case_py);isolated=all(sha(Path(p))==h for p,h in archive_before.items()) and torch.equal(torch.get_rng_state(),case_rng);isolation.append(dict(arm=arm,step=step,status='PASS' if isolated else 'FAILED'));require(isolated,'NEXT_FAILED_CASE_ARCHIVE_OR_RNG_POLLUTION');gc.collect()
    replay_files=list(folder.glob('*/ADAM_REPLAY.json'));replays=[read(p) for p in replay_files]
    all_exact=len(replays)==len(CASES) and all(all(r[k] for k in ('model_exact','optimizer_exact','exposure_exact','delta_digest_exact','gradient_evidence_digest_exact')) for r in replays)
    write(folder/'REPLAY_SUMMARY.json',dict(all_replays_exact=all_exact,cases=len(replays),raw_gradient_vector_digest='NOT_AVAILABLE_IN_HISTORICAL_LOGS',gradient_witness='stored full Gram,norm,loss,ratios digest + exact model/Adam moments/delta'))
    write(folder/'ISOLATION.json',dict(status='PASS' if all(r['status']=='PASS' for r in isolation) else 'FAILED',cases=isolation,completed=len(results),blocked_methods=blocked))
    # Conservative no-go until superiority in retained repair signal at equal evaluation cost is evidenced.
    better=[]
    for r in results:
        p=r['methods']['QUERY_QP_BACKTRACK'];s=r['methods']['STEP_BACKTRACK']
        if not (p['chosen'] and p['progress_not_zero']):continue
        pc=p['chosen'];sc=s['chosen'];improves=False
        if sc is None:improves=pc['pre_fail_to_pass']>0 or (pc['Mode_loss_delta'] is not None and pc['Mode_loss_delta']<0)
        elif pc['pre_fail_to_pass']>=sc['pre_fail_to_pass']:
            improves=pc['pre_fail_to_pass']>sc['pre_fail_to_pass'] or (pc['Mode_loss_delta'] is not None and sc['Mode_loss_delta'] is not None and pc['Mode_loss_delta']<min(0.,sc['Mode_loss_delta'])-1e-9)
        if improves:better.append(dict(arm=r['arm'],step=r['step']))
    decision='NO_GO_FOR_METHOD_IMPLEMENTATION' if blocked or not better else 'POTENTIAL_REQUIRES_TRANSACTION_DESIGN'
    write(folder/'RESULT.json',dict(status='VERIFIED' if not blocked else 'BLOCKED',method_decision=decision,completed=len(results),blocked=blocked,better_repair_signal_cases=better,results=results,formal_updates=0,development_used_for_protection=False))
