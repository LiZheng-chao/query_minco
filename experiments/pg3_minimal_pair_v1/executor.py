"""No mathematical changes: native Engine/MinimalEngine, bound authorization, sealed commits."""
import argparse,copy,time,traceback
from pathlib import Path
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import verify_code
from query_minco.training.initialization import configure
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_next_exact_v1.minimal import MinimalEngine,WEIGHTS,validate as minimal_validate
from query_minco.training.pg3_v2_no_corridor.executor_v21 import Engine,validate_protocol,publish_checkpoint,restore_checkpoint,inspect_chain
from experiments.pg3_v2_no_corridor.formal_v21 import resources,state_fingerprint
from experiments.pg3_v2_no_corridor.qualification_v21 import snapshot
from experiments.pg3_v2_no_corridor.ablation_v21 import load_context
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg3_feasible_mode_v1.fulltrain128_evaluation import evaluate_set,inference_measure
from query_minco.training.pg1_v2_r12.evaluation import evaluate as open8
from query_minco.training.pilot_evaluation import metric_alarm
from .metrics import supplement
SCOPE='PG3_MINIMAL_PAIRED128_V1'
def validate(s):
    require(s['schema']=='pg3_minimal_pair_executor_v1' and s['arm'] in ('A','MINIMAL'),'MINPAIR_ID')
    if s['arm']=='MINIMAL':minimal_validate(s)
    else:
        validate_protocol(s);require(s['method_id']=='query_minco_pg2_guided_v2_safe_v1' and s['weights']=={k:0. for k in WEIGHTS},'MINPAIR_A_ID')
    require(s['stop_mode']=='legacy','MINPAIR_BRAKING')
    require(s['formal_scope']==SCOPE and s['formal_authorized'] is False,'MINPAIR_SCOPE')
def authorize(s,a):
    validate(s)
    require(a.get('approved') is True and a.get('spec_sha256')==digest(s) and a.get('authorized_by')
        and a.get('max_updates')==s['execution']['limit'] and a.get('resources')==s['execution']
        and a.get('scope')==(SCOPE if s['execution']['limit']==128 else 'PG3_MINIMAL_SHORT4_V1'),'MINPAIR_AUTHORIZATION')
def qualified(s):
    q=read(s['minimal_qualification']);require(q['status']=='QUALIFIED' and q['code']==s['code'] and q['sources']==s['sources'],'MINPAIR_QUALIFICATION')
    require(q['formal_spec_sha256'][s['arm']]==digest(s),'MINPAIR_SPEC_BINDING')
    for k in ('cpu0','tests','gradient_audit','short','baseline_short'):artifact(q[k])
    t=read(artifact(q['tests']));require(t['code_commit']==s['code']['git_commit'] and t['failed']==0 and t['not_run']==0 and t['passed']==t['total'],'MINPAIR_TESTS')
    require(read(artifact(q['cpu0']))['status']=='PASS','MINPAIR_CPU0')
def runtime_contract(s):
    verify_code(s['code']);require(snapshot()==s['sources'],'MINPAIR_SOURCE_DRIFT')
    prep,context=load_context(s['bindings'])
    template=read('/home/lizheng/query_minco_mainline/research_outputs/pg3_v21_paired128_v1/runs/A42_A_PG3V21_formal_run001/RUN_SPEC.json')
    for k in ('initialization','order','configuration','pg2_weights','seed','threads','batch_size','updates','optimizer','schedule','precision','device'):
        require(s[k]==template[k],'MINPAIR_FROZEN:'+k)
    old=read(artifact(template['bindings']));new=read(artifact(s['bindings']))
    require(all(new[k]==v for k,v in old.items() if k not in ('code','historical_reference')) and new['code']==s['code'],'MINPAIR_DATA_BINDING')
    require(new['historical_reference']['sha256']==old['historical_reference']['sha256'],'MINPAIR_REFERENCE_BINDING')
    return prep,context

def run(spec_path,authorization,resume=False,pause_after=None):
    s=read(spec_path);a=read(authorization);authorize(s,a) # fail before creating any outputs
    if s['execution']['limit']==128:qualified(s)
    prep,(ds,geo,init,q,train,dev)=runtime_contract(s);limit=s['execution']['limit']
    require(pause_after is None or (limit==4 and 0<pause_after<4),'MINPAIR_PAUSE_SCOPE')
    run=Path(s['output_root'])/s['run_id']
    if not resume:
        run.mkdir(parents=True,exist_ok=False);write(run/'RUN_SPEC.json',s);write(run/'AUTHORIZATION.json',a)
    else:
        require(limit==4 and read(run/'RUN_SPEC.json')==s,'MINPAIR_RESUME_SCOPE')
        prev=sorted(run.glob('PROCESS_*_RESULT.json'));require(prev and read(prev[-1])['status']=='PAUSED','MINPAIR_FAILED_RUN_NOT_RESUMABLE')
    phase=len(list(run.glob('PROCESS_*_START.json')));write(run/('PROCESS_%02d_START.json'%phase),dict(resume=resume))
    configure(dict(seed=42,threads=1,device='cpu'));factory=MinimalEngine if s['arm']=='MINIMAL' else Engine
    e=factory(model_from(init),s,read(artifact(prep['configuration'])));start=time.monotonic();last=None;status='FAILED';error=None
    try:
        resources(run,start,s)
        if resume:last=restore_checkpoint(run,e)
        else:
            targets={}
            items_by_split=(('training384',train),('development192',dev)) if limit==128 else (('training8',train[:8]),)
            for split,items in items_by_split:
                _,t=evaluate_set(e.model,items,run/'evaluation',split,0);targets.update(t);supplement(run/'evaluation',split,0)
            write(run/'INITIAL_TARGETS.json',targets)
            if limit==128:write(run/'INITIAL_OPEN_DEV8.json',open8(e.model,read(artifact(s['bindings']))['development']))
            last=publish_checkpoint(run,e,dict(completed=0))
        order=read(artifact(s['order']))
        while e.completed<limit:
            resources(run,start,s);require(snapshot()==s['sources'],'MINPAIR_RUNTIME_SOURCE_DRIFT')
            ids,roles=geo.orders[42].batch(e.completed);require(ids==order['batches'][e.completed],'MINPAIR_ORDER')
            rec=e.step(ds.load(ids),geo);rec['roles']=roles;alarm=None
            if limit==128 and e.completed in (32,64,96,128):
                cur=open8(e.model,read(artifact(s['bindings']))['development']);history=e.monitor.setdefault('evaluation_alarm_history',[])
                try:metric_alarm(history,cur,read(run/'INITIAL_OPEN_DEV8.json'),{})
                except ValueError as ex:alarm=str(ex)
                rec.update(open_dev8=cur,checkpoint_alarm=alarm)
            resources(run,start,s);last=publish_checkpoint(run,e,rec,sha(last/'COMMIT.json'))
            print('MINPAIR_UPDATE',s['arm'],e.completed,rec['metrics']['pass_to_fail'],flush=True)
            if alarm:raise ValueError(alarm)
            if pause_after==e.completed:status='PAUSED';break
        else:
            status='COMPLETED'
            items_by_split=(('training384',train),('development192',dev)) if limit==128 else (('training8',train[:8]),)
            for split,items in items_by_split:
                resources(run,start,s);evaluate_set(e.model,items,run/'evaluation',split,e.completed,read(run/'INITIAL_TARGETS.json'));supplement(run/'evaluation',split,e.completed)
            write(run/'INFERENCE.json',inference_measure(e.model,train[0][0]));resources(run,start,s)
        inspect_chain(run,s)
    except BaseException:
        status='FAILED';error=traceback.format_exc();e.exposure.rollback(e.completed)
        write(run/('REJECTED_%02d.json'%phase),dict(error=error,phase=e.phase,last_committed=None if last is None else str(last),audit=e.last_audit))
    result=dict(status=status,completed=e.completed,formal_updates=e.completed if limit==128 else 0,diagnostic_updates=e.completed if limit==4 else 0,last_committed=str(last),error=error,wall_s=time.monotonic()-start)
    write(run/('PROCESS_%02d_RESULT.json'%phase),result);require(status!='FAILED','MINPAIR_EXECUTION_FAILED:'+str(error));return result

def compare(base):
    rows={}
    for arm in ('A','MINIMAL'):
        paths={k:base/'short'/('A42_'+arm+'_'+k+'_run001') for k in ('u','r')};specs={k:read(p/'RUN_SPEC.json') for k,p in paths.items()}
        last={k:inspect_chain(p,specs[k]) for k,p in paths.items()};fps={k:state_fingerprint(p/'checkpoint.pt') for k,p in last.items()}
        require(len(set(fps.values()))==1 and read(last['u']/'MONITOR.json')==read(last['r']/'MONITOR.json'),'MINPAIR_RESUME_DIFFERENCE')
        updates=[read(paths['u']/'commits'/('step%04d'%i)/'UPDATE.json') for i in range(1,5)]
        for i,u in enumerate(updates,1):require(u==read(paths['r']/'commits'/('step%04d'%i)/'UPDATE.json'),'MINPAIR_UPDATE_TRACE')
        rows[arm]=dict(status='PASS',fingerprints=fps,metrics=[u['metrics'] for u in updates],gradient_ratios=[u['audit']['ratios'] for u in updates],optimizer_directions=[u['delta_cosine_negative'] for u in updates],formal_training='NOT_RUN')
    require(read(paths['u']/'RUN_SPEC.json')['initialization']==read(base/'short/A42_A_u_run001/RUN_SPEC.json')['initialization'],'MINPAIR_INITIALIZATION')
    require(sha(base/'short/A42_A_u_run001/evaluation/training8_step0000.npz')==sha(base/'short/A42_MINIMAL_u_run001/evaluation/training8_step0000.npz'),'MINPAIR_INITIAL_PREDICTIONS')
    write(base/'SHORT_PAIR_RESULT.json',dict(status='PASS',arms=rows,scope='four diagnostic updates; not formal performance'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--authorization',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--pause-after',type=int);a=p.parse_args();run(a.spec,a.authorization,a.resume,a.pause_after)
