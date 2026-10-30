"""Qualified fixed S/M/G ablations; sealed A/K reuse and explicit formal authority."""
import argparse,copy,os,subprocess,time,traceback
from pathlib import Path
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import verify_code,source_files,ROOT
from query_minco.training.initialization import configure
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_v2_no_corridor.ablation_v21 import *
from query_minco.training.pg3_v2_no_corridor.executor_v21 import publish_checkpoint,restore_checkpoint,inspect_chain
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,step_guard_v21,make_keep_probe_v21
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from experiments.pg3_feasible_mode_v1.fulltrain128 import load_items
from experiments.pg3_feasible_mode_v1.fulltrain128_evaluation import evaluate_set,inference_measure
from experiments.pg3_feasible_mode_v1.common import model_from
from .qualification_v21 import snapshot,ref
from .formal_v21 import resources,state_fingerprint
from query_minco.training.pg1_v2_r12.evaluation import evaluate as open8
from query_minco.training.pilot_evaluation import metric_alarm
HISTORY=Path('/home/lizheng/query_minco_mainline/research_outputs/pg3_v21_paired128_v1')
HIST_PREP=HISTORY/'_preparation/head002'
def load_context(binding):
    configure(dict(seed=42,threads=1,device='cpu'));prep=read(HIST_PREP/'PREPARATION.json');prep=dict(prep,bindings=binding);old=read(artifact(prep['historical_cpu0']))
    return prep,load_items(dict(old,bindings=prep['bindings']))
def source_identity():
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True);require(not status,'ABL_CLEAN_SOURCE_REQUIRED')
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
def bind_context(folder):
    original=read(artifact(read(HIST_PREP/'PREPARATION.json')['bindings']));original['code']=source_identity()
    original['historical_reference']=ref(ROOT/'query_minco/methods/reference.json');write(folder/'BINDINGS.json',original)
    return ref(folder/'BINDINGS.json')
def reuse_contract():
    specs={a:read(HISTORY/'runs'/('A42_'+a+'_PG3V21_formal_run001')/'RUN_SPEC.json') for a in ('A','D')}
    for key in ('initialization','bindings','order','configuration','pg2_weights','seed','threads','batch_size','updates','optimizer','schedule','precision','device'):
        require(specs['A'][key]==specs['D'][key],'ABL_HIST_NOT_PAIRED:'+key)
    require(specs['D']['weights']==FROZEN,'ABL_HIST_WEIGHTS');current=snapshot()
    require(all(current.get(k)==v for k,v in specs['D']['sources'].items()),'ABL_EXISTING_SOURCE_CHANGED')
    # Check sealed checkpoint/evaluation/update members before any reuse.
    for a in specs:
        run=HISTORY/'runs'/('A42_'+a+'_PG3V21_formal_run001');seal=read(run/'RUN_SEAL.json')
        require(all(sha(run/k)==v for k,v in seal['files'].items()),'ABL_HISTORY_SEAL')
        inspect_chain(run,specs[a])
    return specs

def make_spec(template,arm,base,code,purpose='formal'):
    s=copy.deepcopy(template);s.update(schema='pg3_v21_ablation_executor_v1',arm=arm,method_id=METHOD[arm],weights=weights_for(arm),code=code,sources=snapshot(),output_root=str(base/('runs' if purpose=='formal' else 'short')),run_id='A42_'+arm+'_PG3V21_'+purpose+'_run001')
    s['bindings']=ref(base/'BINDINGS.json')
    s['ablation']=dict(active=list(ACTIVE[arm]),frozen_weights=FROZEN,semantics='sequential_conditional_increment_v1')
    s['execution']['limit']=128 if purpose=='formal' else 4;s['execution']['purpose']='PG3_V21_ABLATION128' if purpose=='formal' else 'PG3_V21_ABLATION_SHORT4'
    s['ablation_qualification']=str(base/'QUALIFICATION'/arm/'QUALIFICATION.json')
    validate_ablation_spec(s);return s

def diagnostic_run(spec,ds,geo,init,cfg,run,pause=None,resume=False):
    if not resume:run.mkdir(parents=True,exist_ok=False);write(run/'RUN_SPEC.json',spec)
    e=AblationEngine(model_from(init),spec,cfg);last=restore_checkpoint(run,e) if resume else publish_checkpoint(run,e,dict(completed=0))
    while e.completed<4:
        ids,_=geo.orders[42].batch(e.completed);rec=e.step(ds.load(ids),geo);last=publish_checkpoint(run,e,rec,sha(last/'COMMIT.json'))
        if pause and e.completed==pause:break
    return e,last

def prepare(base):
    base=Path(base).resolve();base.mkdir(parents=True,exist_ok=False);(base/'RUN_SPECS').mkdir();(base/'QUALIFICATION').mkdir();(base/'short').mkdir()
    code=source_identity();hist=reuse_contract();binding=bind_context(base);prep,(ds,geo,init,qualified,train,dev)=load_context(binding);write(base/'DATA_QUALIFICATION.json',qualified);cfg=read(artifact(prep['configuration']))
    write(base/'REUSE_A_K.json',dict(status='VERIFIED',A=ref(HISTORY/'runs/A42_A_PG3V21_formal_run001/RUN_SEAL.json'),K=ref(HISTORY/'runs/A42_D_PG3V21_formal_run001/RUN_SEAL.json'),existing_source_unchanged=True,new_code=code))
    write(base/'ABLATION_CONFIGS.json',dict(frozen=FROZEN,arms={a:dict(weights=weights_for(a),active=ACTIVE[a],method_id=METHOD[a],formal_status='NOT_RUN') for a in ACTIVE},A='sealed PG2 baseline reused',K='sealed D reused; equivalence tested',recalibrated=False))
    # K aliases sealed D only after explicit scalar and parameter-gradient equivalence on real Batch8.
    model=model_from(init);ids,_=geo.orders[42].batch(0);samples=ds.load(ids)
    d=ObjectiveV21(model,state_sha(model),FROZEN);k=AblationObjective(model,state_sha(model),'K')
    zd=d(model,samples,geo,hist['D']['pg2_weights'],TeacherGuidance(),0);zk=k(model,samples,geo,hist['D']['pg2_weights'],TeacherGuidance(),0)
    from query_minco.training.pg1_v2_r12.diagnostics import flat
    pp=list(model.parameters());gd=flat(torch.autograd.grad(zd['loss'],pp,retain_graph=True,allow_unused=True),pp);gk=flat(torch.autograd.grad(zk['loss'],pp,retain_graph=True,allow_unused=True),pp)
    require(torch.equal(zd['loss'],zk['loss']) and torch.equal(gd,gk),'ABL_K_D_NOT_EXACT')
    d.assignment.rollback(0);k.assignment.rollback(0)
    write(base/'K_D_EQUIVALENCE.json',dict(status='PASS',scalar_equal=True,parameter_gradient_equal=True,batch_ids=ids))
    results={}
    for arm in ('S','M','G'):
        folder=base/'QUALIFICATION'/arm;folder.mkdir();s=make_spec(hist['D'],arm,base,code);write(base/'RUN_SPECS'/(arm+'.json'),s)
        for suffix in ('short_u','short_r'):write(base/'RUN_SPECS'/(arm+'_'+suffix+'.json'),make_spec(hist['D'],arm,base,code,suffix))
        records=[];blockers=[]
        try:
            model=model_from(init);obj=AblationObjective(model,state_sha(model),arm)
            for i in range(8):
                ids,_=geo.orders[42].batch(i);z=obj(model,ds.load(ids),geo,s['pg2_weights'],TeacherGuidance(),0)
                rec,_,_=audit_v21(model,z,False);guard=step_guard_v21(rec,cfg);obj.assignment.rollback(0)
                require(all(rec['components'][k]['parameter_norm']==0 for k in KEYS_V21 if k not in ACTIVE[arm]),'ABL_DISABLED_GRADIENT')
                require(not obj.assignment.history,'ABL_PREFLIGHT_EXPOSURE_MUTATION')
                require(rec['ratios']['R_D']<=10,'ABL_CPU0_MAIN_GT10');records.append(dict(batch=i,ids=ids,audit=rec,guard=guard))
                print('ABL_PREFLIGHT',arm,i,rec['ratios'],flush=True)
            write(folder/'GRADIENT_AUDIT.json',dict(records=records,weights=weights_for(arm),recalibrated=False))
            su=read(base/'RUN_SPECS'/(arm+'_short_u.json'));sr=read(base/'RUN_SPECS'/(arm+'_short_r.json'))
            _,lastu=diagnostic_run(su,ds,geo,init,cfg,base/'short'/su['run_id'])
            diagnostic_run(sr,ds,geo,init,cfg,base/'short'/sr['run_id'],pause=2)
            _,lastr=diagnostic_run(sr,ds,geo,init,cfg,base/'short'/sr['run_id'],resume=True)
            require(state_fingerprint(lastu/'checkpoint.pt')==state_fingerprint(lastr/'checkpoint.pt'),'ABL_SHORT_FINGERPRINT')
            mu=read(lastu/'MONITOR.json');mr=read(lastr/'MONITOR.json');require(mu==mr,'ABL_SHORT_MONITOR')
            for step in range(1,5):
                u=read(lastu.parent/('step%04d'%step)/'UPDATE.json');r=read(lastr.parent/('step%04d'%step)/'UPDATE.json')
                require(all(u[k]==r[k] for k in ('ids','model_sha256','exposure_sha256')),'ABL_SHORT_TRACE')
            write(folder/'SHORT_RESTORE.json',dict(status='PASS',model_optimizer_exposure_rng_fingerprint=state_fingerprint(lastu/'checkpoint.pt'),monitor_equal=True,continuous_updates=4,resumed_updates='2+2',formal_updates=0))
        except Exception: blockers.append(traceback.format_exc())
        result=dict(arm=arm,status='BLOCKED' if blockers else 'QUALIFIED',blockers=blockers,code=code,sources=snapshot(),formal_spec_sha256=digest(s),weights=weights_for(arm),recalibrated=False,gradient_audit=ref(folder/'GRADIENT_AUDIT.json') if (folder/'GRADIENT_AUDIT.json').exists() else None,short=ref(folder/'SHORT_RESTORE.json') if (folder/'SHORT_RESTORE.json').exists() else None,formal_training='NOT_RUN')
        write(folder/'QUALIFICATION.json',result);results[arm]=result['status'];print('ABL_QUALIFICATION',arm,result['status'],flush=True)
        write(base/'RUN_SPECS'/(arm+'_AUTHORIZATION_PENDING.json'),dict(approved=False,spec_sha256=digest(s),scope=s['execution']['purpose'],max_updates=128,authorized_by='',resources=s['execution']))
    write(base/'PREPARATION_RESULT.json',dict(arms=results,code=code,formal_training='NOT_RUN',formal_authorization_required=True))

def run(spec_path,auth_path):
    s=read(spec_path);authorize_ablation(s,read(auth_path));verify_code(s['code']);require(snapshot()==s['sources'],'ABL_SOURCE_DRIFT');reuse_contract()
    q=read(s['ablation_qualification']);require(q['status']=='QUALIFIED' and q['code']==s['code'] and q['sources']==s['sources'] and q['formal_spec_sha256']==digest(s),'ABL_QUALIFICATION_BINDING')
    artifact(q['gradient_audit']);artifact(q['short'])
    prep,(ds,geo,init,qualified,train,dev)=load_context(s['bindings']);hist=read(HISTORY/'runs/A42_D_PG3V21_formal_run001/RUN_SPEC.json');cfg=read(artifact(prep['configuration']))
    for key in ('initialization','order','configuration','pg2_weights','seed','threads','batch_size','updates','optimizer','schedule','precision','device'):require(s[key]==hist[key],'ABL_RUNTIME_CONTRACT:'+key)
    oldbinding=read(artifact(hist['bindings']));newbinding=read(artifact(s['bindings']))
    require(all(newbinding[k]==v for k,v in oldbinding.items() if k not in ('code','historical_reference')) and newbinding['code']==s['code'] and newbinding['historical_reference']['sha256']==oldbinding['historical_reference']['sha256'],'ABL_BINDING_DATA_DRIFT')
    run=Path(s['output_root'])/s['run_id'];run.mkdir(parents=True,exist_ok=False);write(run/'RUN_SPEC.json',s);write(run/'AUTHORIZATION.json',read(auth_path));e=AblationEngine(model_from(init),s,cfg);start=time.monotonic();last=None;error=None;status='FAILED'
    try:
        targets={}
        for name,items in (('training384',train),('development192',dev)):
            _,t=evaluate_set(e.model,items,run/'evaluation',name,0);targets.update(t)
        write(run/'INITIAL_TARGETS.json',targets);write(run/'INITIAL_OPEN_DEV8.json',open8(e.model,read(artifact(prep['bindings']))['development']))
        last=publish_checkpoint(run,e,dict(completed=0))
        order=read(artifact(s['order']))
        while e.completed<128:
            resources(run,start,s);require(snapshot()==s['sources'],'ABL_RUNTIME_SOURCE_DRIFT');ids,roles=geo.orders[42].batch(e.completed);require(ids==order['batches'][e.completed],'ABL_ORDER')
            rec=e.step(ds.load(ids),geo);rec['roles']=roles;alarm=None
            if e.completed in (32,64,96,128):
                cur=open8(e.model,read(artifact(prep['bindings']))['development']);history=e.monitor.setdefault('evaluation_alarm_history',[])
                try:metric_alarm(history,cur,read(run/'INITIAL_OPEN_DEV8.json'),{})
                except ValueError as ex:alarm=str(ex)
                rec.update(open_dev8=cur,checkpoint_alarm=alarm)
            last=publish_checkpoint(run,e,rec,sha(last/'COMMIT.json'));print('ABL_UPDATE',s['arm'],e.completed,rec['audit']['ratios'],flush=True)
            if e.completed in (32,64,96,128):
                for name,items in (('training384',train),('development192',dev)):evaluate_set(e.model,items,run/'evaluation',name,e.completed,targets)
                write(run/('INFERENCE_step%04d.json'%e.completed),inference_measure(e.model,train[0][0]))
            if alarm:raise ValueError(alarm)
        inspect_chain(run,s);status='COMPLETED'
    except Exception:
        error=traceback.format_exc();e.exposure.rollback(e.completed)
        write(run/'REJECTED.json',dict(error=error,phase=e.phase,last_committed=None if last is None else str(last),audit=e.last_audit))
    write(run/'RESULT.json',dict(status=status,accepted_updates=e.completed,model_sha256=state_sha(e.model),wall_s=time.monotonic()-start,error=error))
    require(status=='COMPLETED','ABL_FORMAL_FAILED')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',default='/home/lizheng/query_minco_mainline/research_outputs/pg3_v21_ablation_v1');p.add_argument('--spec');p.add_argument('--authorization');a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    else:run(a.spec,a.authorization)
