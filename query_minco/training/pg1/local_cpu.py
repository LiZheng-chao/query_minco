"""Separate WSL CPU-only pilot dispatch; never CUDA/formal qualification.
Numerical update/lifecycle/checkpoint/hard validation remain shared and unchanged.
"""
import argparse,copy,json,os,platform,resource,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import artifact,read,write,require,sha,digest
from query_minco.training.run_spec import ROOT,verify_code,source_files
from query_minco.training.initialization import configure
from query_minco.evaluation.validation import norm_max
from . import execution as ex
from .spec import qualify_bindings,recipe,lr_at
from .state_io import atomic_json,restore,inspect_commit
from .executor_cli import supervise
from .audit import numpy_samples
OUTPUT=ROOT/'research_outputs/pg1_task_complete_v1/local_cpu_pilot32_v1'
METHODS=ex.SUPPORTED
def binding(path):return dict(path=str(path),sha256=sha(path))
def memory():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    return dict(peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        available_mib=values['MemAvailable']/1024**2,total_mib=values['MemTotal']/1024**2)
def environment():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_REQUIRES_CUDA_HIDDEN')
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
       cpu=platform.processor(),logical_cpus=os.cpu_count(),memory=memory(),disk_free_gib=shutil.disk_usage(ROOT).free/1024**3,
       CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],GPU_probe='NOT_RUN',device='cpu')
def make_spec(bb,method,run_id,purpose='pilot32',smoke=None):
    require(method in METHODS and purpose in ('pilot32','smoke2'),'LOCAL_CPU_SCOPE')
    s=ex.make_spec(bb,method,42,'pg1_pilot32_v1',run_id)
    s.update(schema='pg1_local_cpu_spec_v1',component_id='pg1_local_cpu_executor_v1',device='cpu',
       output_root=str(OUTPUT/'runs'),purpose=purpose,cpu_updates=32 if purpose=='pilot32' else 2,
       smoke_receipt=smoke,
       resources=dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,disk_cap_gib=4))
    s['initialization_accounting']['additional_updates']=s['cpu_updates']
    return s
def validate(s):
    expected=make_spec(s['bindings'],s['method'],s['run_id'],s['purpose'],s['smoke_receipt'])
    require(s==expected,'LOCAL_CPU_FIXED_CONTRACT')
    require(s['parent'] is None and s['seed']==42 and s['profile']=='pg1_pilot32_v1','LOCAL_CPU_NO_FORMAL_OR_RESUME')
    require(s['code']['allow_uncommitted_sources'] is False,'LOCAL_CPU_CLEAN_REQUIRED');verify_code(s['code'])
    require(s['run_id'].startswith('A42_') and all(c.isalnum() or c in '_-' for c in s['run_id']),'LOCAL_CPU_RUN_ID')
    require(Path(s['output_root']).resolve()==OUTPUT/'runs','LOCAL_CPU_OUTPUT')
    environment()
    return s
def authorization(s):
    return dict(schema='pg1_local_cpu_authorization_v1',scope='ONE_LOCAL_CPU_RUN_NOT_CUDA',
      run_id=s['run_id'],spec_sha256=digest(s),method=s['method'],purpose=s['purpose'],
      max_optimizer_updates=s['cpu_updates'],approved=False,authorized_by='')
def check_auth(s,a):
    require(a==dict(authorization(s),approved=True,authorized_by=a.get('authorized_by')) and type(a['approved']) is bool and
        isinstance(a['authorized_by'],str) and bool(a['authorized_by'].strip()),'LOCAL_CPU_AUTH')
def resource_check(s):
    m=memory()
    require(m['peak_rss_mib']<=s['resources']['rss_cap_mib'],'LOCAL_CPU_RSS_CAP')
    require(m['available_mib']>=s['resources']['min_available_mib'],'LOCAL_CPU_MEM_AVAILABLE')
    require(shutil.disk_usage(OUTPUT).free>=s['resources']['disk_cap_gib']*1024**3,'LOCAL_CPU_DISK_FREE')
    require(source_files()==s['code']['files'],'LOCAL_CPU_SOURCE_DRIFT')
    return m
def diagnostics(a,matches,limits):
    """Only reporting; never changes loss, gradients, clipping or acceptance thresholds."""
    mask=np.zeros(a['raw'].shape[:2],bool)
    for b,m in enumerate(matches):mask[b,m['queries']]=True
    peaks=np.array([[[max(norm_max(piece,T,order) for piece,T in zip(cf,d)) for order in (1,2,3)]
       for cf,d in zip(c,dt)] for c,dt in zip(a['coeffs'],a['durations'])])
    require(np.isfinite(peaks).all(),'LOCAL_CPU_DYNAMIC_FINITE')
    feasible=(peaks<=np.asarray(limits)).all(-1)
    activation={}
    for k in ('obstacle','goal','dynamic','domain'):
        x=a['per_query_'+k]
        activation[k]={}
        for name,sel in [('all',np.ones_like(mask)),('matched',mask),('unmatched',~mask)]:
            activation[k][name]=dict(count=int(sel.sum()),exact_nonzero=int((x[sel]>0).sum()),above_1e_minus12=int((x[sel]>1e-12).sum()))
    return dict(activation=activation,matched_queries=int(mask.sum()),unmatched_queries=int((~mask).sum()),
       original_minco_dynamic=dict(passed=int(feasible.sum()),count=int(feasible.size),rate=float(feasible.mean()),
          matched_passed=int(feasible[mask].sum()),unmatched_passed=int(feasible[~mask].sum()),
          maxima=peaks.max(axis=(0,1)).tolist(),limits=limits,
          evaluator='unchanged validation.norm_max polynomial roots; dynamic-only, not geometry/execution certification')),peaks
def cpu_update(model,opt,samples,geo,method,profile,index,streak=0):
    rec,a=ex.update(model,opt,samples,geo,method,profile,index,streak)
    diag,peaks=diagnostics(a,rec['matches'],recipe(method)['dynamic_limits'])
    rec['local_cpu_diagnostics']=diag;rec['memory']=memory()
    a['local_cpu_dynamic_maxima']=peaks
    return rec,a
def verify_smoke(s):
    require(s['smoke_receipt'] is not None,'LOCAL_CPU_SMOKE_REQUIRED')
    p=artifact(s['smoke_receipt']);r=read(p)
    require(r['schema']=='pg1_local_cpu_smoke_v1' and r['passed'] is True and r['actual_updates']==2 and
       r['code']==s['code'] and r['bindings']==s['bindings'] and r['method']==s['method'] and r['seed']==42,'LOCAL_CPU_SMOKE_BINDING')
    require(all(sha(p.parent/n)==h for n,h in r['files'].items()),'LOCAL_CPU_SMOKE_CONTENT')
    run=p.parent;ss=read(run/'RUN_SPEC.json');validate(ss)
    actual=smoke_audit(run)
    require(actual['passed'] is True,'LOCAL_CPU_SMOKE_REAUDIT')
def execute(s,auth,run):
    validate(s);check_auth(s,auth)
    if s['purpose']=='pilot32':verify_smoke(s)
    ds,geo,init,q=qualify_bindings(s['bindings'],42)
    root=OUTPUT/'runs';root.mkdir(parents=True,exist_ok=True)
    require(run==root/s['run_id'],'LOCAL_CPU_RUN_PATH');run.mkdir(exist_ok=False)
    claims=OUTPUT/'_authorization_claims';claims.mkdir(exist_ok=True)
    atomic_json(claims/(digest(s)+'.json'),dict(authorization=auth,pid=os.getpid()))
    for name,value in [('RUN_SPEC.json',s),('ROOT_SPEC.json',s),('QUALIFICATION.json',q),('AUTHORIZATION.json',auth),('ENVIRONMENT.json',environment())]:write(run/name,value)
    write(run/'INPUT_HASHES.json',dict(bindings=s['bindings'],initialization=init,source_code=s['code'],spec_sha256=digest(s)))
    resource_check(s);configure(s)
    model=QueryMincoPlanner()
    model.load_state_dict(torch.load(str(artifact(init['checkpoint'])),map_location='cpu',weights_only=True)['state_dict'],strict=True)
    require(state_sha(model)==init['weights_sha256'],'LOCAL_CPU_INITIAL_MODEL')
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    def batch(i):
        resource_check(s);ids,_=geo.orders[42].batch(i)
        return ds.load(ids),geo,s['method']
    dev=read(artifact(s['bindings']))['development'];store=getattr(geo,'store',None)
    def evaluate(m):
        resource_check(s);return ex.evaluate_development(m,dev,store)
    stop={'requested':False}
    previous=signal.signal(signal.SIGTERM,lambda *args:stop.update(requested=True))
    try:ex.run_engine(s,run,model,opt,batch,evaluator=evaluate,
        pause_at=2 if s['purpose']=='smoke2' else None,stop_requested=lambda:stop['requested'],update_fn=cpu_update)
    finally:signal.signal(signal.SIGTERM,previous)
    resource_check(s)
    write(run/'MEMORY_FINAL.json',memory())
def smoke_audit(run):
    s=read(run/'RUN_SPEC.json');validate(s);require(s['purpose']=='smoke2','SMOKE_SCOPE')
    require(read(run/'PROCESS_EXIT.json')['returncode']==0 and read(run/'RESULT.json')['status']=='PAUSED_COMMITTED','SMOKE_PROCESS')
    require(sorted(p.name for p in (run/'commits').glob('step[0-9]*'))==['step0000','step0001','step0002'],'SMOKE_EXACT_UPDATES')
    m=QueryMincoPlanner();o=torch.optim.AdamW(m.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    before=None;memory_final=read(run/'MEMORY_FINAL.json')
    for step in (0,1,2):
        blob,c=restore(run/'commits'/('step%04d'%step),m,o,s,False)
        if step==0:
            before=state_sha(m)
            init=next(v for v in read(artifact(s['bindings']))['initializations'] if v['seed']==42)
            require(before==init['weights_sha256'],'SMOKE_FROZEN_INITIALIZATION')
        else:
            rec=read(run/'commits'/('step%04d'%step)/'UPDATE.json')
            require(rec['completed_optimizer_updates']==step and rec['ratio_alarm_streak']<2 and rec['numpy_reconstruction'],'SMOKE_NUMERIC')
            require(all(torch.isfinite(p).all() for p in m.parameters()) and state_sha(m)!=before,'SMOKE_NOT_UPDATED')
    require(memory_final['peak_rss_mib']<=4096,'SMOKE_MEMORY')
    return dict(passed=True,actual_updates=2,initial_weights_sha256=before,final_weights_sha256=state_sha(m),peak_rss_mib=memory_final['peak_rss_mib'])
def audit(run):
    s=read(run/'RUN_SPEC.json')
    if s['purpose']=='smoke2':
        result=smoke_audit(run)
        path=run/'SMOKE_RESULT.json'
        if not path.exists():write(path,dict(result,schema='pg1_local_cpu_smoke_v1',device='cpu',seed=42,code=s['code'],bindings=s['bindings'],method=s['method'],
          files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()}))
        else:require(all(sha(run/n)==v for n,v in read(path)['files'].items()),'SMOKE_CHANGED')
        return result
    verify_smoke(s)
    result=ex.audit_run(run,cpu_validator=validate);result['device']='cpu';result['CUDA_qualification']=False
    # Recompute reporting metrics instead of trusting extension telemetry.
    for i in range(1,33):
        cp=run/'commits'/('step%04d'%i);rec=read(cp/'UPDATE.json')
        with np.load(run/'intents'/('step%04d_ARRAYS.npz'%i),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        actual,peak=diagnostics(a,rec['matches'],s['recipe']['dynamic_limits'])
        require(actual==rec['local_cpu_diagnostics'] and np.array_equal(peak,a['local_cpu_dynamic_maxima']),'LOCAL_CPU_DIAGNOSTICS_AUDIT')
    ex.seal_run(run,result);return result
def pair(runs,out):
    require(len(runs)==2,'LOCAL_CPU_TWO_RUNS')
    aa=[audit(p) for p in runs];ss=[read(p/'RUN_SPEC.json') for p in runs]
    require({s['method'] for s in ss}==set(METHODS) and all(s['purpose']=='pilot32' for s in ss),'LOCAL_CPU_PAIR_METHODS')
    require(ss[0]['bindings']==ss[1]['bindings'] and ss[0]['code']==ss[1]['code'],'LOCAL_CPU_PAIR_SOURCE')
    by={s['method']:p for s,p in zip(ss,runs)};per=[]
    a,b=[by[m] for m in METHODS]
    for step in range(1,33):
        ra=read(a/'commits'/('step%04d'%step)/'UPDATE.json');rb=read(b/'commits'/('step%04d'%step)/'UPDATE.json')
        require(ra['ids']==rb['ids'] and ra['lr']==rb['lr'],'LOCAL_CPU_PAIR_ORDER')
        per.append(dict(step=step,ids=ra['ids'],E2=ra['losses'],PG1=rb['losses'],
            E2_diagnostics=ra['local_cpu_diagnostics'],PG1_diagnostics=rb['local_cpu_diagnostics'],
            E2_ratio=ra['physical_imitation_gradient_ratio'],PG1_ratio=rb['physical_imitation_gradient_ratio']))
    matched_eval={}
    # Same immutable open-dev8 tasks at the same four checkpoint steps.
    for step in (0,8,16,32):
        ev=[]
        for run in (a,b):
            state=read(run/'commits'/('step%04d'%step)/'UPDATE.json')
            ev.append(state['dev_baseline'] if step==0 else state['development'])
        require([v['id'] for v in ev[0]['rows']]==[v['id'] for v in ev[1]['rows']],'LOCAL_CPU_DEV_PAIR')
        matched_eval[str(step)]=dict(E2=ev[0],PG1=ev[1])
    with np.load(a/'intents/step0032_ARRAYS.npz',allow_pickle=False) as za,np.load(b/'intents/step0032_ARRAYS.npz',allow_pickle=False) as zb:
        raw_diff=float(np.max(abs(za['raw']-zb['raw'])))
    # Training arrays describe pre-step32 outputs; final parameters describe post-step32.
    x=torch.load(str(a/'checkpoints/checkpoint_step0032.pt'),map_location='cpu',weights_only=True)['state_dict']
    y=torch.load(str(b/'checkpoints/checkpoint_step0032.pt'),map_location='cpu',weights_only=True)['state_dict']
    max_weight_delta=max(float((x[k]-y[k]).abs().max()) for k in x)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'PAIRED_RESULTS.json',dict(schema='pg1_local_cpu_pair_v1',device='cpu',CUDA_qualification=False,
      runs=[binding(p/'RUN_SEAL.json') for p in runs],same_initialization=True,paired_steps=per,paired_development=matched_eval,
      pre_step32_raw_max_abs_difference=raw_diff,final_weight_max_abs_difference=max_weight_delta,
      identical_final_weights=max_weight_delta==0,updates_per_run=32,not_a_significance_test=True))
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('environment');t.add_argument('--output',required=True)
    t=sub.add_parser('prepare');t.add_argument('--bindings',required=True);t.add_argument('--sha256',required=True);t.add_argument('--output',required=True)
    t.add_argument('--method',choices=METHODS,required=True);t.add_argument('--purpose',choices=['smoke2','pilot32'],required=True);t.add_argument('--run-id',required=True);t.add_argument('--smoke-receipt');t.add_argument('--smoke-sha256')
    for name in ('authorize','run','_worker'):
        t=sub.add_parser(name);t.add_argument('--spec',required=True)
        if name=='authorize':t.add_argument('--by',required=True);t.add_argument('--output',required=True)
        else:t.add_argument('--authorization',required=True)
    for name in ('audit','status','export'):
        t=sub.add_parser(name);t.add_argument('--run',required=True)
        if name=='export':t.add_argument('--output',required=True)
    t=sub.add_parser('pair');t.add_argument('--run',action='append',required=True);t.add_argument('--output',required=True)
    a=p.parse_args();cmd=a.command
    if hasattr(a,'output'):
        out=Path(a.output);require(out.is_absolute() and out.resolve()==out and (ROOT/'research_outputs') in out.parents,'LOCAL_CPU_OUTPUT_TREE')
    if cmd=='environment':
        out.mkdir(parents=True,exist_ok=False);write(out/'ENVIRONMENT.json',environment())
    elif cmd=='prepare':
        bb=dict(path=a.bindings,sha256=a.sha256);artifact(bb);qualify_bindings(bb,42)
        smoke=dict(path=a.smoke_receipt,sha256=a.smoke_sha256) if a.smoke_receipt else None
        s=make_spec(bb,a.method,a.run_id,a.purpose,smoke);validate(s)
        out.mkdir(parents=True,exist_ok=False);write(out/'RUN_SPEC.json',s);write(out/'AUTHORIZATION_TEMPLATE.json',authorization(s))
    elif cmd=='authorize':
        s=read(a.spec);validate(s)
        write(out,dict(authorization(s),approved=True,authorized_by=a.by))
    elif cmd=='run':
        s=read(a.spec);validate(s);check_auth(s,read(a.authorization))
        require(not (OUTPUT/'runs'/s['run_id']).exists(),'LOCAL_CPU_RUN_EXISTS')
        if s['purpose']=='pilot32':verify_smoke(s)
        argv=[sys.executable,'-B','-s','-m','query_minco.training.pg1.local_cpu','_worker','--spec',a.spec,'--authorization',a.authorization]
        env=dict(os.environ,_PG1_LOCAL_CPU_PARENT=str(os.getpid()))
        launch=OUTPUT/'_launches'/s['run_id'];result=supervise(argv,launch,s['resources']['wall_s'],env)
        run=OUTPUT/'runs'/s['run_id']
        if run.exists():
            for name in ('stdout.log','stderr.log','COMMAND.json','PROCESS_EXIT.json'):
                with (run/name).open('xb') as dst,(launch/name).open('rb') as src:shutil.copyfileobj(src,dst)
        require(result['returncode']==0 and not result['timeout'],'LOCAL_CPU_PROCESS_FAILED')
        print(json.dumps(audit(run),indent=2))
    elif cmd=='_worker':
        require(os.environ.get('_PG1_LOCAL_CPU_PARENT')==str(os.getppid()),'LOCAL_CPU_SUPERVISED_ONLY')
        s=read(a.spec);execute(s,read(a.authorization),OUTPUT/'runs'/s['run_id'])
    elif cmd=='audit':print(json.dumps(audit(Path(a.run)),indent=2))
    elif cmd=='status':
        run=Path(a.run);files=sorted((run/'commits').glob('step[0-9]*/UPDATE.json'))
        if files:
            rec=read(files[-1]);progress=sorted(run.glob('progress_*.json'));timing=read(progress[-1]) if progress else {};print(json.dumps(dict(elapsed_s=timing.get('elapsed_s'),eta_s=timing.get('eta_s'),step=rec.get('completed_optimizer_updates',0),loss=rec.get('loss'),losses=rec.get('losses'),lr=rec.get('lr'),ratio=rec.get('physical_imitation_gradient_ratio'),memory=rec.get('memory'),checkpoints=[p.name for p in (run/'checkpoints').glob('*.pt')],error=read(run/'ERROR.json') if (run/'ERROR.json').exists() else None),indent=2))
        else:print(json.dumps(dict(status='NO_COMMITTED_UPDATE')))
    elif cmd=='pair':pair([Path(v) for v in a.run],out)
    elif cmd=='export':
        run=Path(a.run);audit(run);require((run/'RUN_SEAL.json').exists(),'LOCAL_CPU_EXPORT_PILOT_ONLY')
        import tarfile
        with tarfile.open(str(out),mode='x:gz') as tf:tf.add(run,arcname=run.name)
    print(json.dumps(dict(command=cmd,device='cpu',CUDA_qualification=False)))
if __name__=='__main__':main()
