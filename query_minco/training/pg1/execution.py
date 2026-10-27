"""PG1-only finite executor. Formal dispatch requires clean code, GPU receipts and scoped auth."""
import copy,os,random,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.inference.identity import state_sha
from query_minco.training.io import read,write,artifact,require,sha,digest,exact
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.training.pilot_evaluation import metric_alarm,load_rows
from query_minco.evaluation.validation import validate_teacher,evaluate,sample_curve
from query_minco.evaluation.goal import task_metrics
from .spec import METHODS,profile,lr_at,recipe,qualify_bindings
from .objective import forward
from .audit import numpy_samples,numpy_terms
from query_minco.training.audit_rsps import numpy_physical
from .state_io import atomic_json,save_commit,inspect_commit,restore,recovery_point,presets

SUPPORTED=('query_minco_pg1_e2_paired_v1','query_minco_pg1_task_complete_v1')
OUTPUT=ROOT/'research_outputs/pg1_task_complete_v1'
def scientific(s):
 return {k:v for k,v in s.items() if k not in ('run_id','parent','output_root')}
def make_spec(binding,method,seed,profile_id,run_id):
 require(method in SUPPORTED and profile_id in ('pg1_pilot32_v1','pg1_formal2880_v1') and seed in profile(profile_id)['seeds'],'EXEC_METHOD_PROFILE')
 m=read(artifact(binding))
 return dict(schema='pg1_execution_spec_v1',component_id='pg1_executor_v1',run_id=run_id,method=method,seed=seed,
  profile=profile_id,device='cuda',threads=1,batch_size=8,precision='FP32',optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),schedule='cosine_2e-5_to_1e-6_endpoint_inclusive',bindings=binding,code=m['code'],recipe=recipe(method),
  output_root=str(OUTPUT/'formal_runs_v1/runs'),parent=None,
  initialization_accounting=dict(kind='C1_step2880',C1_stage_updates=2880,prior_shared_history_updates=None,
    additional_updates=profile(profile_id)['updates'],full_lifetime_updates=None),
  resources=dict(wall_s=profile(profile_id)['wall_cap_s'],gpu_mib=6144,gpu_fraction=.75,min_free_mib=512,disk_cap_gib=32),
  preflight=None,activation_review=None,pilot_pair_audit=None)
def validate(s):
 require(s['schema']=='pg1_execution_spec_v1' and s['component_id']=='pg1_executor_v1','EXEC_SCHEMA')
 expected=make_spec(s['bindings'],s['method'],s['seed'],s['profile'],s['run_id'])
 for k in expected:
  if k not in ('parent','preflight','activation_review','pilot_pair_audit'):
   require(s.get(k)==expected[k],'EXEC_FIXED_CONTRACT '+k)
 require(set(s)==set(expected),'EXEC_FIELDS')
 require(s['code']['allow_uncommitted_sources'] is False,'EXEC_CLEAN_SOURCE_REQUIRED');verify_code(s['code'])
 require(s['run_id'] and all(c.isalnum() or c in '_-' for c in s['run_id']) and s['run_id'].startswith('A%d_'%s['seed']),'EXEC_RUN_ID')
 root=Path(s['output_root']);require(root==OUTPUT/'formal_runs_v1/runs' and root.resolve()==root,'EXEC_OUTPUT_ROOT')
 require(s['device']=='cuda','EXEC_CUDA_ONLY')
 return s
def authorization(s):
 return dict(schema='pg1_execution_authorization_v1',scope='ONE_PG1_EXECUTION_RUN',spec_sha256=digest(s),
  run_id=s['run_id'],profile=s['profile'],additional_updates=profile(s['profile'])['updates'],
  approved=False,authorized_by='',authorization_id=s['run_id']+'_'+digest(s)[:20])
def claim(s,a):
 validate(s);template=authorization(s)
 require(a==dict(template,approved=True,authorized_by=a.get('authorized_by')) and type(a['approved']) is bool and isinstance(a['authorized_by'],str) and bool(a['authorized_by'].strip()),'EXEC_AUTHORIZATION')
 root=OUTPUT/'formal_runs_v1/_authorization_claims';root.mkdir(parents=True,exist_ok=True)
 atomic_json(root/(a['authorization_id']+'.json'),dict(authorization=a,claimed_pid=os.getpid(),spec_sha256=digest(s)))
def runtime_gpu(s):
 require(str(ROOT.resolve()).startswith('/data/'),'GPU_PROJECT_MUST_RESIDE_UNDER_APPROVED_DATA')
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'EXACTLY_ONE_VISIBLE_CUDA_DEVICE')
 free,total=torch.cuda.mem_get_info();limit=min(s['resources']['gpu_mib']*1024**2,int(s['resources']['gpu_fraction']*total))
 require(free>=limit+s['resources']['min_free_mib']*1024**2,'GPU_FREE_MEMORY')
 for env in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','PIP_CACHE_DIR'):
  value=os.environ.get(env,'');require(value and str(Path(value).resolve()).startswith('/data/'),'DATA_CACHE_ENV '+env)
 torch.cuda.set_per_process_memory_fraction(max(0,limit-512*1024**2)/total);torch.cuda.reset_peak_memory_stats()
 require(os.statvfs(s['output_root']).f_bavail*os.statvfs(s['output_root']).f_frsize>=s['resources']['disk_cap_gib']*1024**3,'DISK_RESERVE')
 return dict(device=torch.cuda.get_device_name(),total_mib=total/1024**2,limit_mib=limit/1024**2)
def resource_check(s,started,run):
 require(time.monotonic()-started<s['resources']['wall_s'],'WALL_LIMIT')
 if s['device']=='cuda':
  cap=s['resources']['gpu_mib']*1024**2
  require(torch.cuda.max_memory_reserved()<=cap,'GPU_MEMORY_LIMIT')
  import subprocess
  text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
  entries=[line.split(',') for line in text.splitlines() if line.strip()]
  mine=[float(row[1].strip()) for row in entries if row[0].strip()==str(os.getpid())]
  require(mine and sum(mine)<=s['resources']['gpu_mib'],'GPU_NATIVE_MEMORY_TELEMETRY')
  require(source_files()==s['code']['files'],'EXEC_SOURCE_DRIFT')
 size=sum(p.stat().st_size for p in Path(run).rglob('*') if p.is_file())
 require(size<s['resources']['disk_cap_gib']*1024**3,'DISK_LIMIT')
def _flatten(grads):return torch.cat([g.detach().reshape(-1) for g in grads])
def update(model,opt,samples,geo,method,profile_id,index,streak=0):
 opt.zero_grad(set_to_none=True);model.train()
 for group in opt.param_groups:group['lr']=lr_at(profile_id,index)
 begin=time.monotonic();z=forward(model,samples,geo,method);raw=z['raw'];out=z['output'];cfg=z['recipe']
 require(raw.dtype==z['loss'].dtype==out.trajectory.coeffs.dtype==torch.float32,'EXEC_FP32')
 out.parameters.durations.retain_grad()
 rg={k:torch.autograd.grad(v,raw,retain_graph=True)[0] for k,v in z['terms'].items()}
 pg={k:_flatten(torch.autograd.grad(v,list(model.parameters()),retain_graph=True)) for k,v in z['terms'].items()}
 raw.grad=None;out.parameters.durations.grad=None;z['loss'].backward()
 require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'EXEC_FINITE_PARAM_GRAD')
 require(raw.grad is not None and torch.isfinite(raw.grad).all() and not torch.count_nonzero(raw.grad[...,19:21]),'EXEC_RAW_GRAD')
 require(out.parameters.durations.grad is not None and torch.isfinite(out.parameters.durations.grad).all(),'EXEC_DURATION_GRAD')
 for n,p in model.named_parameters():
  if n.startswith('trajectory_head.') and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),'EXEC_HEAD_GRAD')
 params=list(model.parameters())
 physical=pg['obstacle']+sum(cfg[k+'_scale']*pg[k] for k in ('goal','dynamic','domain'))
 im=float(pg['imitation'].norm());pn=float(physical.norm());ratio=pn/im if im else (None if pn==0 else 1e30)
 streak=streak+1 if ratio is not None and ratio>10 else 0
 require(streak<2,'TWO_STEP_GRADIENT_RATIO_GT10')
 c=out.trajectory.coeffs.detach().cpu().numpy();dt=out.parameters.durations.detach().cpu().numpy()
 v=numpy_samples(c,dt);fields=geo.load(samples)
 per=numpy_terms(v,np.stack([x.state[6:9].numpy() for x in samples]),z['goal_enabled'].cpu().numpy(),
  [x.row['flight_center_domain_m'] for x in fields],cfg['dynamic_limits'])
 arrays={k:z[k].detach().cpu().numpy() for k in ('positions','velocities','time_weights')}
 per['obstacle']=numpy_physical(arrays['positions'],arrays['velocities'],arrays['time_weights'],[f.boxes.numpy() for f in fields],cfg)
 for k in per:require(np.allclose(per[k],z['per_query'][k].detach().cpu().numpy(),rtol=8e-4,atol=8e-4),'EXEC_NUMPY_'+k)
 numeric={k:float(p.mean()) for k,p in per.items()}
 imloss=np.mean([float(cst[m['queries'],m['teachers']].detach().mean().cpu()) for cst,m in zip(z['costs'],z['matches'])])
 require(np.isclose(imloss,float(z['terms']['imitation'].detach()),rtol=8e-4,atol=8e-4),'EXEC_IM_LOSS')
 total=imloss+numeric['obstacle']+sum(cfg[k+'_scale']*numeric[k] for k in ('goal','dynamic','domain'))
 require(np.isclose(total,float(z['loss'].detach()),rtol=8e-4,atol=8e-4),'EXEC_TOTAL_RECONSTRUCTION')
 # Preserve existing CPU/GPU raw->MINCO numerical tolerance; never loosen hard physics.
 with torch.no_grad():
  rebuilt=TrajectoryPostprocessor()(raw.detach().cpu(),torch.stack([x.state for x in samples]),num_samples=32)
  require(np.allclose(rebuilt.trajectory.coeffs.numpy(),c,rtol=2e-4,atol=2e-4),'EXEC_RAW_MINCO')
 norm=float(torch.nn.utils.clip_grad_norm_(params,10));require(np.isfinite(norm),'EXEC_CLIP')
 record=dict(schema='pg1_update_v1',completed_optimizer_updates=index+1,order_cursor=index+1,lr=lr_at(profile_id,index),
  ids=[x.sample_id for x in samples],M=[len(x.teacher_coeffs) for x in samples],loss=float(z['loss'].detach()),
  losses={k:float(x.detach()) for k,x in z['terms'].items()},matches=z['matches'],
  parameter_component_norms={k:float(x.norm()) for k,x in pg.items()},physical_imitation_gradient_ratio=ratio,
  gradient_cosine=float((physical*pg['imitation']).sum())/(pn*im) if pn and im else None,
  ratio_alarm_streak=streak,global_grad_norm_before_clip=norm,raw19_21_zero=True,score_rows_zero=True,
  duration_gradient_max=float(out.parameters.durations.grad.abs().max()),goal_masked_count=int((~z['goal_enabled']).sum()),
  raw_sha256=digest(raw.detach().cpu().tolist()),numpy_reconstruction=True,pre_step_wall_s=time.monotonic()-begin)
 for k,vv in z['per_query'].items():arrays['per_query_'+k]=vv.detach().cpu().numpy()
 for k,vv in rg.items():arrays['grad_'+k]=vv.detach().cpu().numpy()
 for b,vv in enumerate(z['costs']):arrays['cost_%d'%b]=vv.detach().cpu().numpy()
 arrays.update(raw=raw.detach().cpu().numpy(),raw_grad=raw.grad.detach().cpu().numpy(),coeffs=c,durations=dt)
 require(all(np.isfinite(a).all() for a in arrays.values()),'EXEC_ARRAY_FINITE')
 return record,arrays

def evaluate_development(model,binding,store=None):
 """Original hard verifier/goal/mode definition, all six; evaluation is CPU-only."""
 if store is None:rows=load_rows(binding)
 else:
  rows=[]
  for row in store.read(binding)['rows']:
   with np.load(store.artifact(row['tensors']),allow_pickle=False) as z:
    fields={k:torch.from_numpy(z[k].copy()) for k in ('depth_input','state','teacher_coeffs','teacher_durations')}
   rows.append((type('Dev',(),dict(sample_id=row['id'],**fields))(),store.read(row['metadata'])))
 replica=QueryMincoPlanner();replica.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()});replica.eval()
 result=[]
 with torch.no_grad():
  for sample,meta in rows:
   raw=replica(depth_input=sample.depth_input[None],state=sample.state[None]);out=TrajectoryPostprocessor()(raw,sample.state[None])
   c=out.trajectory.coeffs[0].numpy();dt=out.parameters.durations[0].numpy();par=out.parameters;physics=[];goals=[];modes=set()
   for q in range(6):
    cert=validate_teacher(c[q],dt[q],par.head_pva[0,q].numpy(),par.tail_pva[0,q].numpy(),par.inner_pts[0,q].numpy(),meta['scene'])
    g=task_metrics(evaluate(c[q,-1],dt[q,-1]),evaluate(c[q,-1],dt[q,-1],1),sample.state.numpy()[6:9])
    physics.append(cert['passed']);goals.append(g['passed'])
    if cert['passed'] and g['passed']:
     _,curve=sample_curve(c[q],dt[q],1025);i=int(np.argmin(abs(curve[:,0]-meta['scene']['crossing_x_m'])))
     mode='direct'
     if meta['scene']['kind']!='free':
      y,z=curve[i,1:];mode=('left' if y>=0 else 'right') if abs(y)>=abs(z) else ('up' if z>=0 else 'down')
     if mode in meta['teacher_modes']:modes.add(mode)
   result.append(dict(id=sample.sample_id,physics=physics,goal=goals,complete_oracle=bool(np.any(np.array(physics)&goals)),known_feasible_modes=sorted(modes)))
 return dict(rows=result,goal_rate=float(np.mean([x['goal'] for x in result])),physics_rate=float(np.mean([x['physics'] for x in result])),oracle_rate=float(np.mean([x['complete_oracle'] for x in result])),feasible_mode_count=sum(len(x['known_feasible_modes']) for x in result),selector='N/A',contract='original hard validation + original frozen mode heuristic')

def run_engine(s,run,model,opt,batch_fn,*,root_spec=None,resume=None,fault=None,evaluator=None,pause_at=None,stop_requested=lambda:False,update_fn=None):
 """Internal lifecycle; public CUDA authorization/data gates are execute()."""
 run=Path(run);root_spec=root_spec or s;rows=head_rows(model);started=time.monotonic()
 (run/'intents').mkdir();(run/'commits').mkdir();state=None;committed=None
 baseline=None;history=[];streak=0
 try:
  if resume:
   blob,c=restore(resume,model,opt,root_spec);state=read(Path(resume)/'UPDATE.json')
   require(rows_unchanged(model,rows),'RESUME_HEAD_ROWS')
   start=c['step'];committed=save_commit(run,model,opt,root_spec,start,state,parent_hash=None)
   if start:
    original_arrays=Path(resume).parent.parent/'intents'/('step%04d_ARRAYS.npz'%start)
    require(sha(original_arrays)==state['arrays_sha256'],'RECOVERY_ARRAYS')
    import shutil
    with (run/'intents'/original_arrays.name).open('xb') as dest,original_arrays.open('rb') as src:shutil.copyfileobj(src,dest);dest.flush();os.fsync(dest.fileno())
   atomic_json(run/'RECOVERY.json',dict(source=str(resume),source_sha256=sha(Path(resume)/'COMMIT.json'),credited_updates=start,replayed_updates=0))
   baseline=state.get('dev_baseline');history=state.get('dev_history',[]);streak=state.get('ratio_alarm_streak',0)
  else:
   start=0;baseline=evaluator(model) if evaluator else None
   state=dict(completed_optimizer_updates=0,dev_baseline=baseline,dev_history=[])
   committed=save_commit(run,model,opt,root_spec,0,state)
  if start in profile(s['profile'])['checkpoints']:presets(run,committed,start)
  for index in range(start,profile(s['profile'])['updates']):
   if stop_requested():raise KeyboardInterrupt('graceful stop at committed boundary')
   resource_check(s,started,run)
   samples,geo,method=batch_fn(index)
   rec,arrays=(update_fn or update)(model,opt,samples,geo,method,s['profile'],index,streak)
   rec['task_exposures_completed']=(index+1)*len(samples)
   rec['teacher_exposures_completed']=state.get('teacher_exposures_completed',0)+sum(rec['M'])
   rec.update(root_spec_sha256=digest(root_spec),spec_sha256=digest(root_spec),dev_baseline=baseline,dev_history=history)
   atomic_json(run/'intents'/('step%04d_INTENT.json'%(index+1)),dict(step=index+1,parent_sha256=sha(committed/'COMMIT.json')))
   if fault:fault('before_step',index+1)
   atomic_json(run/'intents'/('step%04d_STARTED.json'%(index+1)),dict(step=index+1))
   opt.step()
   if fault:fault('after_step',index+1)
   require(all(torch.isfinite(p).all() for p in model.parameters()) and rows_unchanged(model,rows),'POST_UPDATE_CONTRACT')
   if evaluator and index+1 in profile(s['profile'])['checkpoints']:
    ev=evaluator(model);metric_alarm(history,ev,baseline,{})
    rec['development']=ev;rec['dev_history']=history
   rec['elapsed_s']=time.monotonic()-started
   # Preserve exact batch evidence, then create/publish the complete durable state.
   with (run/'intents'/('step%04d_ARRAYS.npz'%(index+1))).open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
   rec['arrays_sha256']=sha(run/'intents'/('step%04d_ARRAYS.npz'%(index+1)))
   committed=save_commit(run,model,opt,root_spec,index+1,rec,parent_hash=sha(committed/'COMMIT.json'),fault=fault)
   state=rec;streak=rec['ratio_alarm_streak']
   if index+1 in profile(s['profile'])['checkpoints']:presets(run,committed,index+1)
   atomic_json(run/('progress_%04d.json'%(index+1)),dict(step=index+1,loss=rec['loss'],lr=rec['lr'],
     elapsed_s=rec['elapsed_s'],eta_s=rec['elapsed_s']/(index+1-start)*(profile(s['profile'])['updates']-index-1),
     cuda_reserved_mib=torch.cuda.memory_reserved()/1024**2 if s['device']=='cuda' else 0,
     last_checkpoint=max(x for x in profile(s['profile'])['checkpoints'] if x<=index+1)))
   print('UPDATE',index+1,'loss',rec['loss'],'lr',rec['lr'],flush=True)
   if fault:fault('committed',index+1)
   if pause_at==index+1:
    write(run/'RESULT.json',dict(status='PAUSED_COMMITTED',completed_optimizer_updates=index+1,full_budget_completed=False))
    return
  write(run/'RESULT.json',dict(status='COMPLETED_AWAITING_AUDIT',completed_optimizer_updates=profile(s['profile'])['updates'],
    task_exposures=state.get('task_exposures_completed',0),teacher_exposures=state.get('teacher_exposures_completed',0),additional_updates_in_this_process=profile(s['profile'])['updates']-start,credited_parent_updates=start,optimizer_replays=0,final_weights_sha256=state_sha(model)))
 except BaseException as e:
  write(run/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),last_published_commit=str(committed) if committed else None,retry=False))
  raise

def execute(s,auth,run,pause_at=None):
 validate(s)
 require(pause_at is None or (s['profile']=='pg1_pilot32_v1' and pause_at in (8,16)),'PAUSE_CONTRACT')
 require(s['preflight'] is not None and s['activation_review'] is not None,'MANUAL_PREFLIGHT_AND_RESEARCH_REVIEW_REQUIRED')
 pfpath=artifact(s['preflight']);pf=read(pfpath);rv=read(artifact(s['activation_review']))
 require(pf['schema']=='pg1_preflight_seal_v1' and all('/' not in n and sha(pfpath.parent/n)==h for n,h in pf['files'].items()),'PREFLIGHT_RECEIPT_INTEGRITY')
 require(set(pf['files'])=={'FIRST.json','SECOND.json','ARRAYS.npz','SECOND_ARRAYS.npz','QUALIFICATION.json','PREFLIGHT.json'},'PREFLIGHT_COMPLETE_FILES')
 detail=read(pfpath.parent/'PREFLIGHT.json')
 require(detail['device']=='cuda' and detail['optimizer_steps']==0 and detail['optimizer_created'] is False and detail['deterministic_repeat'] is True,'PREFLIGHT_ZERO_UPDATE')
 with np.load(pfpath.parent/'ARRAYS.npz',allow_pickle=False) as first,np.load(pfpath.parent/'SECOND_ARRAYS.npz',allow_pickle=False) as second:
  require(set(first.files)==set(second.files) and all(np.isfinite(first[k]).all() and np.array_equal(first[k],second[k]) for k in first.files),'PREFLIGHT_REPEAT_ARRAYS')
 runtime=dict(torch=torch.__version__,numpy=np.__version__)
 require(pf['runtime']==runtime,'PREFLIGHT_RUNTIME_DRIFT')
 activationpath=artifact(rv['activation_result']);activation=read(activationpath)
 require(all('/' not in n and sha(activationpath.parent/n)==h for n,h in activation['files'].items()),'ACTIVATION_CONTENT_INTEGRITY')
 require(activation['complete'] is True and activation['bindings']==s['bindings'] and all(x['unique_tasks']==384 for x in activation['seeds'].values()),'ACTIVATION_COMPLETE_ARTIFACT')
 require(pf['passed'] is True and pf['code']==s['code'] and pf['bindings']==s['bindings'] and pf['method']==s['method'] and pf['seed']==s['seed'],'PREFLIGHT_BINDING')
 require(rv['approved'] is True and activation['complete'] is True and activation['bindings']==s['bindings'] and rv['code']==s['code'],'ACTIVATION_REVIEW')
 if s['profile']=='pg1_formal2880_v1':
  require(s['pilot_pair_audit'] is not None,'FORMAL_PILOT_PAIR_REQUIRED');pa=read(artifact(s['pilot_pair_audit']))
  require(pa['passed'] is True and pa['code']==s['code'] and set(pa['methods'])==set(SUPPORTED),'PILOT_PAIR')
  require(len(pa['runs'])==2,'PILOT_PAIR_RUN_COUNT')
  for rb in pa['runs']:
   leaf=artifact(rb).parent;actual=audit_run(leaf)
   require(actual['profile']=='pg1_pilot32_v1' and actual['seed']==42 and actual['recovery_exercised'],'PILOT_REAL_RESUME_REQUIRED')
   sealed=read(artifact(rb));require(all(sha(leaf/n)==h for n,h in sealed['files'].items()),'PILOT_SEAL_DRIFT')
 ds,geo,init,q=qualify_bindings(s['bindings'],s['seed'])
 require(set(activation['seeds'])=={'42','43','44'} and all(len(x['covered_ids'])==384 and set(x['covered_ids'])==set(ds.rows) for x in activation['seeds'].values()),'ACTIVATION_FULL_TASK_COVERAGE')
 root=Path(s['output_root']);root.mkdir(parents=True,exist_ok=True);run=Path(run)
 require(run==root/s['run_id'],'EXEC_RUN_PATH');run.mkdir(exist_ok=False)
 claim(s,auth);write(run/'RUN_SPEC.json',s);write(run/'QUALIFICATION.json',q)
 write(run/'INPUT_HASHES.json',dict(spec_sha256=digest(s),bindings=s['bindings'],initialization=init,recipe_sha256=digest(s['recipe']),source_code=s['code']))
 write(run/'RUNTIME.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
 gpu=runtime_gpu(s);write(run/'GPU.json',gpu);configure(s)
 model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init['checkpoint'])),map_location='cpu',weights_only=True)['state_dict'],strict=True);model.to('cuda')
 require(state_sha(model)==init['weights_sha256'],'EXEC_INITIAL_WEIGHTS')
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 root_spec=s;resume=None
 if s['parent']:
  parent=artifact(s['parent']).parent # binding is parent RUN_SPEC.json
  old=read(parent/'RUN_SPEC.json');root_spec=read(parent/'ROOT_SPEC.json')
  require(parent==Path(old['output_root'])/old['run_id'],'RESUME_PARENT_PATH')
  runtime=read(parent/'RUNTIME.json');pp=Path('/proc')/str(runtime['pid'])/'stat'
  live=pp.exists() and runtime['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and pp.read_text().split()[21]==runtime['start_ticks']
  require(not live,'RESUME_PARENT_STILL_RUNNING')
  require(scientific(old)==scientific(s),'RESUME_SCIENTIFIC_CONTRACT')
  require(not (parent/'RUN_SEAL.json').exists(),'RESUME_SEALED_RUN_FORBIDDEN')
  resume=recovery_point(parent,root_spec)
  require(inspect_commit(resume,root_spec)['step']<profile(s['profile'])['updates'],'RESUME_FULL_BUDGET_ALREADY_USED')
  claims=OUTPUT/'formal_runs_v1/_authorization_claims'
  atomic_json(claims/('resume_'+digest(dict(parent=str(parent),state=sha(resume/'COMMIT.json')))+'.json'),dict(child_spec_sha256=digest(s)))
 write(run/'ROOT_SPEC.json',root_spec)
 def batches(i):
  ids,roles=geo.orders[s['seed']].batch(i)
  return ds.load(ids),geo,s['method']
 store=getattr(geo,'store',None)
 dev=read(artifact(s['bindings']))['development']
 stop={'requested':False}
 previous=signal.signal(signal.SIGTERM,lambda *args:stop.update(requested=True))
 try:
  run_engine(s,run,model,opt,batches,root_spec=root_spec,resume=resume,evaluator=lambda m:evaluate_development(m,dev,store),pause_at=pause_at,stop_requested=lambda:stop['requested'])
 finally:signal.signal(signal.SIGTERM,previous)

def _tree_equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(_tree_equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(_tree_equal(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b

def _same_inherited_state(first,second):
 a=inspect_commit(first,read(Path(first).parent.parent/'ROOT_SPEC.json'))
 root=read(Path(second).parent.parent/'ROOT_SPEC.json');b=inspect_commit(second,root)
 x=torch.load(str(Path(first)/('checkpoint_step%04d.pt'%a['step'])),map_location='cpu',weights_only=True)
 y=torch.load(str(Path(second)/('checkpoint_step%04d.pt'%b['step'])),map_location='cpu',weights_only=True)
 for key in ('state_dict','optimizer_state_dict','rng','completed_optimizer_updates','order_cursor','next_lr_index','scheduler','spec_sha256'):
  require(_tree_equal(x[key],y[key]),'AUDIT_INHERITED_STATE '+key)

def audit_run(run,ds=None,geo=None,*,cpu_validator=None):
 """Read-only content audit of a completed scientific family, including credited parent."""
 run=Path(run);s=read(run/'RUN_SPEC.json');root=read(run/'ROOT_SPEC.json')
 require(read(run/'RESULT.json')['status']=='COMPLETED_AWAITING_AUDIT','AUDIT_NOT_COMPLETE')
 require(read(run/'PROCESS_EXIT.json')['returncode']==0,'AUDIT_PROCESS_EXIT')
 expected_init=None
 if ds is None:
  if cpu_validator is not None:
   require(s['schema']=='pg1_local_cpu_spec_v1' and s['device']=='cpu','AUDIT_LOCAL_CPU_ONLY');cpu_validator(s)
  else:validate(s)
  ds,geo,init,_=qualify_bindings(s['bindings'],s['seed']);expected_init=init['weights_sha256']
 else:require(s['schema']=='SYNTHETIC_CPU_ONLY' and s['device']=='cpu','AUDIT_TEST_SCOPE')
 families=[];cursor=run;seen=set()
 while True:
  require(str(cursor) not in seen,'RESUME_CYCLE');seen.add(str(cursor));families.append(cursor)
  ss=read(cursor/'RUN_SPEC.json')
  require(scientific(ss)==scientific(s),'AUDIT_FAMILY_CONTRACT')
  if not ss['parent']:break
  cursor=artifact(ss['parent']).parent
 families.reverse();records={};research={}
 for directory in families:
  local=sorted((directory/'commits').glob('step[0-9]*'))
  if read(directory/'RUN_SPEC.json')['parent']:
   rr=read(directory/'RECOVERY.json');source=Path(rr['source'])
   parent=artifact(read(directory/'RUN_SPEC.json')['parent']).parent
   require(source==recovery_point(parent,root) and sha(source/'COMMIT.json')==rr['source_sha256'],'AUDIT_RECOVERY_SOURCE')
   _same_inherited_state(source,local[0])
  for p in local:
   c=inspect_commit(p,root);step=c['step']
   # A child starting at a credited step must contain exactly the inherited state.
   if step in records:
    old=records[step]
    require(c['weights_sha256']==old[2]['weights_sha256'],'AUDIT_RESUME_MODEL')
    continue
   records[step]=(directory,p,c)
  for p in (directory/'checkpoints').glob('*.pt'):
   research[int(p.stem.split('step')[1])]=(directory,p)
 n=profile(s['profile'])['updates']
 require(set(records)==set(range(n+1)) and sorted(research)==profile(s['profile'])['checkpoints'],'AUDIT_EXACT_STEPS_CHECKPOINTS')
 model=QueryMincoPlanner();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 for step in range(n+1):
  directory,p,c=records[step];blob,_=restore(p,model,opt,root,False)
  require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'AUDIT_FINITE_STATE')
  for state in opt.state.values():
   require(all(torch.isfinite(x).all() for x in state.values() if isinstance(x,torch.Tensor)),'AUDIT_OPTIMIZER_FINITE')
  if step==0:
   if expected_init:require(state_sha(model)==expected_init,'AUDIT_INITIAL_C1')
   initial=head_rows(model);continue
  require(rows_unchanged(model,initial),'AUDIT_FROZEN_ROWS')
  rec=read(p/'UPDATE.json');ids,roles=geo.orders[s['seed']].batch(step-1)
  require(rec['ids']==ids and rec['lr']==lr_at(s['profile'],step-1) and rec['completed_optimizer_updates']==rec['order_cursor']==step and rec['spec_sha256']==digest(root),'AUDIT_STEP_ORDER_LR')
  arrays_path=directory/'intents'/('step%04d_ARRAYS.npz'%step)
  require(sha(arrays_path)==rec['arrays_sha256'],'AUDIT_ARRAYS')
  with np.load(arrays_path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  require(all(np.isfinite(v).all() for v in a.values()) and not np.count_nonzero(a['raw_grad'][...,19:21]),'AUDIT_RAW_FINITE_SCORE')
  samples=ds.load(ids);fields=geo.load(samples);v=numpy_samples(a['coeffs'],a['durations'])
  from .geometry import authenticated_task_gates
  _,gates=authenticated_task_gates(samples,geo)
  per=numpy_terms(v,np.stack([x.state[6:9].numpy() for x in samples]),np.array([g['goal_enabled'] for g in gates]),[f.row['flight_center_domain_m'] for f in fields],s['recipe']['dynamic_limits'])
  per['obstacle']=numpy_physical(a['positions'],a['velocities'],a['time_weights'],[f.boxes.numpy() for f in fields],s['recipe'])
  for k in per:require(np.allclose(per[k],a['per_query_'+k],rtol=8e-4,atol=8e-4) and np.isclose(per[k].mean(),rec['losses'][k],rtol=8e-4,atol=8e-4),'AUDIT_INDEPENDENT_'+k)
  from scipy.optimize import linear_sum_assignment
  imitation=[]
  for b,sample in enumerate(samples):
   matrix=a['cost_%d'%b];require(matrix.shape==(6,len(sample.teacher_coeffs)),'AUDIT_COST_SHAPE')
   qq,tt=linear_sum_assignment(matrix);match=rec['matches'][b]
   require(match['sample_id']==sample.sample_id and match['queries']==qq.tolist() and match['teachers']==tt.tolist(),'AUDIT_HUNGARIAN')
   imitation.append(float(matrix[qq,tt].mean()))
  iv=float(np.mean(imitation))
  require(np.isclose(iv,rec['losses']['imitation'],rtol=8e-4,atol=8e-4),'AUDIT_IMITATION')
  total=iv+float(per['obstacle'].mean())+sum(s['recipe'][k+'_scale']*float(per[k].mean()) for k in ('goal','dynamic','domain'))
  require(np.isclose(total,rec['loss'],rtol=8e-4,atol=8e-4),'AUDIT_TOTAL')
  for g in opt.param_groups:
   require(g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 and g['lr']==lr_at(s['profile'],step-1),'AUDIT_OPT_RECIPE')
 for step,(directory,p) in research.items():
  own=directory/'commits'/('step%04d'%step)
  require(sha(p)==sha(own/p.name),'AUDIT_RESEARCH_CHECKPOINT')
 require(read(run/'RESULT.json')['final_weights_sha256']==state_sha(model),'AUDIT_FINAL_WEIGHTS')
 return dict(passed=True,spec_sha256=digest(s),code=s['code'],bindings=s['bindings'],method=s['method'],seed=s['seed'],
  profile=s['profile'],updates=n,actual_optimizer_replays=0,task_exposures=read(run/'RESULT.json')['task_exposures'],teacher_exposures=read(run/'RESULT.json')['teacher_exposures'],
  recovery_exercised=len(families)>1,source_runs=[str(p) for p in families],
  note='Independent array/loss/state audit; old complete hard verifier is executed at development checkpoints.')

def seal_run(run,audit):
 run=Path(run);require(audit['passed'] is True,'NO_SEAL_WITHOUT_AUDIT')
 if (run/'RUN_SEAL.json').exists():
  expected=read(run/'RUN_SEAL.json')['files']
  require({str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='RUN_SEAL.json'}==expected,'EXISTING_SEAL_CHANGED')
  return
 write(run/'CONTENT_AUDIT.json',audit)
 write(run/'RUN_SEAL.json',dict(schema='pg1_run_seal_v1',files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()}))
