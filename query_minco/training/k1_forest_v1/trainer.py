"""Thin K1-Forest lifecycle reusing original network,objective,optimizer and checkpoint code."""
from pathlib import Path
import json,random,time,copy
import numpy as np,torch
from query_minco.training.k1_forest_v1.data import load
from query_minco.training.k1_physics_trainer_v1.initialization import build
from query_minco.training.k1_physics_trainer_v1.optimizer import DeadRowGuard
from query_minco.training.k1_physics_trainer_v1.trainer import (sha,canonical,state_sha,gradient_sha,result_numeric,group_stats,metric_terms,optimizer_for,optimizer_report,save_checkpoint,load_checkpoint,shared_parameter_identity,rng_state)
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.audit.run_provenance_v1 import AtomicAttemptLedger
ID='k1_forest_shared_original_L0_p0c_trainer_v1'
R=Path(__file__).resolve().parents[3]
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def verify(path):
 spec=json.loads(Path(path).read_text())
 if sha(spec['data_binding'])!=spec['data_binding_sha256']:raise ValueError('DATA_BINDING_SHA')
 if spec['schema']!='k1_forest_runspec_v1' or spec['trainer_id']!=ID or spec['max_updates']!=128 or spec['evaluation_steps']!=[0,8,32,64,128]:raise ValueError('SPEC_SCHEMA_BUDGET')
 if spec['optimizer']!=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False) or spec['clip_norm']!=10 or spec['engineering_max_steps']!=2:raise ValueError('OPTIMIZER_OR_ENGINEERING_SPEC')
 if spec['device']!='cpu' or spec['dtype']!='float32' or spec['batch_size']!=1:raise ValueError('CPU_FP32_B1')
 for p,h in spec['source_sha256'].items():
  if sha(p)!=h:raise ValueError('SOURCE_DRIFT '+p)
 for row in spec['train']+spec['dev']:
  if sha(row['manifest'])!=row['manifest_sha256']:raise ValueError('DATA_IDENTITY')
 for split in ['train','dev']:
  if not spec[split] or any(x['split']!=split for x in spec[split]):raise ValueError('SPLIT')
 binding=json.loads(Path(spec['data_binding']).read_text())
 if canonical(sorted(binding['rows'],key=lambda x:x['sample_id']))!=canonical(sorted(spec['train']+spec['dev'],key=lambda x:x['sample_id'])):raise ValueError('DATA_BINDING_MEMBERSHIP')
 train_seeds={x['map_seed'] for x in spec['train'] if x['kind']=='forest'};dev_seeds={x['map_seed'] for x in spec['dev'] if x['kind']=='forest'}
 if train_seeds&dev_seeds or (train_seeds|dev_seeds)&{3101,3102,3201,3202,3301,3302}:raise ValueError('MAP_TEST_LEAKAGE')
 tr={x['family_id'] for x in spec['train'] if x['kind']=='mirror'};dv={x['family_id'] for x in spec['dev'] if x['kind']=='mirror'}
 if tr&dv or len(tr)<2 or len(dv)<1:raise ValueError('MIRROR_FAMILY_SPLIT')
 if spec['train_order']!=[r['sample_id'] for r in spec['train']] or canonical(spec['train_order'])!=spec['order_sha256']:raise ValueError('ORDER')
 for split in ['train','dev']:
  if spec['initialization']!='A_reasonable':raise ValueError('INITIALIZATION')
 return spec
def claim(output):
 p=Path(output).resolve()
 if (R/'research_outputs').resolve() not in p.parents:raise ValueError('OUTPUT_ROOT')
 p.parent.mkdir(parents=True,exist_ok=True);p.mkdir();return p
def model_for(spec,seed):
 if seed not in spec['seeds']:raise ValueError('SEED')
 random.seed(seed);np.random.seed(seed);model=build('A_reasonable',seed).train();record=spec['initial_states'][str(seed)]
 if sha(record['path'])!=record['sha256']:raise ValueError('INITIAL_SHA')
 frozen=np.load(record['path']);model.load_state_dict({k:torch.from_numpy(frozen[k].copy()) for k in frozen.files},strict=True)
 if state_sha(model)!=record['model_sha256']:raise ValueError('INITIAL_MODEL_SHA')
 return model
def evaluate(model,objective,rows):
 before=state_sha(model);grad=gradient_sha(model);training=model.training;model.eval();records=[]
 with torch.no_grad():
  for row in rows:
   batch=load(row,row['split']);raw=model(batch.depth,batch.state);out=objective(raw,batch)
   if not torch.isfinite(out.loss):raise ValueError('EVAL_NONFINITE')
   errors,hard=result_numeric(out,batch);crossings=None
   if row['kind']=='mirror':
    m=json.loads(Path(row['manifest']).read_text());q=json.loads((Path(m['folder'])/'SCENE_QUALIFICATION.json').read_text())
    crossings=[crossing_metric(out.trajectory.coeffs[0,0].numpy(),out.trajectory.durations[0,0].numpy(),s['planes_B0'],s['gap_B0'],q['side']) for s in q['sections']]
   crossing_ok=all(c['passed'] for c in crossings) if crossings is not None else None
   records.append(dict(sample_id=row['sample_id'],split=row['split'],kind=row['kind'],loss=float(out.loss),terms=metric_terms(out),hard=hard,
    crossings=crossings,hard_task_crossing=bool(hard['passed'] and crossing_ok) if crossing_ok is not None else None,
    quality_if_hard_passed={k:float(out.metrics[k].mean()) for k in ['S','P','C_shape']} if hard['passed'] else None,independent_minco_errors=errors))
 model.train(training)
 if state_sha(model)!=before or gradient_sha(model)!=grad:raise ValueError('DEV_EVALUATION_MUTATION')
 return dict(records=records,summary={split:dict(n=sum(r['split']==split for r in records),hard_task=sum(r['hard']['passed'] for r in records if r['split']==split),
  mirror_safe_crossing=sum(bool(r['hard_task_crossing']) for r in records if r['split']==split),mean_loss=float(np.mean([r['loss'] for r in records if r['split']==split]))) for split in ['train','dev']},
  dev_backward_calls=0,model_sha_before=before,model_sha_after=state_sha(model))
def update(model,optimizer,guard,objective,row,atomic,run,attempt):
 if row['split']!='train':raise PermissionError('DEV_BACKWARD_FORBIDDEN')
 before_model=state_sha(model);pid=shared_parameter_identity(model);batch=load(row,'train')
 optimizer.zero_grad(set_to_none=True);raw=model(batch.depth,batch.state);raw.retain_grad();out=objective(raw,batch)
 if not torch.isfinite(out.loss):raise ValueError('TRAIN_LOSS_NONFINITE')
 errors,hard=result_numeric(out,batch);out.loss.backward()
 if not torch.isfinite(raw.grad).all() or raw.grad[...,15:].count_nonzero():raise ValueError('RAW_GRADIENT_BOUNDARY')
 grad=group_stats(model);norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True))
 before=[p.detach().clone() for p in model.parameters()];dead=guard.sha();live=model.trajectory_head.mlp[-1].weight[:15].detach().clone()
 # Atomic outer ledger is authoritative; old guard is reused with a unique per-attempt private compatibility record.
 token=atomic.reserve();compat=run/('guard_attempt%04d.json'%attempt);write(compat,dict(attempted=0,completed=0))
 isolation=guard.step(optimizer,compat,1);ledger=atomic.complete(token)
 if shared_parameter_identity(model)!=pid or guard.sha()!=dead or any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('PARAMETER_CONTRACT')
 live_delta=float((model.trajectory_head.mlp[-1].weight[:15]-live).norm())
 if live_delta<=0:raise ValueError('NO_LIVE_UPDATE')
 return dict(sample_id=row['sample_id'],split='train',loss_before=float(out.loss),terms=metric_terms(out),raw0_15_gradient=raw.grad[0,0,:15].tolist(),
  raw15_21_nonzero=0,module_gradients=grad,module_updates=group_stats(model,before),clip_input_norm=norm,live_head_update=live_delta,
  model_sha_before=before_model,model_sha_after=state_sha(model),dead_sha_before=dead,dead_sha_after=guard.sha(),isolation=isolation,
  optimizer_state=optimizer_report(optimizer),atomic_ledger=ledger,hard=hard,independent_minco_errors=errors)
def preflight(spec_path,output):
 spec=verify(spec_path);run=claim(output);model=model_for(spec,spec['seeds'][0]);obj=make_objective();before=state_sha(model);pid=shared_parameter_identity(model);records=[]
 for row in spec['train']:
  model.zero_grad(set_to_none=True);batch=load(row,'train');raw=model(batch.depth,batch.state);raw.retain_grad();out=obj(raw,batch);out.loss.backward()
  if not torch.isfinite(raw.grad).all() or raw.grad[...,15:].count_nonzero():raise ValueError('PREFLIGHT_GRADIENT')
  records.append(dict(sample_id=row['sample_id'],modules=group_stats(model),raw0_15=raw.grad[0,0,:15].tolist(),numeric=result_numeric(out,batch)[0]))
  if state_sha(model)!=before or shared_parameter_identity(model)!=pid:raise ValueError('NOT_SHARED_NETWORK')
 ev=evaluate(model,obj,spec['train']+spec['dev'])
 write(run/'RESULT.json',dict(status='PASS',train_backward=records,evaluation=ev,optimizer_steps=0,one_shared_parameter_group=True,model_sha_before=before,model_sha_after=state_sha(model)))
 return run
def smoke(spec_path,output,resume=None):
 spec=verify(spec_path);owner=Path(spec['owner_root']);run=claim(output)
 if run.parent!=owner:raise ValueError('ENGINEERING_OWNER')
 model=model_for(spec,spec['seeds'][0]);options=dict(spec['optimizer'],weight_decay=.01,amsgrad=True);opt=optimizer_for(model,options);guard=DeadRowGuard(model);obj=make_objective()
 metadata=dict(trainer_id=ID,runspec_sha256=sha(spec_path),engineering_only=True,optimizer=options)
 atomic=AtomicAttemptLedger(owner/'ENGINEERING_ATOMIC_LEDGER.json',2);cursor=0;resume_report=None
 if resume is not None:
  resume=Path(resume);side=json.loads(Path(str(resume)+'.json').read_text())
  blob=torch.load(resume,map_location='cpu',weights_only=True)
  for p in guard.parameters:
   ix=next(i for i,x in enumerate(opt.param_groups[0]['params']) if x is p);key=blob['optimizer']['param_groups'][0]['params'][ix]
   for name,value in [('exp_avg',.4),('exp_avg_sq',.7),('max_exp_avg_sq',.9)]:blob['optimizer']['state'][key][name][15:]=value
  dirty=run/'DIRTY_CHECKPOINT_PROBE.pt'
  with dirty.open('xb') as f:torch.save(blob,f)
  cursor,state=load_checkpoint(dirty,sha(dirty),model,opt,guard,metadata)
  clean=torch.load(resume,map_location='cpu',weights_only=True)
  for key,values in clean['optimizer']['state'].items():
   for k,v in values.items():
    if isinstance(v,torch.Tensor) and not torch.equal(v,opt.state_dict()['state'][key][k]):raise ValueError('RESUME_ACTIVE_STATE')
  if state_sha(model)!=side['model_sha256'] or cursor!=1:raise ValueError('RESUME_MODEL_CURSOR')
  # Restored RNG must reproduce an uninterrupted checkpoint's three RNG streams.
  expected=(torch.rand(3),random.random(),np.random.rand(3))
  load_checkpoint(resume,sha(resume),model,opt,guard,metadata)
  actual=(torch.rand(3),random.random(),np.random.rand(3))
  if not torch.equal(expected[0],actual[0]) or expected[1]!=actual[1] or not np.array_equal(expected[2],actual[2]):raise ValueError('RNG_RESUME')
  load_checkpoint(resume,sha(resume),model,opt,guard,metadata)
  for p in guard.parameters:
   for k,value in [('exp_avg',.2),('exp_avg_sq',.5),('max_exp_avg_sq',.8)]:opt.state[p][k][15:]=value
  resume_report=dict(status='PASS',cross_process_checkpoint=True,cursor=cursor,dirty_state=state,active_optimizer_exact=True,rng_exact=True,source_sha=sha(resume))
 elif (owner/'ENGINEERING_ATOMIC_LEDGER.json').exists():raise PermissionError('ENGINEERING_ALREADY_STARTED')
 row=spec['train'][0] if cursor==0 else next(x for x in spec['train'] if x['kind']=='mirror')
 report=update(model,opt,guard,obj,row,atomic,run,cursor+1);report['classification']='ENGINEERING_OPTIMIZER_STEPS'
 save_checkpoint(run/'checkpoint.pt',model,opt,guard,metadata,cursor+1)
 ev=evaluate(model,obj,spec['train']+spec['dev'])
 write(run/'RESULT.json',dict(status='PASS',engineering_optimizer_steps=1,formal_research_steps=0,report=report,resume=resume_report,evaluation=ev,deployable=False))
 return run
def authorize(spec_path,approval_path,output,seed,resume_sha=None):
 spec=verify(spec_path)
 if approval_path is None:raise PermissionError('RESEARCHER_APPROVAL_REQUIRED')
 a=json.loads(Path(approval_path).read_text());run_id='A_reasonable_seed'+str(seed)+('_resume' if resume_sha else '')
 expected=spec['resume_run_directories' if resume_sha else 'formal_run_directories'].get(run_id)
 if a.get('decision')!='APPROVE_EXACT_RUNSPEC' or a.get('runspec_sha256')!=sha(spec_path) or a.get('run_id')!=run_id or not a.get('researcher_statement') or seed not in spec['seeds'] or Path(output).resolve()!=Path(expected or '/INVALID').resolve() or a.get('resume_sha256')!=resume_sha:raise PermissionError('APPROVAL_SPEC_OUTPUT_RESUME_BINDING')
 return spec
def train(spec_path,approval_path,output,seed,resume=None,resume_sha=None):
 spec=authorize(spec_path,approval_path,output,seed,resume_sha);run=claim(output);model=model_for(spec,seed);opt=optimizer_for(model,spec['optimizer']);guard=DeadRowGuard(model);obj=make_objective();cursor=0
 metadata=dict(trainer_id=ID,runspec_sha256=sha(spec_path),engineering_only=False,optimizer=spec['optimizer'],seed=seed,approval_sha256=sha(approval_path))
 if resume:
  if sha(resume)!=resume_sha:raise ValueError('RESUME_APPROVAL_SHA')
  parent=torch.load(resume,map_location='cpu',weights_only=True);m=parent['metadata']
  for key in ['trainer_id','runspec_sha256','engineering_only','optimizer','seed']:
   if m[key]!=metadata[key]:raise ValueError('RESUME_RUN_IDENTITY')
  cursor,_=load_checkpoint(resume,resume_sha,model,opt,guard,m);metadata['parent_checkpoint']=dict(path=str(resume),sha256=resume_sha,cursor=cursor)
 if cursor>=128:raise ValueError('NO_REMAINING_AUTHORIZED_UPDATES')
 atomic=AtomicAttemptLedger(run/'ATOMIC_LEDGER.json',128-cursor);pid=shared_parameter_identity(model)
 save_checkpoint(run/('step%03d.pt'%cursor),model,opt,guard,metadata,cursor);write(run/('evaluation%03d.json'%cursor),evaluate(model,obj,spec['train']+spec['dev']))
 try:
  with (run/'STEPS.jsonl').open('x') as stream:
   for i in range(cursor,128):
    # Verify frozen identities before each attempted update.
    verify(spec_path);row=spec['train'][i%len(spec['train'])];record=update(model,opt,guard,obj,row,atomic,run,i+1)
    if shared_parameter_identity(model)!=pid:raise ValueError('NETWORK_REINITIALIZED')
    record.update(step=i+1,classification='FORMAL_RESEARCH_TRAINING')
    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
    if i+1 in spec['evaluation_steps']:
     save_checkpoint(run/('step%03d.pt'%(i+1)),model,opt,guard,metadata,i+1);write(run/('evaluation%03d.json'%(i+1)),evaluate(model,obj,spec['train']+spec['dev']))
  write(run/'RESULT.json',dict(status='COMPLETED_EXECUTION_NOT_SCIENTIFIC_SUCCESS',formal_updates_this_process=128-cursor))
 except BaseException as e:
  write(run/'STOP.json',dict(error=repr(e),automatic_resume=False));raise
 return run
