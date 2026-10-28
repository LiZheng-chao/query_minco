"""Complete Frozen384 C1 training-mode 0-update audit and planning."""
import json,time,resource,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import read,write,sha,artifact,digest,require
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pg1.objective import forward as old_forward
from query_minco.training.pg1.losses import dynamic_limit_loss
from query_minco.training.pg1_v2_r1.losses import bottleneck_safety_loss
from query_minco.training.initialization import configure
from .spec import R0,RTAIL,OUTPUT,make_spec,authorization,external_sources,cpu0_contract,runtime_identity
from .diagnostics import prepare_gradients,flat,cosine
def run(output,clean=False):
 require(output.is_absolute() and output.parent==OUTPUT,'R12_CPU0_PATH')
 # Parent creates new directory; no reuse of prior complete or failed entry.
 output.mkdir(parents=True,exist_ok=False)
 for n in ['audit','runs','summary','_preparation']:(output/n).mkdir()
 start=time.monotonic();thread=time.thread_time()
 try:
  require(__import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='','R12_CPU_HIDDEN')
  raw=read(ROOT/'research_outputs/pg1_v2_r11/audit003/_preparation/CPU0_ONLY_BINDINGS.json')
  raw['code']=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=source_files(),allow_uncommitted_sources=not clean)
  bp=output/'_preparation/CPU0_ONLY_BINDINGS.json';write(bp,raw);binding=dict(path=str(bp),sha256=sha(bp))
  frozen_external=external_sources()
  ds,geo,init,q=qualify_bindings(binding,42);write(output/'audit/QUALIFICATION.json',q)
  configure(dict(threads=1,seed=42,device='cpu'))
  model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init['checkpoint'])),map_location='cpu',weights_only=True)['state_dict'],strict=True);model.train()
  weights=state_sha(model);require(weights==init['weights_sha256'],'R12_C1_INITIALIZATION')
  seen=set();covered=set();streaks={R0:0,RTAIL:0};rows=[];batches=[];firstreg=None
  with (output/'runs/BATCHES.jsonl').open('x') as ledger:
   for i in range(len(geo.orders[42].ids)):
    verify_code(raw['code'])
    require(external_sources()==frozen_external,'R12_CPU0_EXTERNAL_SOURCE_DRIFT')
    require(time.monotonic()-start<600,'R12_CPU0_WALL')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,'R12_CPU0_RSS')
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
    require(available>=3072,'R12_CPU0_MEM_AVAILABLE')
    ids,_=geo.orders[42].batch(i);samples=ds.load(ids);batches.append(ids)
    pair={}
    for method in (R0,RTAIL):
     rec,a,z,pg=prepare_gradients(model,samples,geo,method,streaks[method],groups=True)
     streaks[method]=rec['ratio_alarm_streak']
     if method==R0:
      orig=old_forward(model,samples,geo,'query_minco_pg1_task_complete_v1')
      require(torch.equal(orig['raw'],z['raw']) and torch.equal(orig['loss'],z['loss']),'R12_R0_BITWISE')
      pp=list(model.parameters())
      gold=flat(torch.autograd.grad(orig['loss'],pp,retain_graph=True,allow_unused=True),pp)
      actual=torch.cat([p.grad.flatten() for p in pp])
      require(torch.equal(gold,actual),'R12_R0_PARAMETER_GRADIENT_BITWISE')
      for k in orig['terms']:require(torch.equal(orig['terms'][k],z['terms'][k]),'R12_OLD_TERM_BITWISE')
      dyn,dt=dynamic_limit_loss(z['analytic']['v'],z['analytic']['a'],z['analytic']['jerk'],z['analytic']['weights'],[3.,8.,100.])
      for k,v in dt.items():require(torch.equal(v,z['dynamic_terms'][k]),'R12_DYNAMIC_COMPONENT_UNCHANGED')
      if firstreg is None:firstreg=dict(raw=a['raw'].copy(),loss=float(z['loss'].detach()),R0_same_old_loss=True,R0_same_old_parameter_gradients=True)
     pair[method]=(rec,a)
     with (output/'runs'/('%s_batch%03d.npz'%('R0' if method==R0 else 'Rtail',i))).open('xb') as array_file:
      np.savez_compressed(array_file,**a)
     del z,pg
    a0=pair[R0][1];at=pair[RTAIL][1]
    for key in ['raw','coeffs','durations','per_query_obstacle','per_query_goal','per_query_dynamic','per_query_domain','grad_imitation','grad_obstacle','grad_goal','grad_dynamic','grad_domain']:
     require(np.array_equal(a0[key],at[key]),'R12_PAIR_OLD_COMPONENT_'+key)
    for method,(rec,aa) in pair.items():
     rec.update(batch=i,weights_before=weights,weights_after=state_sha(model),optimizer_created=False,optimizer_steps=0,
       original_loss_and_dynamic_regression=True,paired_forward_same_raw=True)
     require(rec['weights_after']==weights,'R12_WEIGHTS_MUTATION')
     ledger.write(json.dumps(rec,allow_nan=False)+'\n');ledger.flush();rows.append(rec)
    for sid in ids:covered.add(sid)
    if i%8==0:print('R12_CPU0',i+1,'unique',len(covered),'ratios',[round(pair[m][0]['physical_imitation_gradient_ratio'],4) for m in (R0,RTAIL)],flush=True)
    if len(covered)==384:break
  require(len(covered)==384 and len(batches)==96,'R12_FULL384')
  verify_code(raw['code'])
  require(external_sources()==frozen_external,'R12_CPU0_EXTERNAL_SOURCE_DRIFT')
  require(state_sha(model)==weights,'R12_CPU0_FINAL_WEIGHTS')
  result=dict(passed=True,schema='pg1_v2_r12_cpu0_v2',contract=cpu0_contract(),runtime=runtime_identity(),methods=[R0,RTAIL],lambda_tail=.0025,
    model_weights_before=weights,model_weights_after=state_sha(model),optimizer_created=False,optimizer_steps=0,
    binding=binding,bindings=binding,code=raw['code'],external_sources=external_sources(),
    batches_per_method=len(batches),exposures_per_method=len(batches)*8,unique_tasks_per_method=len(covered),
    source_bundle_clean=clean,CPU0_only=True,training_authorized=False,
    wall_s=time.monotonic()-start,thread_cpu_s=time.thread_time()-thread,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  write(output/'audit/CPU0_RESULT.json',result)
  write(output/'_preparation/COVERAGE.json',dict(batches=batches,covered_ids=sorted(covered),covered_ids_sha256=digest(sorted(covered)),frozen_order=next(x['artifact'] for x in raw['orders'] if x['seed']==42)))
  for method,name in [(R0,'R0'),(RTAIL,'Rtail')]:
   spec=make_spec(binding,method,'A42_'+name+'_R12_CPU_Pilot32_run001')
   spec['cpu0']=None
   # Replace binding to actual seal only in final preparation after CPU0 file closure.
   write(output/'_preparation'/(name+'_SPEC_DRAFT.json'),spec)
  print('R12_CPU0_PASS',len(rows),round(time.monotonic()-start,3),flush=True)
 except BaseException:
  write(output/'audit/FAILURE.json',dict(status='FAIL',error=traceback.format_exc(),optimizer_created=False,optimizer_steps=0))
  raise
