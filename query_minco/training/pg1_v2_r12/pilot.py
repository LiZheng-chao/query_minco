"""Future CPU-only dispatch; no action at import, no automatic authorization."""
import os,time,signal,resource,shutil,json
from pathlib import Path
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import read,write,artifact,require,sha,digest
from query_minco.training.run_spec import source_files
from query_minco.training.initialization import configure
from query_minco.training.pg1.spec import qualify_bindings,lr_at
from query_minco.training.pg1.execution import run_engine
from query_minco.training.pg1.state_io import atomic_json
from .diagnostics import prepare_gradients
from .spec import validate,check_authorization,OUTPUT,external_sources,validate_cpu0_seal,runtime_identity
from .evaluation import evaluate
def resources(s,started,run):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','R12_CPU_CUDA_HIDDEN')
 require(time.monotonic()-started<s['resources']['wall_s'],'R12_WALL')
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
 require(mem>=s['resources']['min_available_mib'],'R12_MEM_AVAILABLE')
 require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=s['resources']['rss_cap_mib'],'R12_RSS')
 require(shutil.disk_usage(OUTPUT).free>=s['resources']['disk_cap_gib']*1024**3,'R12_DISK_FREE')
 require(sum(p.stat().st_size for p in Path(run).rglob('*') if p.is_file())<s['resources']['disk_cap_gib']*1024**3,'R12_RUN_DISK_CAP')
 require(source_files()==s['code']['files'],'R12_CODE_DRIFT')
 require(external_sources()==s['recipe']['external_sources'],'R12_EXTERNAL_SOURCE_DRIFT')
def update(model,opt,samples,geo,method,profile,index,streak):
 opt.zero_grad(set_to_none=True);model.train()
 for g in opt.param_groups:g['lr']=lr_at(profile,index)
 rec,a,_,_=prepare_gradients(model,samples,geo,method,streak,groups=True)
 norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),10))
 require(np.isfinite(norm),'R12_CLIP')
 rec.update(schema='pg1_v2_r12_update_v1',completed_optimizer_updates=index+1,order_cursor=index+1,
  lr=lr_at(profile,index),global_grad_norm_before_clip=norm)
 # Only the existing run_engine owns optimizer.step after durable STARTED intent.
 return rec,a
def preflight(s):
 validate(s,training=True)
 require(not (Path(s['output_root'])/s['run_id']).exists(),'R12_RUN_EXISTS')
 require(not (OUTPUT/'_authorization_claims'/(digest(s)+'.json')).exists(),'R12_AUTH_ALREADY_CLAIMED')
 ds,geo,init,_=qualify_bindings(s['bindings'],42)
 pfpath=artifact(s['cpu0']);pf=validate_cpu0_seal(s,pfpath,init['weights_sha256'])
 coverage=read(pfpath.parent/'_preparation/COVERAGE.json')
 require(all(coverage['batches'][i]==geo.orders[42].batch(i)[0] for i in range(96)),'R12_ACTUAL_FROZEN_ORDER')
 resources(s,time.monotonic(),Path(s['output_root'])/s['run_id'])
 return ds,geo,init,pf

def execute(spec_path,auth_path):
 s=read(spec_path);a=read(auth_path);check_authorization(s,a)
 ds,geo,init,pf=preflight(s)
 configure(s)
 run=Path(s['output_root'])/s['run_id'];run.parent.mkdir(parents=True,exist_ok=True);run.mkdir(exist_ok=False)
 claims=OUTPUT/'_authorization_claims';claims.mkdir(exist_ok=True)
 atomic_json(claims/(digest(s)+'.json'),dict(spec_sha256=digest(s),authorization=a,pid=os.getpid()))
 for name,v in [('RUN_SPEC.json',s),('ROOT_SPEC.json',s),('AUTHORIZATION.json',a),('ENVIRONMENT.json',runtime_identity())]:write(run/name,v)
 started=time.monotonic();resources(s,started,run)
 model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init['checkpoint'])),map_location='cpu',weights_only=True)['state_dict'],strict=True)
 require(state_sha(model)==init['weights_sha256'],'R12_INITIAL_WEIGHTS')
 # Forbidden during CPU0: this constructor is reached only after all run authorization gates.
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 stop={'requested':False};previous=signal.signal(signal.SIGTERM,lambda *args:stop.update(requested=True))
 def batch(i):
  resources(s,started,run);ids,_=geo.orders[42].batch(i);return ds.load(ids),geo,s['method']
 dev=read(artifact(s['bindings']))['development']
 def checked_update(*args):
  rec,arrays=update(*args)
  resources(s,started,run)  # after costly math, before returning to durable STARTED/step
  return rec,arrays
 def checked_evaluate(m):
  resources(s,started,run)
  result=evaluate(m,dev)
  resources(s,started,run)
  return result
 try:run_engine(s,run,model,opt,batch,evaluator=checked_evaluate,update_fn=checked_update,stop_requested=lambda:stop['requested'])
 finally:signal.signal(signal.SIGTERM,previous)
 resources(s,started,run)
 write(run/'CPU_FINISH.json',dict(weights_sha256=state_sha(model),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
