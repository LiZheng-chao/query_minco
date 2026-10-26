"""Explicit one-stage commands. No command auto-chains Pilot to Formal or approves auth."""
import argparse,json,os,subprocess,sys,time,traceback,shutil
from pathlib import Path
import torch,numpy as np
from query_minco.training.io import read,write,artifact,require,sha
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.initialization import configure
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from .spec import qualify_bindings
from . import execution as ex
from .transport import bind,export_package,import_binding
def supervise(argv,out,timeout_s,env=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();timed_out=False
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
  try:rc=proc.wait(timeout=timeout_s)
  except subprocess.TimeoutExpired:
   timed_out=True;proc.kill();rc=proc.wait()
  stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
 result=dict(returncode=rc,timeout=timed_out,wall_s=time.monotonic()-started)
 write(out/'PROCESS_EXIT.json',result);write(out/'COMMAND.json',dict(argv=argv,cwd=os.getcwd()))
 return result

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 for cmd in ('rebind','qualify','activation','export-assets','import-assets','prepare','preflight'):
  t=sub.add_parser(cmd);t.add_argument('--bindings',required=cmd!='import-assets');t.add_argument('--sha256',required=True);t.add_argument('--output',required=True)
  if cmd=='rebind':t.add_argument('--cpu-development-only',action='store_true')
  if cmd=='qualify':t.add_argument('--seed',type=int,choices=[42,43,44],required=True)
  if cmd=='activation':t.add_argument('--max-batches',type=int,default=48);t.add_argument('--wall-s',type=int,default=3600)
  if cmd=='export-assets':t.add_argument('--copy',action='store_true')
  if cmd=='import-assets':t.add_argument('--package',required=True)
  if cmd in ('prepare','preflight'):
   t.add_argument('--method',choices=ex.SUPPORTED,required=True);t.add_argument('--seed',type=int,choices=[42,43,44],required=True)
  if cmd=='prepare':
   t.add_argument('--profile',choices=['pg1_pilot32_v1','pg1_formal2880_v1'],required=True);t.add_argument('--run-id',required=True)
  if cmd=='prepare':
   for name in ('preflight','activation-review','pilot-pair-audit'):t.add_argument('--'+name);t.add_argument('--'+name+'-sha256')
 e=sub.add_parser('environment');e.add_argument('--output',required=True);e.add_argument('--cuda-probe',action='store_true')
 t=sub.add_parser('resume-plan');t.add_argument('--parent',required=True);t.add_argument('--run-id',required=True);t.add_argument('--output',required=True)
 t=sub.add_parser('pilot-pair-audit');t.add_argument('--run',action='append',required=True);t.add_argument('--output',required=True)
 for cmd in ('run','_worker','auth-template'):
  t=sub.add_parser(cmd);t.add_argument('--spec',required=True)
  if cmd in ('run','_worker'):t.add_argument('--authorization',required=True);t.add_argument('--pause-at',type=int,choices=[8,16])
  else:t.add_argument('--output',required=True)
 for cmd in ('audit','status','export-results'):
  t=sub.add_parser(cmd);t.add_argument('--run',required=True)
  if cmd!='status':t.add_argument('--output',required=True)
 a=p.parse_args();cmd=a.command
 if hasattr(a,'output'):
  out=Path(a.output);require(out.is_absolute() and out.resolve()==out and (ROOT/'research_outputs') in out.parents,'OUTPUT_TREE')
 if cmd=='rebind':
  original=read(artifact(dict(path=a.bindings,sha256=a.sha256)));out.mkdir(parents=True,exist_ok=False)
  original['code']=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),files=source_files(),allow_uncommitted_sources=bool(a.cpu_development_only))
  write(out/'BINDINGS.json',original);print(json.dumps(bind(out/'BINDINGS.json')));return
 if cmd in ('qualify','activation','export-assets','prepare','preflight'):
  binding=dict(path=a.bindings,sha256=a.sha256);artifact(binding)
 if cmd=='qualify':
  _,_,_,q=qualify_bindings(binding,a.seed);out.mkdir(parents=True,exist_ok=False);write(out/'QUALIFICATION.json',q)
 elif cmd=='activation':
  from .activation import run
  run(binding,out,a.max_batches,a.wall_s)
 elif cmd=='export-assets':
  require(read(artifact(binding))['code']['allow_uncommitted_sources'] is False,'EXPORT_REQUIRES_CLEAN_RELEASE')
  qualify_bindings(binding,42)
  print(json.dumps(export_package(binding,out,a.copy)))
 elif cmd=='import-assets':print(json.dumps(import_binding(dict(path=a.package,sha256=a.sha256),out)))
 elif cmd=='prepare':
  s=ex.make_spec(binding,a.method,a.seed,a.profile,a.run_id);out.mkdir(parents=True,exist_ok=False)
  for key in ('preflight','activation_review','pilot_pair_audit'):
   path=getattr(a,key,None)
   if path:
    bb=dict(path=path,sha256=getattr(a,key+'_sha256'));artifact(bb);s[key]=bb
  write(out/'RUN_SPEC.json',s);write(out/'AUTHORIZATION_TEMPLATE.json',ex.authorization(s))
 elif cmd=='auth-template':
  out.parent.mkdir(parents=True,exist_ok=True);write(out,ex.authorization(read(a.spec)))
 elif cmd=='preflight':
  # Zero optimizer objects/steps; uses real CUDA and registered numerical reconstruction.
  require(read(artifact(binding))['code']['allow_uncommitted_sources'] is False,'PREFLIGHT_CLEAN_CODE')
  ds,geo,init,q=qualify_bindings(binding,a.seed)
  out.mkdir(parents=True,exist_ok=False)
  s=ex.make_spec(binding,a.method,a.seed,'pg1_pilot32_v1' if a.seed==42 else 'pg1_formal2880_v1',f'A{a.seed}_preflight')
  Path(s['output_root']).mkdir(parents=True,exist_ok=True);gpu=ex.runtime_gpu(s);configure(s)
  m=QueryMincoPlanner().to('cuda');m.load_state_dict(torch.load(str(artifact(init['checkpoint'])),map_location='cpu',weights_only=True)['state_dict'])
  # update() computes backward/clip, NEVER opt.step. Constructing AdamW is unnecessary.
  class GradOnly:
   param_groups=[dict(lr=0)]
   def zero_grad(self,set_to_none=True):m.zero_grad(set_to_none=set_to_none)
  ids,_=geo.orders[a.seed].batch(0);samples=ds.load(ids);before=state_sha(m)
  r1,x1=ex.update(m,GradOnly(),samples,geo,a.method,'pg1_pilot32_v1',0)
  r2,x2=ex.update(m,GradOnly(),samples,geo,a.method,'pg1_pilot32_v1',0)
  require(all(np.array_equal(x1[k],x2[k]) for k in x1),'PREFLIGHT_DETERMINISTIC')
  require(before==state_sha(m)==init['weights_sha256'],'PREFLIGHT_UPDATE_FORBIDDEN')
  ex.resource_check(s,time.monotonic(),out)
  gpu.update(peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
  for value in (r1,r2):
   value['planned_update_index']=value.pop('completed_optimizer_updates');value['optimizer_updates_executed']=0;value['order_cursor']=0
  for name,value in [('FIRST',r1),('SECOND',r2)]:write(out/(name+'.json'),value)
  with (out/'ARRAYS.npz').open('xb') as f:np.savez(f,**x1)
  with (out/'SECOND_ARRAYS.npz').open('xb') as f:np.savez(f,**x2)
  write(out/'QUALIFICATION.json',q)
  write(out/'PREFLIGHT.json',dict(passed=True,device='cuda',code=s['code'],bindings=binding,method=a.method,seed=a.seed,gpu=gpu,optimizer_created=False,optimizer_steps=0,deterministic_repeat=True))
  write(out/'PREFLIGHT_SEAL.json',dict(schema='pg1_preflight_seal_v1',passed=True,code=s['code'],bindings=binding,method=a.method,seed=a.seed,runtime=dict(torch=torch.__version__,numpy=np.__version__),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
 elif cmd=='environment':
  out.mkdir(parents=True,exist_ok=False)
  r=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_cuda_build=torch.version.cuda,numpy=np.__version__,root=str(ROOT),disk_free_bytes=shutil.disk_usage(ROOT).free,cuda_probe='NOT_RUN')
  if a.cuda_probe:
   torch.backends.cuda.matmul.allow_tf32=False
   require(torch.cuda.is_available() and torch.cuda.device_count()==1,'CUDA_UNAVAILABLE_OR_AMBIGUOUS')
   x=torch.randn(64,64,device='cuda',requires_grad=True);v=(x@x.T).square().mean();v.backward();torch.cuda.synchronize()
   require(torch.isfinite(v) and torch.isfinite(x.grad).all(),'CUDA_REAL_OPERATION_FAILED')
   r.update(cuda_probe='PASS',device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability())
  write(out/'ENVIRONMENT.json',r)
 elif cmd=='resume-plan':
  parent=Path(a.parent);s=read(parent/'RUN_SPEC.json');s.update(run_id=a.run_id,parent=bind(parent/'RUN_SPEC.json'))
  out.mkdir(parents=True,exist_ok=False);write(out/'RUN_SPEC.json',s);write(out/'AUTHORIZATION_TEMPLATE.json',ex.authorization(s))
 elif cmd=='pilot-pair-audit':
  require(len(a.run)==2,'TWO_PILOTS_REQUIRED');audits=[ex.audit_run(r) for r in a.run]
  require({x['method'] for x in audits}==set(ex.SUPPORTED) and all(x['profile']=='pg1_pilot32_v1' and x['seed']==42 and x['recovery_exercised'] for x in audits),'PILOT_PAIR_REAL_RECOVERY')
  require(audits[0]['code']==audits[1]['code'] and audits[0]['bindings']==audits[1]['bindings'],'PILOT_PAIR_IDENTITY')
  for run,audit in zip(a.run,audits):ex.seal_run(run,audit)
  out.mkdir(parents=True,exist_ok=False);write(out/'PILOT_PAIR_AUDIT.json',dict(passed=True,code=audits[0]['code'],bindings=audits[0]['bindings'],methods=list(ex.SUPPORTED),runs=[bind(Path(r)/'RUN_SEAL.json') for r in a.run],recovery_exercised=True))
 elif cmd=='run':
  s=read(a.spec);ex.validate(s)
  require(not (Path(s['output_root'])/s['run_id']).exists(),'RUN_ALREADY_EXISTS')
  launch=ROOT/'research_outputs/pg1_task_complete_v1/formal_runs_v1/_launches'/s['run_id']
  argv=[sys.executable,'-B','-s','-m','query_minco.training.pg1.executor_cli','_worker','--spec',a.spec,'--authorization',a.authorization]
  if a.pause_at:argv+=['--pause-at',str(a.pause_at)]
  env=dict(os.environ,_PG1_PARENT_PID=str(os.getpid()))
  result=supervise(argv,launch,s['resources']['wall_s'],env)
  run=Path(s['output_root'])/s['run_id']
  if run.exists():
   # Child has exited; all captured logs are now immutable.
   for name in ('stdout.log','stderr.log','COMMAND.json','PROCESS_EXIT.json'):
    target=run/name
    require(not target.exists(),'LAUNCH_RECEIPT_CONFLICT')
    with target.open('xb') as dst,(launch/name).open('rb') as src:shutil.copyfileobj(src,dst)
  require(result['returncode']==0 and not result['timeout'],'TRAINING_PROCESS_FAILED')
 elif cmd=='_worker':
  require(os.environ.get('_PG1_PARENT_PID')==str(os.getppid()),'SUPERVISED_WORKER_ONLY')
  s=read(a.spec);run=Path(s['output_root'])/s['run_id'];rc=1
  try:ex.execute(s,read(a.authorization),run,a.pause_at);rc=0
  finally:
   pass # supervisor records actual exit status, including hard timeout
 elif cmd=='audit':
  result=ex.audit_run(a.run);out.mkdir(parents=True,exist_ok=False);write(out/'AUDIT.json',result);ex.seal_run(a.run,result)
 elif cmd=='status':
  run=Path(a.run);files=sorted(run.glob('progress_*.json'))
  print(json.dumps(read(files[-1]) if files else dict(status='no committed update'),indent=2))
 elif cmd=='export-results':
  run=Path(a.run);require((run/'RUN_SEAL.json').exists(),'EXPORT_ONLY_SEALED')
  seal=read(run/'RUN_SEAL.json');require(all(sha(run/n)==h for n,h in seal['files'].items()),'EXPORT_SEAL')
  require(not out.exists() and not Path(str(out)+'.tar.gz').exists(),'EXPORT_EXISTS')
  import tarfile
  with tarfile.open(str(out)+'.tar.gz',mode='x:gz') as tf:tf.add(run,arcname=run.name)
 print(json.dumps(dict(command=cmd,optimizer_steps='only run can update',status='finished')))
if __name__=='__main__':main()
