"""One isolated fixture per exclusive run; no training/episode dispatch."""
import argparse,os,json,hashlib,subprocess,sys
from pathlib import Path
N=Path(__file__).resolve().parents[1];R=Path('/home/lizheng/reach_planner')
O=N/'research_outputs/migration_v1';PY='/home/lizheng/anaconda3/envs/yopo/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['import','old','new','training_old','training_new'])
 ap.add_argument('--run',required=True);a=ap.parse_args()
 if not a.run.isidentifier():raise ValueError('explicit ASCII run identifier required')
 run=O/'runs'/a.run;run.mkdir(parents=True,exist_ok=False)
 for n in ['home','tmp','cache']:(run/n).mkdir()
 cfg=json.loads((R/'query_minco/outputs/method_inference_equivalence_v1/_preparation/CONFIG.json').read_text())
 allow=dict(cfg['dependencies']);allow.update(cfg['readonly_metadata'])
 fixture_dir=O/'fixtures'
 if not fixture_dir.exists():
  fixture_dir.mkdir()
  for k,z in cfg['fixtures'].items():
   src=R/z['path'];assert sha(src)==z['sha256']
   with (fixture_dir/(k+'.npz')).open('xb') as f:f.write(src.read_bytes())
 for n,h in allow.items():
  if sha(R/n)!=h:raise ValueError('HISTORICAL_SHA '+n)
 sources={str(p.relative_to(N)):sha(p) for folder in ['query_minco','tools','tests'] for p in (N/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json']}
 inputs={str(p):sha(p) for p in fixture_dir.glob('*.npz')}
 runtime=dict(uname_sha256=sha(Path('/usr/bin/uname')),platform_sha256=sha(Path('/home/lizheng/anaconda3/envs/yopo/lib/python3.8/platform.py')))
 frozen=dict(action=a.action,sources=sources,inputs=inputs,legacy=allow,runtime=runtime)
 write(run/'SOURCE_FREEZE.json',frozen)
 argv=['/usr/bin/bwrap','--unshare-user','--unshare-pid','--unshare-net','--die-with-parent','--new-session',
 '--ro-bind','/','/','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
 '--tmpfs',str(R),'--tmpfs',str(O)]
 fds=[]
 def bind(source,target,writable=False):
  fd=os.open(str(source),os.O_RDONLY|os.O_CLOEXEC);fds.append(fd)
  argv.extend(['--bind-fd' if writable else '--ro-bind-fd',str(fd),str(target)])
 try:
  for n in sorted(allow):bind(R/n,R/n)
  for p in fixture_dir.glob('*.npz'):bind(p,p)
  bind(run,run,True)
  argv+=['--remount-ro',str(R),'--remount-ro',str(O),'--remount-ro','/tmp','--chdir',str(run),'--',PY,'-B','-s',str(N/'tests/numeric_worker.py'),a.action,str(run)]
  env=dict(os.environ);env.pop('PYTHONPATH',None)
  env.update(PATH='/usr/bin:/bin',HOME=str(run/'home'),TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),
   CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPYCACHEPREFIX=str(run/'cache/pyc'),
   OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  write(run/'COMMAND.json',dict(argv=argv,environment={k:env[k] for k in ['PATH','HOME','TMPDIR','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
  with (run/'stdout.log').open('x') as out,(run/'stderr.log').open('x') as err:
   c=subprocess.run(argv,env=env,pass_fds=tuple(fds),stdout=out,stderr=err)
  write(run/'PROCESS_EXIT.json',dict(returncode=c.returncode))
  print(json.dumps(dict(run=str(run),returncode=c.returncode)),flush=True)
  raise SystemExit(c.returncode)
 finally:
  for fd in fds:os.close(fd)
if __name__=='__main__':main()
