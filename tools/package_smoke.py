"""Build/install only into an exclusive temporary fixture under audit outputs."""
import os,sys,subprocess,json,shutil
from pathlib import Path
N=Path(__file__).resolve().parents[1];O=N/'research_outputs/migration_v1'
def main():
 run=O/sys.argv[1];assert sys.argv[1].isidentifier();run.mkdir(exist_ok=False)
 src=run/'buildsrc';src.mkdir()
 shutil.copytree(N/'query_minco',src/'query_minco')
 shutil.copy2(N/'pyproject.toml',src/'pyproject.toml')
 (run/'home').mkdir();(run/'tmp').mkdir();(run/'cache').mkdir()
 py='/home/lizheng/anaconda3/envs/yopo/bin/python'
 base=['/usr/bin/bwrap','--unshare-net','--die-with-parent','--ro-bind','/','/','--proc','/proc','--dev','/dev',
 '--tmpfs','/tmp','--tmpfs','/home/lizheng/reach_planner','--remount-ro','/home/lizheng/reach_planner','--bind',str(run),str(run)]
 env=dict(os.environ,PATH='/usr/bin:/bin',HOME=str(run/'home'),TMPDIR=str(run/'tmp'),
 XDG_CACHE_HOME=str(run/'cache'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 env.pop('PYTHONPATH',None)
 commands=[]
 def execute(tag,args):
  cmd=base+args
  with (run/(tag+'.stdout.log')).open('x') as out,(run/(tag+'.stderr.log')).open('x') as err:
   c=subprocess.run(cmd,env=env,stdout=out,stderr=err)
  commands.append(dict(tag=tag,argv=cmd,returncode=c.returncode))
  with (run/(tag+'.COMMAND.json')).open('x') as f:json.dump(commands[-1],f,indent=2)
  if c.returncode:raise RuntimeError(tag+' failed; preserve logs')
 execute('wheel',['--chdir',str(src),'--',py,'-B','-m','pip','wheel','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',str(run/'wheels'),str(src)])
 wheels=list((run/'wheels').glob('*.whl'));assert len(wheels)==1
 execute('install',['--chdir',str(run),'--',py,'-B','-m','pip','install','--no-deps','--no-index','--no-cache-dir','--target',str(run/'site'),str(wheels[0])])
 # Restore exactly one immutable checkpoint through a real source path.
 cp='/home/lizheng/reach_planner/query_minco/outputs/b2c1/train_v3/A42_C1_orthogonal/checkpoint_step2880.pt'
 base+=['--ro-bind',cp,'/tmp/reference_checkpoint.pt','--remount-ro','/tmp']
 execute('imports_infer',['--chdir',str(run),'--',py,'-B','-s',str(N/'tests/package_fixture.py'),str(run)])
 with (run/'COMMANDS.json').open('x') as f:json.dump(commands,f,indent=2)
 print('PACKAGE_FIXTURE_PASS',run)
if __name__=='__main__':main()
