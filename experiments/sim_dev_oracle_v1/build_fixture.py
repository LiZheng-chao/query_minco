from pathlib import Path
import json,subprocess,os,hashlib
R=Path(__file__).resolve().parents[2];D=Path(__file__).parent;O=R/'research_outputs/sim_dev_oracle_20261118_v1';B=O/'build_v4';OLD=R/'research_outputs/g4cp7b_fix_20261112_v1/build_v2'
def main():
 B.mkdir();rows=[]
 for stage in ['compile','link']:
  prev=json.loads((OLD/stage/'PROCESS_RESULT.json').read_text());argv=prev['argv']
  argv=[x.replace(str(OLD/'renderer.cpp'),str(D/'native_fixture.cpp')).replace(str(OLD/'renderer.o'),str(B/'fixture.o')).replace(str(OLD/'native_renderer'),str(B/'native_fixture')) for x in argv]
  if stage=='link':argv=[x for x in argv if not x.endswith(('source1.o','source2.o'))]+['-ldl']
  with (B/(stage+'.stdout.log')).open('x') as so,(B/(stage+'.stderr.log')).open('x') as se:p=subprocess.run(argv,cwd=prev['cwd'],stdout=so,stderr=se,timeout=180)
  rows.append(dict(argv=argv,cwd=prev['cwd'],exit_code=p.returncode));(B/'COMMANDS.json').write_text(json.dumps(rows,indent=2));print(stage,p.returncode,flush=True)
  if p.returncode:raise RuntimeError(stage)
 env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/home/lizheng/YOPO/Simulator/devel/lib:/opt/ros/noetic/lib')
 with (O/'CUDA_FIXTURE_RESULT_v4.json').open('x') as so,(O/'CUDA_FIXTURE_v4.stderr.log').open('x') as se:p=subprocess.run([str(B/'native_fixture'),str(O)],stdout=so,stderr=se,env=env,timeout=60)
 rows.append(dict(argv=[str(B/'native_fixture'),str(O)],exit_code=p.returncode));(B/'COMMANDS.json').write_text(json.dumps(rows,indent=2));print('NATIVE_FIXTURE',p.returncode)
if __name__=='__main__':main()
