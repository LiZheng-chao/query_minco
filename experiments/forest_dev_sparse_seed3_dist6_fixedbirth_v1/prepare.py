from pathlib import Path
import json,subprocess,hashlib,yaml,os
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1';B=O/'build';OLD=R/'research_outputs/g4cp7b_fix_20261112_v1/build_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfgpath=Path('/home/lizheng/YOPO/Simulator/src/config/config.yaml');old=yaml.safe_load(cfgpath.read_text());new=dict(old,tree_dist=6)
 assert new['seed']==3 and new['maze_type']==5
 assert [k for k in new if new[k]!=old[k]]==['tree_dist']
 p=B/'config.yaml';p.write_text(yaml.safe_dump(new));commands=[]
 for stage in ['compile','link']:
  d=json.loads((OLD/stage/'PROCESS_RESULT.json').read_text());argv=d['argv'];argv=[x.replace(str(OLD/'renderer.o'),str(B/'renderer.o')).replace(str(OLD/'native_renderer'),str(B/'native_renderer')).replace('-DCONFIG_FILE_PATH="'+str(cfgpath)+'"','-DCONFIG_FILE_PATH="'+str(p)+'"') for x in argv]
  with (B/(stage+'.stdout.log')).open('x') as so,(B/(stage+'.stderr.log')).open('x') as se:
   proc=subprocess.run(argv,cwd=d['cwd'],stdout=so,stderr=se,timeout=180)
  commands.append(dict(stage=stage,argv=argv,cwd=d['cwd'],exit_code=proc.returncode));(B/'COMMANDS.json').write_text(json.dumps(commands,indent=2))
  if proc.returncode:raise RuntimeError(stage)
 inputs=[cfgpath,OLD/'renderer.cpp',OLD/'sensor_simulator.cuh',Path('/home/lizheng/YOPO/Simulator/devel/lib/libraycast_cuda.so')]+[Path(x) for x in argv if x.endswith('.o')]
 identity=dict(experiment=O.name,config=new,config_changes={'tree_dist':[old['tree_dist'],6]},binary=str(B/'native_renderer'),binary_sha256=sha(B/'native_renderer'),config_sha256=sha(p),inputs={str(x):sha(x) for x in inputs},GT_used_for_planning=False)
 (O/'BUILD_IDENTITY.json').write_text(json.dumps(identity,indent=2));print('BUILD_PASS',identity['binary_sha256'])
if __name__=='__main__':main()
