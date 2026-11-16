from pathlib import Path
import json,subprocess,hashlib,time
R=Path.cwd();O=R/'research_outputs/g4cp1_startup_control_20261111_v1'
O.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=dict(allowed_new=['query_minco/integration/yopo_startup_v3/','ros/g4cp1_startup_v3/','experiments/g4cp1_startup_control_v1/','tests/g4cp1_startup_control_v1/','research_outputs/g4cp1_startup_control_20261111_v1/'],source_before={},history_before={},repositories_before={})
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and 'g4cp1_' not in str(p) and 'yopo_startup_v3' not in str(p):e['source_before'][str(p)]=sha(p)
for repo in [R,R.parent/'YOPO',R.parent/'reach_planner',R.parent/'query_minco_publish']:
 e['repositories_before'][str(repo)]={}
 for name,args in [('head',['rev-parse','HEAD']),('status',['status','--short']),('diff',['diff']),('cached',['diff','--cached'])]:
  argv=['git','-C',str(repo)]+args;p=subprocess.run(argv,capture_output=True,text=True);e['repositories_before'][str(repo)][name]=dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
for p in (R/'research_outputs').rglob('*'):
 if p.is_file() and O not in p.parents:e['history_before'][str(p)]=sha(p)
for p in [R/'.gitignore',R/'AGENTS.md']:
 if p.is_file():e['source_before'][str(p)]=sha(p)
with (O/'ENTRY.json').open('x') as f:json.dump(e,f,indent=2)
print('ENTRY',len(e['source_before']),len(e['history_before']))
