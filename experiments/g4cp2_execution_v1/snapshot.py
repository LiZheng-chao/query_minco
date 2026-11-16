from pathlib import Path
import json,hashlib,subprocess
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1';O.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=dict(allowed_new=['query_minco/integration/yopo_execution_v4/','ros/g4cp2_execution_v4/','experiments/g4cp2_execution_v1/','tests/g4cp2_execution_v1/','research_outputs/g4cp2_execution_20261111_v1/'],source_before={},history_before={},repositories_before={})
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and not any(str(p.relative_to(R)).startswith(a) for a in e['allowed_new']):e['source_before'][str(p)]=sha(p)
e['source_before'][str(R/'.gitignore')]=sha(R/'.gitignore')
for repo in [R,R.parent/'YOPO',R.parent/'reach_planner',R.parent/'query_minco_publish']:
 e['repositories_before'][str(repo)]={}
 for name,args in [('head',['rev-parse','HEAD']),('status',['status','--short']),('diff',['diff']),('cached',['diff','--cached'])]:
  argv=['git','-C',str(repo)]+args;p=subprocess.run(argv,capture_output=True,text=True);e['repositories_before'][str(repo)][name]=dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
for p in (R/'research_outputs').rglob('*'):
 if p.is_file() and O not in p.parents:e['history_before'][str(p)]=sha(p)
with (O/'ENTRY.json').open('x') as f:json.dump(e,f,indent=2)
print('ENTRY_COUNTS',len(e['source_before']),len(e['history_before']))
