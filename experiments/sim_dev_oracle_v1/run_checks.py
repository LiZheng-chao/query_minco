from pathlib import Path
import sys,json,subprocess,time,os
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/sim_dev_oracle_20261118_v1';D=Path(__file__).parent

def main():
 logs=O/'release_checks';logs.mkdir();rows=[]
 commands=[(name,[sys.executable,'-B',str(D/name)]) for name in ['replay.py','runtime_replay.py','synthetic_follow.py','prefix_study.py']]+[(name,[sys.executable,'-B','-m','unittest','discover','-s',name,'-v']) for name in ['tests/sim_dev_oracle_v1','tests/forest_f1_v1','tests/depth_minco_t2_v1','tests/f0_continuity_v1']]
 for index,(name,argv) in enumerate(commands):
  begin=time.perf_counter()
  with (logs/('%02d.stdout.log'%index)).open('x') as so,(logs/('%02d.stderr.log'%index)).open('x') as se:p=subprocess.run(argv,cwd=R,stdout=so,stderr=se,timeout=180)
  rows.append(dict(name=name,argv=argv,cwd=str(R),exit_code=p.returncode,elapsed_s=time.perf_counter()-begin,stdout=str(logs/('%02d.stdout.log'%index)),stderr=str(logs/('%02d.stderr.log'%index))));(O/'CHECK_COMMANDS.json').write_text(json.dumps(rows,indent=2));print(name,p.returncode,flush=True)
  if p.returncode:print((logs/('%02d.stderr.log'%index)).read_text());sys.exit(p.returncode)
if __name__=='__main__':main()
