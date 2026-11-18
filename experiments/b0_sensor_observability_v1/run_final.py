from pathlib import Path
import sys,subprocess,time,json,os
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/b0_sensor_observability_20261118_v1';D=Path(__file__).parent;logs=O/'final_runs';logs.mkdir();records=[]
for name,argv in [(name,[sys.executable,'-B',str(D/name)]) for name in ['reference.py','multiview.py','test_reference.py']]+[('forest_regression',[sys.executable,'-B','-m','unittest','discover','-s','tests/forest_f1_v1','-v'])]:
 begin=time.time()
 with (logs/(name+'.stdout.log')).open('x') as so,(logs/(name+'.stderr.log')).open('x') as se:p=subprocess.run(argv,cwd=R,stdout=so,stderr=se,timeout=180)
 records.append(dict(argv=argv,cwd=str(R),start_unix=begin,elapsed_s=time.time()-begin,exit_code=p.returncode,stdout=str(logs/(name+'.stdout.log')),stderr=str(logs/(name+'.stderr.log'))));print(name,p.returncode,flush=True)
 (O/'EXECUTION_COMMANDS.json').write_text(json.dumps(records,indent=2))
 if p.returncode:sys.exit(p.returncode)
