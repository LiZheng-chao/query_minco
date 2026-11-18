from pathlib import Path
import subprocess,json,os,sys,time
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1';rows=[]
commands=[[sys.executable,'-B',str(Path(__file__).with_name('test_sparse_contract.py'))]]+[[sys.executable,'-B','-m','unittest','discover','-s',folder,'-v'] for folder in ['tests/forest_f1_v1','tests/depth_minco_t2_v1','tests/f0_continuity_v1']]
for index,cmd in enumerate(commands):
 begin=time.perf_counter()
 with (O/('test_%d.stdout.log'%index)).open('x') as so,(O/('test_%d.stderr.log'%index)).open('x') as se:p=subprocess.run(cmd,cwd=R,stdout=so,stderr=se,timeout=180)
 rows.append(dict(argv=cmd,cwd=str(R),exit_code=p.returncode,elapsed_s=time.perf_counter()-begin,stdout='test_%d.stdout.log'%index,stderr='test_%d.stderr.log'%index));print(index,p.returncode,flush=True)
(O/'TEST_COMMANDS.json').write_text(json.dumps(rows,indent=2));sys.exit(any(x['exit_code'] for x in rows))
