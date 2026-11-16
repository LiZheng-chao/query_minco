from pathlib import Path
import sys,subprocess,time,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.owned_stop_v2.stop import OwnedProcessStop
O=Path('research_outputs/stop_latency_fix_20261116_v1')
loads=[];rows=[]
try:
 for i in range(2):loads.append(subprocess.Popen([sys.executable,'-B','-c','import time;end=time.monotonic()+60\nwhile time.monotonic()<end: sum(range(30000))'],start_new_session=True))
 for i in range(12):
  code="import time\nx=bytearray(64*1024*1024)\nprint('READY',flush=True)\ntime.sleep(60)"
  p=subprocess.Popen([sys.executable,'-B','-c',code],stdout=subprocess.PIPE,text=True,start_new_session=True)
  assert p.stdout.readline().strip()=='READY'
  r=OwnedProcessStop(p).stop('NONMOTION_STRESS');rows.append(r);p.stdout.close()
finally:
 for p in loads:
  if p.poll() is None:p.kill()
  p.wait()
(O/'FIXED_NONMOTION.json').write_text(json.dumps(dict(cases=rows,load='2 CPU workers,target64MiB',scope='ordinary processes,no ROS/no motion,no assertion historical174ms reproduced'),indent=2))
print('FIXED',len(rows),'MAX_MS',max(r['elapsed_s'] for r in rows)*1000)
