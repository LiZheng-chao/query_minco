"""CPU clock and separate Python GC replay. Not native ROS acceptance."""
import argparse,subprocess,time,gc,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 binary=out/'runtime/native_reference_publisher';prepared=out/'preparation/NATIVE_REFERENCES.bin'
 test=subprocess.run([str(binary),'--cpu-test',str(prepared)],capture_output=True,text=True)
 (out/'logs/CPP_CPU_CHECK.log').write_text(test.stdout+test.stderr);assert test.returncode==0
 proc=subprocess.Popen([str(binary),'--cpu-timer',str(prepared),str(out/'CPU_NATIVE_TIMER.csv')])
 records=[];start=time.monotonic()
 import query_minco.inference.ros_committed_reference
 def cb(phase,info):records.append(dict(phase=phase,generation=info['generation'],mono=time.monotonic(),thread_cpu=time.thread_time()))
 gc.callbacks.append(cb)
 for _ in range(3):
  objects=[]
  for i in range(30000):
   z=[];z.append(z);objects.append(z)
  del objects;gc.collect(2)
 gc.callbacks.remove(cb);proc.wait(timeout=15);assert proc.returncode==0
 x=np.genfromtxt(out/'CPU_NATIVE_TIMER.csv',delimiter=',',names=True);gaps=np.diff(x['mono'])*1000;wake=(x['mono']-x['due'])*1000
 pairs=[];last={}
 for e in records:
  k=e['generation']
  if e['phase']=='start':last[k]=e
  elif k in last:
   b=last.pop(k);pairs.append(dict(generation=k,wall_ms=(e['mono']-b['mono'])*1000,thread_cpu_ms=(e['thread_cpu']-b['thread_cpu'])*1000))
 result=dict(scope='CPU_NATIVE_CLOCK_WITH_SEPARATE_PYTHON_IMPORT_GC_ALLOCATION_NOT_ROS',samples=len(x),gap_p50_p95_max_ms=np.quantile(gaps,[.5,.95,1]).tolist(),wake_p50_p95_max_ms=np.quantile(wake,[.5,.95,1]).tolist(),python_gc=pairs,
  elapsed=time.monotonic()-start,historical_claim='mechanism replay only; old75ms has noGC/CPU trace, unique cause unidentifiable',passed=bool(gaps.max()<=40 and wake.max()<=10))
 with (out/'CPU_NATIVE_STRESS.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='python_gc'},indent=2));assert result['passed']
if __name__=='__main__':main()
