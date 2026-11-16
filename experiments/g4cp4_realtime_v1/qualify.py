from pathlib import Path
import sys,json,time,unittest,importlib.util,numpy as np,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared,sample_batch
from query_minco.integration.observed_space_v1.map import ObservedMap as OldMap
from query_minco.integration.yopo_execution_v4.trajectory import build
O=Path('research_outputs/g4cp4_realtime_20261112_v1')
spec=importlib.util.spec_from_file_location('testp4','tests/g4cp4_realtime_v1/test_core.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
assert result.wasSuccessful()
p=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.4,0,0]),10)
ts=np.linspace(0,2,129)
def stats(v):
 return dict(n=len(v),p50_ms=float(np.percentile(v,50)*1000),p95_ms=float(np.percentile(v,95)*1000),p99_ms=float(np.percentile(v,99)*1000),max_ms=float(max(v)*1000))
timings={}
for name,fn in [('original',lambda:np.stack([p.sample(x) for x in ts])),('batch',lambda:sample_batch(p,ts))]:
 durations=[]
 for i in range(30):
  t=time.perf_counter();out=fn();durations.append(time.perf_counter()-t)
 timings[name]=stats(durations)
root=Path('research_outputs/g4cp3_manual_E0_20261111_192400_13667')
data=json.loads(next(root.rglob('E0_RESULT.json')).read_text())
maps={};states={}
for name,cls in [('original',OldMap),('optimized',ObservedMap)]:
 m=cls(scene_id='e0_frozen_replay');duration=[];failed=[];count=0
 for r in data['records']:
  with np.load(r['raw_file']) as a:d=a['depth'];pose=dict(p=a['pose'].tolist(),q=a['quaternion'].tolist());t=float(a['stamp'])
  start=time.perf_counter()
  try:
   m.add(d,pose,t,t,t);s=m.snapshot(t);count+=1;states[name]=s.states.copy()
  except Exception as e:failed.append(repr(e))
  duration.append(time.perf_counter()-start)
 maps[name]=dict(latency=stats(duration),deadline_violations=sum(v>.15 for v in duration),failures=failed,accepted=count,total=len(duration))
equivalent=bool(np.array_equal(states['original'],states['optimized']))
record=dict(status='STATIC_PASS',tests=result.testsRun,sampling=timings,map_replay=maps,last_states_equal=equivalent,
 bootstrap='NO_GO_NO_REAL_MEASUREMENT',real_100Hz_control='NOT_RUN',live_5Hz='NOT_RUN',live_10Hz='NOT_RUN',new_network_updates=0,new_physics_runs=0)
(O/'QUALIFICATION.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2));assert equivalent
