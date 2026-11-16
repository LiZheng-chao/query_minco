from pathlib import Path
import sys,json,time,unittest,importlib.util,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.observed_scheduler_v1.pipeline import Pipeline
from query_minco.integration.yopo_execution_v4.trajectory import build
O=Path('research_outputs/g4cp4_ros_scheduler_20261112_v1')
spec=importlib.util.spec_from_file_location('t','tests/g4cp4_scheduler_v1/test_pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));assert r.wasSuccessful()
root=Path('research_outputs/g4cp3_manual_E0_20261111_192400_13667')
e=json.loads(next(root.rglob('E0_RESULT.json')).read_text());now=[e['records'][0]['stamp']];p=Pipeline(clock=lambda:now[0]);rows=[];poll=[]
try:
 first=e['records'][0]['pose']['p'];plan=build(np.array(first),np.zeros(3),np.zeros(3),[0,0,0,1],np.array(first)+[.4,0,0],now[0]);p.request(plan.record)
 for rec in e['records']:
  with np.load(rec['raw_file']) as a:d=a['depth'];pose=dict(p=a['pose'].tolist(),q=a['quaternion'].tolist());stamp=float(a['stamp'])
  now[0]=stamp;n=p.status()['processed'];start=time.perf_counter();accepted=p.ingest(d,pose,stamp,stamp);ingress=time.perf_counter()-start
  end=time.monotonic()+1.
  while accepted and p.status()['processed']==n and time.monotonic()<end:
   t=time.perf_counter();permission=p.consume(plan.id);poll.append(time.perf_counter()-t);time.sleep(.001)
  rows.append(dict(stamp=stamp,accepted=accepted,ingress_s=ingress,status=p.status(),permission=p.consume(plan.id) is not None))
finally:p.close()
def stats(v):return dict(p50_ms=float(np.percentile(v,50)*1000),p95_ms=float(np.percentile(v,95)*1000),p99_ms=float(np.percentile(v,99)*1000),max_ms=float(max(v)*1000))
record=dict(status='STATIC_REPLAY_ONLY',frames=len(rows),tests=r.testsRun,rows=rows,ingress=stats([x['ingress_s'] for x in rows]),consumer=stats(poll),authorized=sum(x['permission'] for x in rows),
 physical_100Hz='NOT_RUN',initial_measurement='UNAVAILABLE_RESEARCHER_CONFIRMED',new_network_updates=0,new_physics_runs=0)
(O/'REPLAY.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='rows'},indent=2))
assert record['authorized']==0
