import sys,json,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.f0_continuity_v1.authority import BirthAuthority
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
from query_minco.integration.yopo_execution_v4.trajectory import build,Admission
O=Path('research_outputs/f0_continuity_20261116_v1')
rows=json.loads(Path('research_outputs/t2_acceptance_20261116_v1/replay/RESULT.json').read_text())['rows']
details=json.loads(Path('research_outputs/t2_usability_20261116_v1/OBSERVABILITY_RECORDS.json').read_text())
run=Path(rows[0]['path']).parents[1]
events=[json.loads(x) for x in (run/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
depth_events={x['stamp']:x for x in events if x.get('kind')=='depth' and x.get('accepted')}
truth=[json.loads(x) for x in (run/'truth_data/EVENTS.jsonl').read_text().splitlines()]
truth=[x for x in truth if x.get('kind')=='truth' and x['status']=='PASS']
birth=(run/'BIRTH_CERTIFICATE.json').read_bytes();args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
selected=json.loads(Path('research_outputs/t2_usability_20261116_v1/BIRTH_SELECTION.json').read_text())
out=[]
for i,row in enumerate(rows):
 state=next(x['camera_pose'] for x in details if x.get('frame')==i and x.get('mode')=='single')
 ts=row['stamp'];event=depth_events[ts];receipt=event['ros_now'];start=time.perf_counter();clock=lambda:receipt+time.perf_counter()-start
 tr=[x for x in truth if 0<=event['monotonic']-x['monotonic']<=.2][-1]
 with np.load(row['path']) as z:depth=z['depth'].copy()
 a=BirthAuthority(lambda t:BirthGrant.from_records(birth,tr,args,selected,t,event['monotonic']+t-receipt),clock)
 a.observe(depth,state,ts);plan=build(state['p'],state['v'],state['a'],state['q'],np.asarray(state['p'])+[.28,0,0],ts)
 try:proof=a.check(plan=plan);result='PASS';reason=None
 except Exception as e:proof=None;result='FAIL';reason=str(e)
 out.append(dict(frame=i,result=result,reason=reason,proof=proof,path=row['path']))
(O/'FROZEN_AUTHORITY_PROBE.json').write_text(json.dumps(out,indent=2))
print([(x['frame'],x['result'],x['reason']) for x in out])
