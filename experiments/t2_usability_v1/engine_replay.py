"""Real frozen inputs with advancing replay clock and actual Engine commit."""
import sys,json,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
O=Path('research_outputs/t2_usability_20261116_v1')
prior=json.loads(Path('research_outputs/t2_acceptance_20261116_v1/replay/RESULT.json').read_text())
details=json.loads((O/'OBSERVABILITY_RECORDS.json').read_text())
run=Path(prior['rows'][0]['path']).parents[1]
events=[json.loads(x) for x in (run/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
depth={x['stamp']:x for x in events if x.get('kind')=='depth' and x.get('accepted')}
truth=[json.loads(x) for x in (run/'truth_data/EVENTS.jsonl').read_text().splitlines()]
truth=[x for x in truth if x.get('kind')=='truth' and x['status']=='PASS']
birth=(run/'BIRTH_CERTIFICATE.json').read_bytes();args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
selected=json.loads((O/'BIRTH_SELECTION.json').read_text());rows=[]
for i,row in enumerate(prior['rows']):
 state=next(x['camera_pose'] for x in details if x.get('frame')==i and x.get('mode')=='single')
 ts=row['stamp'];event=depth[ts];receipt=event.get('ros_now',ts)
 tr=[x for x in truth if 0<=event['monotonic']-x['monotonic']<=.2][-1]
 with np.load(row['path']) as z:image=z['depth'].copy()
 for label,offset in [('birth_short',.28),('outside_birth',1.)]:
  start=time.perf_counter()
  clock=lambda:receipt+time.perf_counter()-start
  loader=lambda now:BirthGrant.from_records(birth,tr,args,selected,now,event['monotonic']+now-receipt)
  e=Engine(loader,clock)
  e.set_state(state,ts);e.set_goal(np.asarray(state['p'])+[offset,0,0],receipt);e.observe(image,state,ts,ts)
  committed=e.run_once();cmd=e.command(clock())
  rows.append(dict(frame=i,task=label,committed=committed,command_available=cmd is not None,
    elapsed_s=time.perf_counter()-start,events=list(e.events),command=cmd,
    plan=e.committed['plan'].record if e.committed else None))
(O/'ENGINE_REAL_REPLAY.json').write_text(json.dumps(rows,indent=2))
print({label:dict(committed=sum(x['committed'] for x in rows if x['task']==label),available=sum(x['command_available'] for x in rows if x['task']==label)) for label in ['birth_short','outside_birth']})
