"""Frozen real Depth/Odom replay; no ROS, optimizer or Plant."""
import argparse,json,time,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--output',required=True);p.add_argument('--max-frames',type=int,default=12);a=p.parse_args()
 run=Path(a.source_run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 selected=json.loads(Path('research_outputs/t2_depth_minco_20261116_v1/BIRTH_SELECTION.json').read_text())
 birth=(run/'BIRTH_CERTIFICATE.json').read_bytes();bc=json.loads(birth)
 args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
 truth=[json.loads(x) for x in (run/'truth_data/EVENTS.jsonl').read_text().splitlines() if x]
 truth=[x for x in truth if x.get('kind')=='truth' and x['status']=='PASS']
 events=[json.loads(x) for x in (run/'supervisor_data/EVENTS.jsonl').read_text().splitlines() if x]
 odom={x['stamp']:x for x in events if x.get('kind')=='odom' and x.get('accepted')}
 imu={x['stamp']:x for x in events if x.get('kind')=='imu' and x.get('accepted')}
 depth={x['stamp']:x for x in events if x.get('kind')=='depth' and x.get('accepted')}
 cutoff=json.loads((run/'FAULT.json').read_text())['monotonic']
 samples=[];excluded=Counter()
 for f in sorted((run/'reference_data').glob('*.npz')):
  data=np.load(f);stamp=float(data['stamp'])
  if stamp not in odom or stamp not in imu or stamp not in depth:excluded['NO_EXACT_PVA_DEPTH_MATCH']+=1;continue
  if not bc['issued']<=stamp<=bc['expires'] or depth[stamp]['monotonic']>=cutoff:excluded['OUTSIDE_HEALTHY_CERT_WINDOW']+=1;continue
  prior=[x for x in truth if 0<=depth[stamp]['monotonic']-x['monotonic']<=.2]
  if not prior:excluded['NO_FRESH_HISTORICAL_TRUTH_AUTHORITY']+=1;continue
  state=dict(odom[stamp]['payload'],a=imu[stamp]['payload']['a_world'])
  samples.append(dict(path=f,stamp=stamp,depth=data['depth'].copy(),state=state,truth=prior[-1],mono=depth[stamp]['monotonic']))
 if not samples:raise ValueError('NO_QUALIFIED_REPLAY_SAMPLES')
 indices=np.linspace(0,len(samples)-1,min(a.max_frames,len(samples)),dtype=int)
 current=[samples[0]]
 def loader(now):
  s=current[0]
  return BirthGrant.from_records(birth,s['truth'],args,selected,now,s['mono']+(now-s['stamp']))
 clock=[samples[0]['stamp']]
 engine=Engine(loader,clock=lambda:clock[0]);rows=[]
 # Frozen task: +1m world X; same goal for every frame. No GT route or witnesses.
 goal=np.asarray(samples[0]['state']['p'])+np.array([1.,0,0])
 for index in indices:
  s=samples[int(index)];current[0]=s;clock[0]=s['stamp']
  engine=Engine(loader,clock=lambda:clock[0]) # explicitly independent-frame diagnostic
  engine.set_state(s['state'],s['stamp']);engine.set_goal(goal,s['stamp'])
  engine.observe(s['depth'],s['state'],s['stamp'],s['stamp'])
  t=time.perf_counter();ok=engine.run_once();elapsed=time.perf_counter()-t
  command=engine.command(clock[0]);event=list(engine.events)[-1]
  rows.append(dict(path=str(s['path']),sha256=sha(s['path']),stamp=s['stamp'],committed=ok,command_available=command is not None,elapsed_s=elapsed,event=event))
 (out/'RESULT.json').write_text(json.dumps(dict(scope='FROZEN_NONMOTION_REPLAY',eligible=len(samples),selected=len(rows),selection='uniform frame indices predeclared max12; independent single-frame maps',goal=goal.tolist(),committed=sum(x['committed'] for x in rows),command_available=sum(x['command_available'] for x in rows),reasons=dict(Counter(x['event'].get('reason','PASS') for x in rows)),excluded=dict(excluded),rows=rows),indent=2))
 manifest={str(f):sha(f) for f in [run/'BIRTH_CERTIFICATE.json',run/'plant/PROCESS_START.json',run/'truth_data/EVENTS.jsonl',run/'supervisor_data/EVENTS.jsonl',run/'FAULT.json']}
 (out/'INPUT_SHA.json').write_text(json.dumps(manifest,indent=2))
 print('REPLAY',len(rows),'COMMITTED',sum(x['committed'] for x in rows),'REASONS',dict(Counter(x['event'].get('reason','PASS') for x in rows)))
if __name__=='__main__':main()
