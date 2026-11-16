"""NONPHYSICAL scheduling fixture, not real forest feasibility."""
import sys,time,json
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.observed_space_v1.map import Snapshot,Config,KNOWN_FREE
def main():
 rows=[]
 for moving in [False,True]:
  e=Engine(lambda now:None)
  def certified_fixture(now):
   time.sleep(.04) # explicit compute-latency injection to test epoch fencing
   stamp=e.ledger.last_stamp
   return Snapshot(np.full((40,40,30),KNOWN_FREE,np.uint8),[-2,-2,-1.5],Config(),
                   [dict(id='SYNTHETIC_CERTIFIED_CELL_FIXTURE',stamp=stamp)],now,e.ledger.epoch,lambda:e.ledger.epoch)
  e.ledger.snapshot=certified_fixture
  e.start();begin=time.perf_counter();calls=[];published=[];sensor_index=-1
  try:
   for tick in range(150):
    until=begin+tick*.01
    if until>time.perf_counter():time.sleep(until-time.perf_counter())
    now=time.time();index=int((time.perf_counter()-begin)*33)
    state=dict(p=[index*.001 if moving else 0.,0.,0.],v=[0.,0.,0.],a=[0.,0.,0.],q=[0.,0.,0.,1.])
    e.set_state(state,now);e.set_goal([.6,0,0],now)
    if index!=sensor_index:
     e.observe(np.full((90,160),5.,np.float32),state,now,now);sensor_index=index
    t=time.perf_counter()
    e.with_current(now,lambda cmd:published.append(cmd))
    calls.append(time.perf_counter()-t)
  finally:e.close()
  xs=sorted(calls)
  rows.append(dict(scope='SYNTHETIC_SCHEDULING_ONLY_40ms_WORKER_DELAY',
   pose_changes=moving,control_calls=len(calls),commands_available=len(published),
   commit_count=e.counts['CERTIFIED_COMMITTED'],reject_reasons=dict(Counter(x.get('reason') for x in e.events if x['event']=='REJECTED')),
   control_ms=dict(p50=xs[round((len(xs)-1)*.5)]*1000,p95=xs[round((len(xs)-1)*.95)]*1000,p99=xs[round((len(xs)-1)*.99)]*1000,max=max(xs)*1000),
   control_10ms_violations=sum(x>.01 for x in calls),events=list(e.events)))
 Path('research_outputs/t2_depth_minco_20261116_v1/ASYNC_BENCHMARK.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in rows],indent=2))
if __name__=='__main__':main()
