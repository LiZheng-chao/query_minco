"""Non-driving native-rate bridge/guardian-process test, NO ROS or actuators."""
from pathlib import Path
import sys,json,time,collections,os
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';B=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
import numpy as np,torch
from experiments.sim_dev_oracle_v1.bridge import Engine

def main():
 torch.set_num_threads(1);run=O/'runtime_replay_release';run.mkdir();(run/'RUNTIME_PARAMETERS.json').write_text(json.dumps(dict(run_nonce='NONDRIVING_NATIVE_REPLAY',experiment_mode='SIM_DEV_ORACLE')))
 for ext in ['.json','.u8']:(run/('native_grid'+ext)).symlink_to(B/('capture_ordered/native_grid_GT_NOT_USED'+ext))
 rows=json.loads((B/'capture_ordered/INPUTS.json').read_text())['records'];start=rows[0]['stamp'];wall=time.monotonic();clock=lambda:start+time.monotonic()-wall;e=Engine(run,clock);wall=time.monotonic();events=[];samples=[];inputs=[]
 def sample():
  now=clock();cmd=e.command(now)
  # Eligibility sampling only: no fake PositionCommand publication or consumption.
  samples.append(dict(now=now,valid=cmd is not None,plan_id=cmd['plan_id'] if cmd else None,pending=len(e.pending),guardian_ready=e.guardian_ready));events.extend(e.take_events())
 try:
  for item in rows:
   while clock()<item['stamp']:sample();time.sleep(.001)
   s=item['stamp']
   try:e.state(dict(item['pose'],a=[0,0,0]),s,s);e.set_goal([3.25,-.25,2.],s);e.observe(np.load(item['path'])['depth'],item['pose'],s);inputs.append(dict(stamp=s,accepted=True))
   except Exception as ex:inputs.append(dict(stamp=s,accepted=False,reason=str(ex)))
   sample()
  end=clock()+.25
  while clock()<end:sample();time.sleep(.001)
 finally:e.close();events.extend(e.take_events())
 good=[x for x in samples if x['valid']];first=good[0]['now'] if good else None;steady=[x for x in samples if first is not None and first<=x['now']<=rows[-1]['stamp']];gap_start=None;longest=0.
 for x in steady:
  if not x['valid'] and gap_start is None:gap_start=x['now']
  if x['valid'] and gap_start is not None:longest=max(longest,x['now']-gap_start);gap_start=None
 if gap_start is not None:longest=max(longest,steady[-1]['now']-gap_start)
 result=dict(scope='NONDRIVING_RECORDED_NATIVE_RATE_WITH_SEPARATE_GT_PROCESS_NOT_CONTROLLER_OR_PLANT',frames=len(rows),inputs=inputs,events=events,samples=samples,distinct_eligible_plan_ids=len({x['plan_id'] for x in good}),steady_valid_fraction=sum(x['valid'] for x in steady)/len(steady) if steady else 0,first_ready_after_s=first-start if first else None,steady_longest_invalid_s=longest,queue_peak=max(x['pending'] for x in samples),evidence_overflow=e.evidence_overflow,counts=dict(collections.Counter(x['event'] for x in events)),physical_steps=0,controller_consumption='NOT_TESTED',STOP_plant='NOT_TESTED',guardian_process_exit=e.guard.exitcode,exit_code=0)
 (O/'NONDRIVING_RUNTIME_REPLAY.json').write_text(json.dumps(result,indent=2));print({k:result[k] for k in ['distinct_eligible_plan_ids','steady_valid_fraction','first_ready_after_s','steady_longest_invalid_s','queue_peak','counts']})
if __name__=='__main__':main()
