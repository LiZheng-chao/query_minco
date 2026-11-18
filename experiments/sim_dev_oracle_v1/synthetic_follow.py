"""Synthetic ideal-following timing isolation. NOT native flight qualification."""
from pathlib import Path
import sys,json,time,collections
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';B=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
import numpy as np,torch
from experiments.sim_dev_oracle_v1.bridge import Engine

def main():
 torch.set_num_threads(1);run=O/'synthetic_follow_release';run.mkdir();(run/'RUNTIME_PARAMETERS.json').write_text(json.dumps(dict(run_nonce='abcdef0123456789',experiment_mode='SIM_DEV_ORACLE')))
 for ext in ['.json','.u8']:(run/('native_grid'+ext)).symlink_to(B/('capture_ordered/native_grid_GT_NOT_USED'+ext))
 start=100.;wall=time.monotonic();clock=lambda:start+time.monotonic()-wall;e=Engine(run,clock);wall=time.monotonic();next_depth=100.;events=[];samples=[];first=None;last_valid=None;reason=None
 try:
  while clock()<105.:
   n=clock();last=e.last_executed;state_pva=last.at(n) if last else np.array([[.25,-.25,2.],[0,0,0],[0,0,0]]);state=dict(p=state_pva[0].tolist(),v=state_pva[1].tolist(),a=state_pva[2].tolist(),q=[0,0,0,1],frame='world')
   e.state(state,n,n);e.set_goal([3.25,-.25,2.],n)
   if n>=next_depth:
    # Synthetic plane at worldX5, camera identity; no old native Depth is moved.
    depth=np.full((90,160),5.-state['p'][0],np.float32);e.observe(depth,state,n);next_depth=n+.0303
   command=e.command(clock())
   if command is not None:
    e.published(command);first=n if first is None else first;last_valid=n
   samples.append(dict(t=n,valid=command is not None,p=state['p'],plan_id=command['plan_id'] if command else None,pending=len(e.pending)))
   events.extend(e.take_events())
   if last_valid is not None and n-last_valid>.15:reason='LEASE_CONTINUITY_GAP_GT_STOP_REQUIRED';break
   if state['p'][0]-.25>=.80:reason='SYNTHETIC_TARGET_REACHED';break
   time.sleep(.001)
 finally:e.close();events.extend(e.take_events())
 eligible=[x for x in samples if first is not None and x['t']>=first];report=dict(scope='SYNTHETIC_IDEAL_FOLLOWER_ANALYTIC_DEPTH_NOT_NATIVE_PHYSICAL_QUALIFICATION',reason=reason,steady_valid_fraction=sum(x['valid'] for x in eligible)/len(eligible) if eligible else 0,progress_m=samples[-1]['p'][0]-.25,distinct_simulated_command_ids=len({x['plan_id'] for x in samples if x['valid']}),events=events,samples=samples,counts=dict(collections.Counter(x['event'] for x in events)),rejections=dict(collections.Counter(x.get('reason') for x in events if x['event']=='SIM_REJECTED')),queue_peak=max(x['pending'] for x in samples),guardian_exit=e.guard.exitcode,physical_steps=0,exit_code=0)
 (O/'SYNTHETIC_FOLLOW_RESULT.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['events','samples']})
if __name__=='__main__':main()
