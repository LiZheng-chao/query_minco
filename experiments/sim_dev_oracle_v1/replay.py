"""Actual saved native inputs, no actuator interfaces."""
from pathlib import Path
import sys,json,time,hashlib,collections
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';B=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
import numpy as np,torch
from experiments.sim_dev_oracle_v1.planner import CandidatePlanner
from experiments.sim_dev_oracle_v1.guardian import Guardian
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot

def main():
 torch.set_num_threads(1);records=json.loads((B/'capture_ordered/INPUTS.json').read_text())['records'];planner=CandidatePlanner();guardian=Guardian(B/'capture_ordered/native_grid_GT_NOT_USED','SIM_OFFLINE_ONLY');rows=[]
 for item in records:
  stamp=item['stamp'];depth=np.load(item['path'])['depth'];assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'];state=dict(item['pose'],a=[0.,0.,0.]);row=dict(frame=item['index'],stamp=stamp,input_sha=item['sha256']);start=time.perf_counter()
  try:
   plan,info=planner.propose(depth,item['pose'],state,[3.25,-.25,2.],stamp,allow_diagnostic=True);row.update(candidate=info,plan=plan.record,plan_id=plan.id,maxima=plan.maxima)
   tick=time.perf_counter()
   try:row['oracle']=guardian.check(plan.record,state,stamp,stamp,item['index']);row['status']='PREFIX_AND_BRAKE_MODEL_ACCEPTED_SIM_ONLY'
   except Exception as ex:row.update(status='ORACLE_REJECTED',reason=str(ex))
   row['oracle_s']=time.perf_counter()-tick
  except Exception as ex:row.update(status='NO_CANDIDATE',reason=str(ex))
  strict_sensor=planner.ledger.snapshot(stamp);strict_snap=authorized_snapshot(strict_sensor,BirthGrant((.25,-.25,2.),.6,stamp+.2,'OFFLINE_ONLY'),stamp,strict_sensor.epoch,lambda:strict_sensor.epoch);row['strict_probe_results']=[]
  for distance in [.6,.8]:
   probe=build(np.array(state['p']),np.zeros(3),np.zeros(3),state['q'],np.array(state['p'])+[distance,0,0],stamp)
   try:proof=strict_snap.certify(probe,stamp);row['strict_probe_results'].append(dict(distance=distance,status='CERTIFIED',proof=proof))
   except Exception as ex:row['strict_probe_results'].append(dict(distance=distance,status='REJECTED',reason=str(ex),detail=getattr(ex,'record',None)))
  try:
   sim_plan,sim_info=planner.propose(depth,item['pose'],state,[3.25,-.25,2.],stamp,allow_diagnostic=True,prefer_strict=False);row['runtime_diagnostic_candidate']=sim_info;row['runtime_diagnostic_plan']=sim_plan.record;row['runtime_diagnostic_oracle']=guardian.check(sim_plan.record,state,stamp,stamp,item['index'])
  except Exception as ex:row['runtime_diagnostic_rejection']=str(ex)
  row['total_s']=time.perf_counter()-start;rows.append(row)
 result=dict(scope='SAVED_NATIVE_SENSOR_REPLAY_NO_ROS_NO_PLANT_NO_ARM',frames=len(records),counts=dict(collections.Counter(x['status'] for x in rows)),strict_depth_unchanged=True,strict_candidate_count=sum(x.get('candidate',{}).get('mode')=='STRICT_DEPTH_CANDIDATE' for x in rows),diagnostic_candidate_count=sum('plan' in x for x in rows),runtime_diagnostic_candidates=sum('runtime_diagnostic_plan' in x for x in rows),runtime_diagnostic_prefix_accepts=sum('runtime_diagnostic_oracle' in x for x in rows),oracle_accepted=sum('oracle' in x for x in rows),physical_steps=0,rows=rows,exit_code=0)
 (O/'SIM_REPLAY.json').write_text(json.dumps(result,indent=2));print(result['counts'],'CANDIDATES',result['diagnostic_candidate_count'],'ORACLE',result['oracle_accepted'])
 # Strict maximum-range rule is NOT changed: real CUDA has a near thin-voxel
 # miss returning20, so a valid native range-clear certificate is unavailable.
 prior=json.loads((R/'research_outputs/b0_sensor_observability_20261118_v1/B0_EXACT_RAY_REFERENCE.json').read_text());(O/'MAX_RANGE_REPLAY_RESULT.json').write_text(json.dumps(dict(frames=69,input_manifest=str(B/'capture_ordered/INPUTS.json'),native_content_unique=prior['unique_depth_contents'],max_range_clear_ray_enabled=False,reason='ACTUAL_CUDA_THIN_OCCUPIED_VOXEL_MISS_COUNTEREXAMPLE_NO_CONTINUOUS_RAY_COVERAGE',strict_06_08_results={str(d):sum(x['status']=='CERTIFIED' for row in rows for x in row['strict_probe_results'] if x['distance']==d) for d in [.6,.8]},per_frame_results='SIM_REPLAY.json',strict_reference_bound_file=str(R/'research_outputs/b0_sensor_observability_20261118_v1/B0_EXACT_RAY_REFERENCE.json'),sim_diagnostic_candidate_count=result['runtime_diagnostic_candidates'],sim_prefix_guard_accept_count=result['runtime_diagnostic_prefix_accepts'],not_physical_PASS=True,exit_code=0),indent=2))
if __name__=='__main__':main()
