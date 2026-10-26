"""CPU-only frozen C1 replay with true source-time100Hz acceleration. No ROS."""
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np,torch
from query_minco.inference.shadow import make_state,FROZEN
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.proposal_worker import NativeWorker,decode_body
from query_minco.inference.candidate_feasibility import build_prefix
from query_minco.inference.selector_safety import world_coefficients
from query_minco.inference.trajectory_execution import Program,MotionLimits,trim,concatenate
from query_minco.inference.observed_space import *
from query_minco.inference.c1_proposal_shadow_v4 import corrected_main_source
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007';RUN=OUT/'shadow_inputs001'
def main():
 corrected=corrected_main_source();compile(corrected,'corrected_v4_main','exec');(OUT/'runtime/C1_SOURCE_FIXED_MAIN_REVIEW.py').write_text(corrected)
 o=np.genfromtxt(RUN/'ODOM.csv',delimiter=',',names=True);frames=[json.loads(x) for x in (RUN/'c1/events.jsonl').read_text().splitlines()]
 binding=next(b for b in json.loads(FROZEN.read_text()) if b['label']=='C1')
 adapter=InferenceAdapter(binding['checkpoint']['path'],binding['method_id'],binding['checkpoint']['sha256']);adapter.model.requires_grad_(False);torch.set_num_threads(1)
 worker=NativeWorker(OUT/'runtime/libproposal_numerics.so')
 with (OUT/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:refs=pickle.load(f)
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],refs[0].valid_until_ros,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 results=[]
 for b in frames:
  z=np.load(RUN/('c1/frame_%03d.npz'%b['frame']));actual=o[np.abs(o['stamp']-b['source_stamp'])<1e-7][-1];a=np.array([actual[k] for k in ('a0','a1','a2')]);v=np.array([actual[k] for k in ('v0','v1','v2')])
  row=dict(frame=b['frame'],true_source_acceleration=a.tolist(),old_network_acceleration=z['state'][3:6].tolist(),passed=False)
  if np.linalg.norm(a)>.05:row['reason']='TRUE_SOURCE_ACCELERATION_SUPPORT_REJECT';results.append(row);continue
  state,goal=make_state(z['position_W'],z['rotation_W_B0'],v,a,np.array([3,0,2]))
  begin=time.perf_counter()
  with torch.inference_mode():
   td=torch.from_numpy(z['depth'][None].copy());ts=torch.from_numpy(state[None].copy());raw=adapter.model(td,ts);decoded=adapter.post(raw,ts,num_samples=65);data=adapter.export(decoded,raw,decoded.samples)
  q=next(q for q in b['queries'] if q['passed']);w=Program(world_coefficients(data['coefficients'][0,5],z['position_W'],z['rotation_W_B0']),data['durations'][0,5])
  conn,ret,prefix=build_prefix(w,np.array(q['planned_head_pva']),1.,.7,.5);now=b['planning_ros'];mono=refs[0].prepared_mono+now-refs[0].prepared_ros
  bank=worker.prepare(prefix,volume,now,mono,inherited_mono_expiry=refs[0].valid_until_mono);elapsed=(time.perf_counter()-begin)*1000
  pre,stops=decode_body(bank.body)
  for phase,stop in stops:
   full=stop if phase==0 else concatenate(trim(pre,0,phase),stop);full.validate(MotionLimits())
   assert volume.certify(full.coefficients,full.durations,now,required_until=bank.valid_until_ros)['status']=='CERTIFIED_FREE'
  row.update(passed=True,phases=len(stops),CPU_generation_and_bank_ms=elapsed,raw_difference_from_zero_acc_input=float(np.max(np.abs(data['raw'][0]-z['raw']))),source_state_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest())
  results.append(row)
 report=dict(mode='OFFLINE_DIAGNOSTIC_CORRECTED_INPUT_NOT_REALTIME',rows=results,source_supported=sum(r['passed'] for r in results),source_rejected=sum(not r['passed'] for r in results),real_corrected_Shadow='NOT_RUN',frozen_model=binding,Query_control=False)
 (OUT/'SOURCE_CORRECTED_REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('rows','frozen_model')},indent=2))
if __name__=='__main__':main()
