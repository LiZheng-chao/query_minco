"""Nine true execution006 frames, historical known commands and causal PVA."""
import json,pickle,time
from pathlib import Path
import numpy as np
from query_minco.inference.candidate_feasibility import build_prefix
from query_minco.inference.proposal_worker import NativeWorker,decode_body
from query_minco.inference.selector_safety import world_coefficients
from query_minco.inference.trajectory_execution import Program,MotionLimits,Measurement,trim,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007';OLD=OUT.parent/'f3_execution006'
def main():
 worker=NativeWorker(OUT/'runtime/libproposal_numerics.so')
 with (OLD/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:refs=pickle.load(f)
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],refs[0].valid_until_ros,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 events=[json.loads(s) for s in (OLD/'shadow_inputs001/c1/events.jsonl').read_text().splitlines()];rows=[]
 for e in events:
  q=next(q for q in e['queries'] if q['query']==5);z=np.load(OLD/('shadow_inputs001/c1/frame_%03d.npz'%e['frame']))
  w=Program(world_coefficients(z['coefficients'][5],z['position_W'],z['rotation_W_B0']),z['durations'][5])
  conn,ret,prefix=build_prefix(w,np.array(q['planned_head_pva']),1,.7,.5)
  m=Measurement(q['measurement_stamp'],q['measurement_pva'])
  residual=np.linalg.norm(m.predict(e['activation_ros'])-prefix.pva(0),axis=1);assert np.all(residual<=[.02,.05,.1])
  now=e['planning_ros'];mono=refs[0].prepared_mono+now-refs[0].prepared_ros
  b=worker.prepare(prefix,volume,now,mono,inherited_mono_expiry=refs[0].valid_until_mono)
  pre,stops=decode_body(b.body);pre.validate(MotionLimits())
  for phase,stop in stops:
   full=stop if phase==0 else concatenate(trim(pre,0,phase),stop)
   full.validate(MotionLimits());assert stop.extrema()[2]<=.35+1e-8
   assert volume.certify(full.coefficients,full.durations,now,required_until=b.valid_until_ros)['status']=='CERTIFIED_FREE'
   np.testing.assert_allclose(stop.pva(0),pre.pva(phase),atol=1e-8);np.testing.assert_allclose(stop.pva(stop.duration)[1:],0,atol=1e-8)
  rows.append(dict(frame=e['frame'],phases=len(stops),causal_head_error=residual.tolist(),original_native_bank_ms=q['complete_bank_ms'],new_isolated_CPU_ms=b.timing['total_wall_ms'],timing=b.timing,passed=True))
 report=dict(mode='OFFLINE_DIAGNOSTIC_NOT_ROS',passed=len(rows),frames=len(events),rows=rows,provenance='Uses exact saved raw inputs, original planned command head, causal measured PVA. Failed originals did not save checknow: historical planning_ros used; not bit-identical online timing replay.')
 (OUT/'offline/REAL_INPUT_REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(frames=len(rows),passed=len(rows),new_CPU_P95_max_ms=np.percentile([r['new_isolated_CPU_ms'] for r in rows],[95,100]).tolist())))
if __name__=='__main__':main()
