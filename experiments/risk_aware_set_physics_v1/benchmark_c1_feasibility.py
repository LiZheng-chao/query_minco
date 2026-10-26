"""CPU-only timing of frozen six-query inference and one complete safety bank."""
import json,time
from pathlib import Path
import numpy as np,torch
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.shadow import FROZEN
from query_minco.inference.trajectory_execution import Program
from query_minco.inference.selector_safety import world_coefficients
from query_minco.inference.observed_space import *
from query_minco.inference.candidate_feasibility import build_prefix,prepare_fast_bound_reference
def main():
 root=Path('/home/lizheng/query_minco_mainline');out=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution006'
 old=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution005'
 rows=json.loads((out/'offline/CANDIDATES.json').read_text());torch.set_num_threads(1)
 binding=next(r for r in json.loads(FROZEN.read_text()) if r['label']=='C1')
 adapter=InferenceAdapter(binding['checkpoint']['path'],binding['method_id'],binding['checkpoint']['sha256'])
 adapter.model.requires_grad_(False)
 first=np.load(old/'shadow_inputs001/c1/frame_000.npz')
 with torch.inference_mode():
  for _ in range(4):
   td=torch.from_numpy(first['depth'][None].copy());ts=torch.from_numpy(first['state'][None].copy())
   raw=adapter.model(td,ts);adapter.post(raw,ts,num_samples=65)
 results=[]
 for fi in range(8):
  z=np.load(old/('shadow_inputs001/c1/frame_%03d.npz'%fi))
  r=next(x for x in rows if x['frame']==fi and x['query']==5)
  v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],r['certificate_original_expiry'],'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
  for repeat in range(3):
   t0=time.perf_counter()
   with torch.inference_mode():
    td=torch.from_numpy(z['depth'][None].copy());ts=torch.from_numpy(z['state'][None].copy())
    raw=adapter.model(td,ts);t1=time.perf_counter()
    decoded=adapter.post(raw,ts,num_samples=65);data=adapter.export(decoded,raw,decoded.samples);t2=time.perf_counter()
   w=Program(world_coefficients(data['coefficients'][0,5],z['position_W'],z['rotation_W_B0']),data['durations'][0,5])
   row=dict(frame=fi,repeat=repeat,mode='OFFLINE_CPU_BENCHMARK_NOT_ROS',network_ms=(t1-t0)*1000,decode_ms=(t2-t1)*1000,
      raw_max_difference=float(np.max(np.abs(data['raw'][0]-z['raw']))),all_six_generated=int(data['coefficients'].shape[1]),passed=False)
   try:
    connection,retained,prefix=build_prefix(w,np.array(r['head_pva']),1.,.7,.5);t3=time.perf_counter()
    b=prepare_fast_bound_reference(prefix,v,r['planning_ros'],0.);t4=time.perf_counter()
    row.update(passed=True,connection_ms=(t3-t2)*1000,bank_ms=(t4-t3)*1000,post_generation_ms=(t4-t2)*1000,total_ms=(t4-t0)*1000,phases=len(b.stops))
   except ValueError as e:row.update(reason=str(e),post_generation_ms=(time.perf_counter()-t2)*1000,total_ms=(time.perf_counter()-t0)*1000)
   results.append(row)
 summary=dict(mode='OFFLINE_CPU_BENCHMARK_NOT_ROS',model=binding,rows=results,
   passed=sum(r['passed'] for r in results),post_generation_budget_ms=40.,
   post_generation_p50_p95_max=np.percentile([r['post_generation_ms'] for r in results],[50,95,100]).tolist(),
   six_generation_plus_complete_q5_p50_p95_max=np.percentile([r['total_ms'] for r in results],[50,95,100]).tolist(),
   qualified_realtime_latency=None,policy='All six generated. Deterministic q5 first, fixed (1s,.7,.5); other queries NOT checked after first complete success. No online guarantee if missed deadline; selection NONE.')
 (out/'offline/END_TO_END_CPU_BENCHMARK.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','model')}))
if __name__=='__main__':main()
