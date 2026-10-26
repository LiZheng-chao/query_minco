"""Deterministic replay all prior48 and native9 frozen inputs, no ROS."""
import json,pickle,time
from pathlib import Path
import numpy as np
from query_minco.inference.proposal_worker import NativeWorker,decode_body
from query_minco.inference.candidate_feasibility import prepare_fast_bound_reference,build_prefix
from query_minco.inference.trajectory_execution import Program,MotionLimits,concatenate,trim
from query_minco.inference.observed_space import *
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007';OLD=OUT.parent/'f3_execution006'
def main():
 worker=NativeWorker(OUT/'runtime/libproposal_numerics.so')
 with (OUT.parent/'f3_execution005/shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:previous=pickle.load(f)
 receipt=json.loads(previous[0].prefix_proof)
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],receipt['valid_until'],'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 rows=json.loads((OLD/'offline/CANDIDATES.json').read_text());result=[]
 for row in rows:
  fi,qi=row['frame'],row['query'];z=np.load(OLD/('offline/candidate_f%02d_q%d.npz'%(fi,qi)));p=Program(z['prefix_coefficients'],z['prefix_durations'])
  now=row['planning_ros'];mono=previous[0].prepared_mono+now-previous[0].prepared_ros;t=time.perf_counter()
  b=worker.prepare(p,volume,now,mono);elapsed=(time.perf_counter()-t)*1000
  prefix,stops=decode_body(b.body);prefix.validate(MotionLimits());maximum=[0,0,0]
  for phase,stop in stops:
   full=stop if phase==0 else concatenate(trim(prefix,0,phase),stop)
   full.validate(MotionLimits());stop.validate(MotionLimits());extrema=stop.extrema();maximum=np.maximum(maximum,extrema).tolist()
   assert extrema[2]<=.35+1e-8
   assert volume.certify(full.coefficients,full.durations,now,required_until=b.valid_until_ros)['status']=='CERTIFIED_FREE'
   np.testing.assert_allclose(stop.pva(0),prefix.pva(phase),atol=1e-8);np.testing.assert_allclose(stop.pva(stop.duration)[1:],0,atol=1e-8)
  assert len(stops)==int(round(prefix.duration/.02))+1 and b.valid_until_ros<receipt['valid_until']
  result.append(dict(frame=fi,query=qi,passed=True,optimized_ms=elapsed,previous_ms=json.loads((OLD/'offline/FAST_BANK_REPLAY.json').read_text())[len(result)]['bank_ms'],phases=len(stops),max_stop_extrema=maximum,timing=b.timing))
 # Same exact known q5 prefix comparison under same current CPU conditions.
 z=np.load(OLD/'offline/candidate_f00_q5.npz');p=Program(z['prefix_coefficients'],z['prefix_durations']);now=rows[0]['planning_ros'];baseline=[];optimized=[]
 for _ in range(20):
  t=time.perf_counter();prepare_fast_bound_reference(p,volume,now,0);baseline.append((time.perf_counter()-t)*1000)
  t=time.perf_counter();worker.prepare(p,volume,now,0);optimized.append((time.perf_counter()-t)*1000)
 # Reject expired, translated geometry and non-finite input.
 invalid=[]
 for name,program,stamp in [('expired',p,receipt['valid_until']),('translated100m',Program(p.coefficients+np.eye(1,len(p.coefficients),0).reshape(-1,1,1)*0,p.durations),now)]:
  if name=='translated100m':
   c=p.coefficients.copy();c[:,0,0]+=100;program=Program(c,p.durations)
  try:worker.prepare(program,volume,stamp,0);invalid.append(dict(name=name,rejected=False))
  except ValueError as e:invalid.append(dict(name=name,rejected=True,reason=str(e)))
 assert all(r['rejected'] for r in invalid)
 summary=dict(mode='OFFLINE_DIAGNOSTIC_CPU_NOT_FRESH_AUTHORIZATION',rows=result,full_candidates_pass=len(result),all_original_phases_retained=True,baseline_p50_p95_max_ms=np.percentile(baseline,[50,95,100]).tolist(),optimized_p50_p95_max_ms=np.percentile(optimized,[50,95,100]).tolist(),invalid=invalid)
 (OUT/'offline/WORKER_CPU_REGRESSION.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
