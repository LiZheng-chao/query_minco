"""All 48 original candidates. OFFLINE_DIAGNOSTIC; no shared deadline, no ROS."""
import argparse,collections,csv,itertools,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from query_minco.inference.trajectory_execution import Program,Measurement,MotionLimits,concatenate
from query_minco.inference.selector_safety import world_coefficients
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope,ErrorBudget
from query_minco.inference.constrained_prefix import TEMPLATES
from query_minco.inference.committed_reference import reserved_stop
from query_minco.inference.candidate_feasibility import build_prefix,soft_jerk_witness,contribution,complete_bank
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',required=True);a=p.parse_args()
 src=Path(a.source);out=Path(a.run)/'offline';out.mkdir(exist_ok=True)
 with (src/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:bundles=pickle.load(f)
 proof=json.loads(bundles[0].prefix_proof)
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],proof['valid_until'],proof['evidence'][0]['cell']),),Scope.EXTERNAL_SUPERVISOR,ErrorBudget(**proof['error_budget']))
 with (src/'shadow_inputs001/EVENT.csv').open() as f:epoch=next(r for r in csv.DictReader(f) if r['code']=='11' and r['bundle']=='1')
 er,em=float(epoch['y']),float(epoch['z'])
 with (src/'shadow_inputs001/ODOM.csv').open() as f:odom=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
 events=[json.loads(s) for s in (src/'shadow_inputs001/c1/events.jsonl').read_text().splitlines()]
 templates=list(TEMPLATES)
 # Fixed finite comparison; original training Goal/model and hard limits untouched.
 extra=list(itertools.product((1.5,2.,2.5,3.),(.4,.7,1.,1.3),(.15,.25,.4)))
 templates+= [x for x in extra if x not in templates]
 rows=[];attempts=[];started=time.monotonic()
 with (out/'ATTEMPTS.jsonl').open('x',buffering=1) as log:
  for ev in events:
   fi=ev['frame'];z=np.load(src/('shadow_inputs001/c1/frame_%03d.npz'%fi))
   saved=Program(z['q0_prefix_coefficients'],z['q0_prefix_durations']);head=saved.pva(0)
   phase=brentq(lambda t:bundles[1].prefix.pva(t)[0,0]-head[0,0],5.,11.,xtol=1e-10)
   activation=er+phase;now=activation-.06;mono=em+phase-.06
   causal=[r for r in odom if r['stamp']<=now and r['recv']<=mono]
   m=causal[-1];mpva=np.array([[m[k+str(i)] for i in range(3)] for k in ('p','v','a')])
   measurement=Measurement(m['stamp'],mpva);residual=np.linalg.norm(measurement.predict(activation)-head,axis=1)
   headres=np.linalg.norm(bundles[1].prefix.pva(phase)-head,axis=1)
   for qi in range(6):
    begin=time.perf_counter();world=Program(world_coefficients(z['coefficients'][qi],z['position_W'],z['rotation_W_B0']),z['durations'][qi])
    rawext=world.extrema();reasons=collections.Counter();best=None;old_pass=False
    row=dict(mode='OFFLINE_DIAGNOSTIC',frame=fi,query=qi,raw_extrema=rawext,
      raw_hard_kinematic_pass=bool(np.all(np.array(rawext)<=[1,1.5,10])),
      source_stamp=float(z['source_stamp']),planning_ros=now,activation_ros=activation,
      source_age_ms=(now-float(z['source_stamp']))*1000,measurement_stamp=m['stamp'],measurement_received_mono=m['recv'],
      measurement_age_ms=(now-m['stamp'])*1000,predicted_head_residual=residual.tolist(),
      saved_head_reconstruction_error=headres.tolist(),input_ood=ev['local_goal'].get('ood',[]),
      causal_measurement_gate=bool(np.all(residual<=[.02,.05,.1]) and 0<=now-m['stamp']<=.05),
      certificate_original_expiry=proof['valid_until'],remaining_lifetime_s=proof['valid_until']-now,
      head_pva=head.tolist(),terminal_pva=world.pva(world.duration).tolist(),passed=False)
    for ti,(ct,sp,rate) in enumerate(templates):
     tick=time.perf_counter();r=dict(frame=fi,query=qi,template=ti,connection_s=ct,source_phase=sp,rate=rate,passed=False)
     try:
      conn,ret,prefix=build_prefix(world,head,ct,sp,rate)
      ext=prefix.extrema();r.update(prefix_extrema=ext,connection_extrema=conn.extrema(),retimed_extrema=ret.extrema(),terminal_pva=prefix.pva(prefix.duration).tolist())
      if np.any(np.array(ext)>np.array([1,1.5,10])+1e-8):raise ValueError('KINEMATIC_LIMIT')
      r['endpoint_kinematic_pass']=True
      progress=np.linalg.norm(ret.pva(ret.duration)[0]-ret.pva(0)[0])
      if progress<.005:raise ValueError('INSUFFICIENT_ORIGINAL_MINCO_PROGRESS')
      witness=soft_jerk_witness(prefix);r['soft_jerk_witness']=witness
      if witness:raise ValueError('SOFT_JERK_NECESSARY_SPEED_VIOLATION')
      brake=reserved_stop(prefix.pva(prefix.duration));r['endpoint_brake_extrema']=brake.extrema()
      r['endpoint_brake_duration_s']=brake.duration
      full=concatenate(prefix,brake);gp=volume.certify(full.coefficients,full.durations,now,required_until=activation+full.duration)
      r['endpoint_swept_body']=gp
      if gp['status']!='CERTIFIED_FREE':raise ValueError('GEOMETRY_'+gp['reason'])
      b0=time.perf_counter()
      try:bank=complete_bank(prefix,volume,now,mono)
      finally:r['complete_bank_ms']=(time.perf_counter()-b0)*1000
      r['passed']=True;r['bank_phases']=len(bank.stops);r['bound_expiry']=bank.valid_until_ros
      r['complete_horizon_s']=max(s.phase+s.polynomial.duration for s in bank.stops)
      r['parts']={k:contribution(p) for k,p in [('connection',conn),('original_MINCO_retained',ret),('endpoint_brake',brake)]}
      r['all_phase_brake_max_extrema']=[max(Program(np.frombuffer(s.polynomial.coefficients,dtype='<f8').reshape(-1,3,6),s.polynomial.durations).extrema()[k] for s in bank.stops) for k in range(3)]
      if ti<len(TEMPLATES):old_pass=True
      if best is None:
       best=r.copy()
       np.savez(out/('candidate_f%02d_q%d.npz'%(fi,qi)),prefix_coefficients=prefix.coefficients,prefix_durations=prefix.durations,connection_coefficients=conn.coefficients,connection_durations=conn.durations,retained_coefficients=ret.coefficients,retained_durations=ret.durations,brake_coefficients=brake.coefficients,brake_durations=brake.durations)
     except ValueError as ex:r['reason']=str(ex);reasons[str(ex)]+=1
     r['compute_ms']=(time.perf_counter()-tick)*1000;log.write(json.dumps(r)+'\n');attempts.append(r)
     # One complete proof per candidate suffices; all original templates still checked
     # until first success, then stop the bounded comparison.
     if r['passed']:break
    row.update(passed=best is not None,original_template_pass=old_pass,best=best,reasons=dict(reasons),compute_ms=(time.perf_counter()-begin)*1000,templates_attempted=sum(r['frame']==fi and r['query']==qi for r in attempts))
    rows.append(row);(out/'CANDIDATES.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(frame=fi,query=qi,passed=row['passed'],old=old_pass,reasons=dict(reasons),elapsed=round(time.monotonic()-started,2))),flush=True)
 summary=dict(mode='OFFLINE_DIAGNOSTIC',frames=len(events),candidates=len(rows),complete_cpu_pass=sum(r['passed'] for r in rows),original_template_pass=sum(r['original_template_pass'] for r in rows),causal_state_pass=sum(r['causal_measurement_gate'] for r in rows),live_authorization=False,query_control=False,original_cell_expiry=proof['valid_until'],historical_evidence_expired_now=True,attempts=len(attempts),attempt_reasons=dict(collections.Counter(r.get('reason','FULL_PASS') for r in attempts)),elapsed_s=time.monotonic()-started,
 provenance_caveat='Historical C1 latest-state callback cache was not saved. Replay uses latest causally received CPP ODOM and saved command head, never future Odom. Saved control cell receipt reused at historical times; original transient C1 cell was not persisted. This is not bit-identical online replay or fresh authorization.')
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
