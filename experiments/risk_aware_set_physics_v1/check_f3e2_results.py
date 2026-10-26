"""Read-only reconstruction, all stage denominators and causal failure timeline."""
import collections,csv,hashlib,json,pickle
from pathlib import Path
import numpy as np
from query_minco.inference.proposal_worker import decode_body
from query_minco.inference.trajectory_execution import Program,MotionLimits,trim,concatenate
from query_minco.inference.candidate_feasibility import contribution
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.selector_safety import world_coefficients
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007';RUN=OUT/'shadow_inputs001'
def yes(x):return x is True or x=='true'
def quant(v):return np.percentile(v,[50,95,100]).tolist() if len(v) else None
def main():
 for p,h in json.loads((OUT/'REAL_SHADOW_BINDING.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 frames=[json.loads(s) for s in (RUN/'c1/events.jsonl').read_text().splitlines()]
 gate=[json.loads(s) for s in (RUN/'DYNAMIC_GATE.jsonl').read_text().splitlines()]
 native=json.loads((OUT/'NATIVE_MEASURED_RESULTS_FINAL.json').read_text())
 with (OUT/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:refs=pickle.load(f)
 celluntil=refs[0].valid_until_ros;volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],celluntil,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 odom=np.genfromtxt(RUN/'ODOM.csv',delimiter=',',names=True);cmd=np.genfromtxt(RUN/'COMMAND.csv',delimiter=',',names=True);events=np.genfromtxt(RUN/'EVENT.csv',delimiter=',',names=True)
 contributions=[];stages=collections.defaultdict(list);actual=[];direction=[];checks=0;sourcechecks=[]
 stages_count=collections.Counter();outcomes=collections.Counter();e2e=[];qualified=[];reserved=[];candidate_cpu=[];connwall=[]
 for f in frames:
  stages_count['decision_opportunities']+=1
  z=np.load(RUN/('c1/frame_%03d.npz'%f['frame']));assert z['solve_valid'].all() and len(z['coefficients'])==6
  stages_count['all_six_generated_frames']+=1;stages_count['generated_candidates']+=6
  # Independent actual100Hz source acceleration versus finite-camera-window input.
  same=odom[np.abs(odom['stamp']-f['source_stamp'])<1e-7]
  if len(same):
   a=np.array([same[-1][k] for k in ('a0','a1','a2')]);netacc=z['rotation_W_B0']@z['state'][3:6]
   sourcechecks.append(dict(frame=f['frame'],source_stamp=f['source_stamp'],instant_100hz_acc_norm=float(np.linalg.norm(a)),camera_interval_acc_norm=float(np.linalg.norm(netacc)),difference_norm=float(np.linalg.norm(a-netacc)),instant_source_acc_bound_pass=bool(np.linalg.norm(a)<=.05)))
  vectors=[]
  for qi in range(6):
   w=Program(world_coefficients(z['coefficients'][qi],z['position_W'],z['rotation_W_B0']),z['durations'][qi])
   v=w.pva(.75)[0]-w.pva(.7)[0];vectors.append(v/np.linalg.norm(v))
  angles=[np.rad2deg(np.arccos(np.clip(vectors[i]@vectors[j],-1,1))) for i in range(6) for j in range(i)]
  direction.append(dict(frame=f['frame'],retained_original_curve_directions=np.array(vectors).tolist(),pairwise_angle_deg_min_p50_max=np.percentile(angles,[0,50,100]).tolist(),actual_Query_executed=False))
  for q in f['queries']:
   outcomes[q.get('reason','FULL_PASS')]+=1
   if q.get('worker_timing'):
    for k,v in q['worker_timing'].items():
     if isinstance(v,dict):stages[k].append(v)
    reserved.append(q['worker_timing']['total_wall_ms'])
   if 'candidate_thread_cpu_ms' in q:candidate_cpu.append([q['candidate_total_ms'],q['candidate_thread_cpu_ms']])
   if 'connection_ms' in q:connwall.append(q['connection_ms'])
   if q.get('full_bank_and_sweep_pass'):stages_count['complete_56phase_and_sweep_candidates']+=1
   if q.get('cpp_submission_pass'):stages_count['cpp_verified_candidates']+=1
   if q['passed']:stages_count['deadline_ready_candidates']+=1
   if not q['passed']:continue
   with (RUN/('c1/banks_%03d.pkl'%f['frame'])).open('rb') as file:payload=pickle.load(file)[q['query']]
   bank,meta=payload['bank'],payload['envelope'];assert bank.digest==hashlib.sha256(bank.body).hexdigest()==meta['body_sha256']
   prefix,stops=decode_body(bank.body);prefix.validate(MotionLimits())
   for phase,stop in stops:
    full=stop if phase==0 else concatenate(trim(prefix,0,phase),stop)
    full.validate(MotionLimits());assert stop.extrema()[2]<=.35+1e-8
    assert volume.certify(full.coefficients,full.durations,bank.prepared_ros,required_until=bank.valid_until_ros)['status']=='CERTIFIED_FREE'
    np.testing.assert_allclose(stop.pva(0),prefix.pva(phase),atol=1e-8);np.testing.assert_allclose(stop.pva(stop.duration)[1:],0,atol=1e-8)
    assert bank.valid_until_ros<celluntil and bank.valid_until_mono<refs[0].valid_until_mono
    checks+=1
   assert len(stops)==56
   seq=meta['sequence'];ag=next(x for x in gate if x.get('type')=='ACTUAL_READONLY_GATE' and int(x['sequence'])==seq)
   actual.append(ag)
   if yes(ag['passed']):
    stages_count['actual_causal_gate_candidates']+=1;qualified.append(q['source_to_complete_ms'])
   e2e.append(q['source_to_complete_ms'])
   conn=trim(prefix,0,1);ret=Program(z['q%d_retained_coefficients'%q['query']],z['q%d_retained_durations'%q['query']]);brake=stops[-1][1]
   parts={name:contribution(p) for name,p in [('connection',conn),('original_MINCO',ret),('endpoint_stop',brake)]}
   parts['endpoint_stop']['distance_upper_bound_m']=brake.extrema()[0]*brake.duration
   total=sum(p['arc_length_estimate_m'] for p in parts.values())
   for p in parts.values():p['arc_fraction']=p['arc_length_estimate_m']/total
   contributions.append(dict(frame=f['frame'],query=q['query'],parts=parts,all_reserve_stop_distance_upper_bound_max=max(stop.extrema()[0]*stop.duration for _,stop in stops),actual_Query_executed=False))
 stageperf={name:{key:quant([v[key] for v in rows]) for key in ('wall_ms','thread_cpu_ms','runqueue_wait_ms','wall_minus_thread_cpu_ms')} for name,rows in stages.items()}
 n=len(frames);num=sum(q['passed'] for f in frames for q in f['queries']);times=[f['source_stamp'] for f in frames]
 camera=[json.loads(l) for l in (RUN/'CAMERA_TRACE.jsonl').read_text().splitlines()];ctimes=[x['stamp_ns']/1e9 for x in camera if 'stamp_ns' in x]
 originalBridge=(ROOT/'query_minco/inference/shadow_camera_low_speed.py').read_text();assert 'next_query+=200_000_000' in originalBridge
 fail=events[events['code']==42][0];names=[k+str(i) for k in ('p','v','a') for i in range(3)]
 m=np.array([fail[k] for k in names]).reshape(3,3);ref=np.array([fail['r'+k] for k in names]).reshape(3,3);dt=fail['x'];prediction=m.copy();prediction[0]+=m[1]*dt+.5*m[2]*dt*dt;prediction[1]+=m[2]*dt
 j=int(np.argmax(np.diff(cmd['begin'])));near=odom[(odom['stamp']>=fail['ros']-.05)&(odom['stamp']<=fail['ros']+.05)]
 causal=dict(first_failure_event=dict(ros=float(fail['ros']),mono=float(fail['mono']),code=42,measured_pva=m.tolist(),current_reference_pva=ref.tolist(),predicted_current_error=np.linalg.norm(prediction-ref,axis=1).tolist(),source_age_s=float(fail['x']),receive_age_s=float(fail['y'])),
 max_command_gap=dict(previous_ros=float(cmd[j]['begin']),next_ros=float(cmd[j+1]['begin']),gap_ms=float((cmd[j+1]['begin']-cmd[j]['begin'])*1000),wake_delay_ms=float((cmd[j+1]['mono']-cmd[j+1]['wake'])*1000)),
 odom_around_failure=[{k:float(r[k]) for k in ('stamp','recv','v0','a0')} for r in near],
 last_C1_source_before_failure_s=float(fail['ros']-frames[-1]['source_stamp']),
 interpretation='A single acceleration finite-difference excursion0.14218>0.1 triggered event42, immediately after33.650ms publication gap /13.761ms wake delay. Odom interval17.969ms carried small velocity change; adjacent a values return near-.16. Fixed-step simulator vs wall stamp is a concrete mechanism from source, but no runtime integration-step counter was saved: cannot uniquely attribute root cause to physics timing, Controller or callback scheduling. Not caused by a demonstrated long proposal computation; no algorithm/source condition relaxed.',
 backup_actual=native['backups'],stop_dwells=native['stop_dwells'])
 (OUT/'CONTROL_FAILURE_CAUSAL_REVIEW.json').write_text(json.dumps(causal,indent=2))
 stats={k:quant([f[k] for f in frames]) for k in ('network_ms','network_thread_cpu_ms','decode_export_ms','decode_thread_cpu_ms','six_generation_ms','source_to_decision_ms','six_policy_processing_ms','post_generation_ms','async_evidence_IO_ms')}
 stats.update(complete_bank_wall_ms=quant(reserved),connection_and_snapshot_wall_ms=quant(connwall),candidate_wall_ms=quant([x[0] for x in candidate_cpu]),candidate_thread_cpu_ms=quant([x[1] for x in candidate_cpu]))
 receipts=[q['cpp_receipt'] for f in frames for q in f['queries'] if q.get('cpp_submission_pass')]
 stats['cpp_independent_verify_wall_ms']=quant([float(r['wall_ms']) for r in receipts]);stats['cpp_independent_verify_thread_cpu_ms']=quant([float(r['thread_cpu_ms']) for r in receipts]);stats['cpp_verify_runqueue_wait_ms']=quant([float(r['runqueue_wait_ms']) for r in receipts])
 prereg=json.loads((OUT/'SHADOW_PREREG.json').read_text())
 reliability=bool(n>=20 and times[-1]-times[0]>=.5 and num/n>=.95 and sum(yes(a['passed']) for a in actual)/n>=.95 and np.diff(times).max()<=.1 and native['necessary_control_gates_pass'])
 result=dict(status='FAIL_BLOCKED_NATIVE_TRACKING_AND_INSUFFICIENT_OPPORTUNITIES',stages=dict(stages_count),
 ready_success_rate_observed=num/n,timeout_rate_observed=1-num/n,actual_causal_pass_observed=sum(yes(a['passed']) for a in actual)/n,
 qualified_source_to_complete_p50_p95_max_ms=quant(qualified),qualified_latency_sample_count=len(qualified),
 accepted_window_s=times[-1]-times[0],max_accepted_source_gap_s=float(np.diff(times).max()),required_opportunities=20,opportunity_count_pass=n>=20,source_gap_contract_pass=bool(np.diff(times).max()<=.1),reliability_pass=reliability,
 all_frame_timing_p50_p95_max_ms=stats,native_worker_stage_timing=stageperf,all_stop_rows_independently_rechecked=checks,
 outcomes=dict(outcomes),source_state_audit=sourcechecks,
 sensor_cadence=dict(native_config_fps=33,inherited_query_period_ns=200_000_000,observed_query_fps=1/float(np.median(np.diff(ctimes)))),actual_causal_gates=actual,
 record_limit='Python connection wall includes causal snapshot RPC; its thread CPU not separately bracketed. Candidate/network/decode and all native numerical stages wall/CPU recorded. schedstat is Linux runqueue waiting; wall-minusCPU is not uniquely GIL or scheduler. NativeBank total CPU includes timing dict construction while its wall ends before dict construction; use per-native-stage paired metrics for CPU/wait inference.',
 Query_control=False,first_3m_takeover='BLOCKED',native_control_result_file='NATIVE_MEASURED_RESULTS_FINAL.json')
 (OUT/'E_REALTIME_RESULTS.json').write_text(json.dumps(result,indent=2));(OUT/'NETWORK_CONTRIBUTION.json').write_text(json.dumps(dict(proposals=contributions,six_query_direction=direction,Query_executed_count=0),indent=2))
 with (OUT/'REALTIME_FRAME_STAGES.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['frame','source_stamp','all_six_generated','full_bank','cpp_submit','deadline_ready','actual_gate','source_to_complete_ms','selection'])
  writer.writeheader()
  for b in frames:
   q=next(q for q in b['queries'] if q['passed']);a=next(a for a in actual if int(a['sequence'])==b['frame']*6+q['query']+1)
   writer.writerow(dict(frame=b['frame'],source_stamp=b['source_stamp'],all_six_generated=True,full_bank=q['full_bank_and_sweep_pass'],cpp_submit=q['cpp_submission_pass'],deadline_ready=q['passed'],actual_gate=yes(a['passed']),source_to_complete_ms=q['source_to_complete_ms'],selection=b['selection']))
 print(json.dumps({k:v for k,v in result.items() if k in ('status','stages','ready_success_rate_observed','timeout_rate_observed','qualified_source_to_complete_p50_p95_max_ms','accepted_window_s','max_accepted_source_gap_s','all_frame_timing_p50_p95_max_ms','sensor_cadence','source_state_audit')},indent=2))
if __name__=='__main__':main()
