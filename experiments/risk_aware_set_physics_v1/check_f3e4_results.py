"""Independent E4 real receipts, all18 input fields and13 complete banks."""
import collections,csv,hashlib,json,pickle
from pathlib import Path
import numpy as np
from query_minco.inference.shadow import make_state
from query_minco.inference.proposal_worker import decode_body
from query_minco.inference.trajectory_execution import MotionLimits,Program,trim,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.candidate_feasibility import contribution
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution008';RUN=OUT/'shadow_inputs001'
class NumpyEncoder(json.JSONEncoder):
 def default(self,value):
  if isinstance(value,np.generic):return value.item()
  if isinstance(value,np.ndarray):return value.tolist()
  return super().default(value)
def yes(v):return v is True or v=='true'
def quant(v):return np.percentile(v,[50,95,100]).tolist() if len(v) else None
def main():
 for p,h in json.loads((OUT/'REAL_SHADOW_BINDING.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 frames=[json.loads(s) for s in (RUN/'c1/events.jsonl').read_text().splitlines()]
 ledger=[json.loads(s) for s in (RUN/'c1/INPUT_LEDGER.jsonl').read_text().splitlines()]
 gates=[json.loads(s) for s in (RUN/'DYNAMIC_GATE.jsonl').read_text().splitlines()]
 camera=[json.loads(s) for s in (RUN/'CAMERA_TRACE.jsonl').read_text().splitlines()]
 node=json.loads((RUN/'c1/RESULT.json').read_text());native=json.loads((OUT/'NATIVE_MEASURED_RESULTS_FINAL.json').read_text())
 events=np.genfromtxt(RUN/'EVENT.csv',delimiter=',',names=True);odom=np.genfromtxt(RUN/'ODOM.csv',delimiter=',',names=True);cmd=np.genfromtxt(RUN/'COMMAND.csv',delimiter=',',names=True)
 fail=events[events['code']==42][0];failure=float(fail['ros'])
 with (OUT/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:references=pickle.load(f)
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],references[0].valid_until_ros,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 staged=collections.Counter();fields=[];receipts=[];outcomes=collections.Counter();qualified=[];parts=[];stopchecks=0;actual=[];network_inputs=[r for r in ledger if r['kind']=='NETWORK_INPUT'];directions=[]
 for b in frames:
  fi=b['frame'];z=np.load(RUN/('c1/frame_%03d.npz'%fi));inputrow=next(r for r in network_inputs if r['frame']==fi)
  state=np.array(z['state']);pva=z['source_pva'];previous=float(z['source_previous_stamp']);past=z['source_previous_velocity'];stamp=float(z['source_stamp'])
  assert previous<stamp and .005<=stamp-previous<=.05
  np.testing.assert_allclose((pva[1]-past)/(stamp-previous),pva[2],atol=1e-7)
  expected,_=make_state(z['position_W'],z['rotation_W_B0'],pva[1],pva[2],np.array([3,0,2]))
  np.testing.assert_array_equal(state,expected)
  assert inputrow['source_received_mono']<=inputrow['network_start_mono']
  digest=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest()
  assert digest==inputrow['state_sha256']
  assert hashlib.sha256(z['depth'].astype('<f4').tobytes()).hexdigest()==inputrow['input_depth_mask_sha256']
  assert len(z['coefficients'])==6 and z['solve_valid'].all()
  # Unrounded expected values expose per-field FP32 error, not just array equality.
  delta=np.array([3,0,2])-z['position_W'];R=z['rotation_W_B0']
  exact=np.r_[R.T@pva[1],R.T@pva[2],R.T@delta*2.4/np.linalg.norm(delta),R.T@[0,0,-1]]
  fields.append(dict(frame=fi,source_stamp=stamp,errors=np.abs(state-exact).tolist(),state12=state.tolist(),world_source_acceleration=pva[2].tolist(),acceleration_norm=float(np.linalg.norm(pva[2])),digest=digest))
  staged['neural_input_fidelity_frames']+=1;staged['all_six_generated_frames']+=1;staged['generated_candidates']+=6
  # Source-time direction variation without any GT selector.
  dirs=[]
  from query_minco.inference.selector_safety import world_coefficients
  for qi in range(6):
   w=Program(world_coefficients(z['coefficients'][qi],z['position_W'],R),z['durations'][qi]);d=w.pva(.75)[0]-w.pva(.7)[0];dirs.append(d/np.linalg.norm(d))
  angles=[np.degrees(np.arccos(np.clip(dirs[i]@dirs[j],-1,1))) for i in range(6) for j in range(i)]
  directions.append(dict(frame=fi,retained_direction_W=np.array(dirs).tolist(),pair_angle_deg_min_median_max=np.percentile(angles,[0,50,100]).tolist()))
  for q in b['queries']:
   outcomes[q.get('reason','PASS')]+=1
   if not q['passed']:continue
   staged['full_56phase_sweep_and_ready_frames']+=1
   ack=q['cpp_receipt'];audit=ack['input_audit']
   assert max(map(float,audit['absolute_field_errors12']))<=2e-6 and audit['depth_mask_sha256']==inputrow['input_depth_mask_sha256']
   receipts.append(dict(frame=fi,audit=audit,cpp_wall_ms=float(ack['wall_ms']),source_to_ready_ms=q['source_to_complete_ms']))
   staged['cpp_complete_input_identity_geometry_version_pass']+=1
   with (RUN/('c1/banks_%03d.pkl'%fi)).open('rb') as file:payload=pickle.load(file)[q['query']]
   bank=payload['bank'];envelope=payload['envelope'];assert bank.digest==hashlib.sha256(bank.body).hexdigest()==envelope['body_sha256']
   prefix,stops=decode_body(bank.body);assert len(stops)==56;prefix.validate(MotionLimits())
   for phase,stop in stops:
    full=stop if phase==0 else concatenate(trim(prefix,0,phase),stop);full.validate(MotionLimits())
    assert stop.extrema()[2]<=.35+1e-8
    proof=volume.certify(full.coefficients,full.durations,bank.prepared_ros,required_until=bank.valid_until_ros)
    assert proof['status']=='CERTIFIED_FREE'
    np.testing.assert_allclose(stop.pva(0),prefix.pva(phase),atol=1e-8)
    np.testing.assert_allclose(stop.pva(stop.duration)[1:],0,atol=1e-8)
    assert bank.valid_until_ros<references[0].valid_until_ros and bank.valid_until_mono<references[0].valid_until_mono
    stopchecks+=1
   seq=envelope['sequence'];gate=next(g for g in gates if g['type']=='ACTUAL_READONLY_GATE' and int(g['sequence'])==seq)
   gate=dict(gate,frame=fi,source_stamp=stamp,source_to_actual_gate_ms=(float(gate['actual_ros'])-stamp)*1000,planned_activation_ros=b['activation_ros'],lateness_ms=(float(gate['actual_ros'])-b['activation_ros'])*1000)
   actual.append(gate)
   if yes(gate['passed']):
    staged['fully_qualified_readonly_frames']+=1
    qualified.append(dict(frame=fi,source_stamp=stamp,source_to_ready_ms=q['source_to_complete_ms'],source_to_gate_ms=gate['source_to_actual_gate_ms']))
   connection=trim(prefix,0,1);retained=Program(z['q5_retained_coefficients'],z['q5_retained_durations']);brake=stops[-1][1]
   components={name:contribution(p) for name,p in [('connection',connection),('MINCO_retained',retained),('endpoint_stop',brake)]}
   components['endpoint_stop']['arc_upper_bound_m']=brake.extrema()[0]*brake.duration
   parts.append(dict(frame=fi,parts=components,query_executed=False))
 # All frame denominators, including fault-braking observations.
 total=len(frames);before=[b for b in frames if b['source_stamp']<failure];after=[b for b in frames if b['source_stamp']>=failure]
 queryCam=[r for r in camera if 'query_index' in r];nativeCam=[r for r in camera if r.get('type')=='NATIVE_INPUT']
 def cadence(rows):
  stamps=np.array([r['stamp_ns']/1e9 for r in rows]);return dict(count=len(stamps),mean_source_hz=(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else None,source_gap_p50_p95_max_ms=quant(np.diff(stamps)*1000),first_source=float(stamps[0]) if len(stamps) else None,last_source=float(stamps[-1]) if len(stamps) else None)
 normalstamps=[b['source_stamp'] for b in before];allstamps=[b['source_stamp'] for b in frames];qualstamps=[q['source_stamp'] for q in qualified]
 rejects=[r for r in ledger if r['kind']=='REJECT'];joint=[r for r in rejects if r['reason']=='JOINT_STATE_BOUND']
 qidcounts=collections.Counter(r['reason'] for r in rejects)
 cadence_result=dict(target_query_hz=30,native=cadence(nativeCam),query=cadence(queryCam),synced=node['input_counters'],queued_replaced=node['input_counters'].get('queue_replaced',0),
 processed_rejects=dict(qidcounts),joint_reason_overlapping=dict(speed=sum(r.get('speed_error',0)>.03 for r in joint),acceleration=sum(r.get('acceleration_norm',0)>.05 for r in joint),tilt=sum(r.get('tilt_deg',0)>.5 for r in joint)),
 unledgered_processed_inputs=node['input_counters']['processed']-len(rejects)-len(network_inputs),
 unledgered_reason='One RPC connection refused after reference process exited, raw Python traceback retained' ,
 effective_network_frames=total,normal_pre_fault_frames=len(before),post_fault_frames=len(after),normal_frame_max_gap_ms=max(np.diff(normalstamps))*1000,all_network_max_gap_ms=max(np.diff(allstamps))*1000,qualified_max_gap_ms=max(np.diff(qualstamps))*1000,
 camera_receiver_age_ms=quant([r['input_age_s']*1000 for r in queryCam]),source_to_bridge_emit_ms=quant([(r['emit_mono_ns']-r['received_mono_ns'])/1e6 for r in queryCam]),
 neural_queue_wait_ms=quant([(r['network_start_mono']-r['received_mono'])*1000 for r in network_inputs]))
 (OUT/'CAMERA_CADENCE_RESULTS.json').write_text(json.dumps(cadence_result,indent=2,cls=NumpyEncoder))
 (OUT/'NEURAL_INPUT_FIELD_AUDIT.json').write_text(json.dumps(dict(passed=True,checked_frames=total,nonzero_acceleration_frames=sum(r['acceleration_norm']>1e-12 for r in fields),per_field_max_float32_error=np.max([r['errors'] for r in fields],axis=0).tolist(),cpp_bound_frames=len(receipts),field_order=['v_B0x','v_B0y','v_B0z','a_B0x','a_B0y','a_B0z','goal_B0x','goal_B0y','goal_B0z','gravity_B0x','gravity_B0y','gravity_B0z'],frames=fields,cpp_receipts=receipts),indent=2,cls=NumpyEncoder))
 result=dict(status='FAIL_BLOCKED_NATIVE_TRACKING_INSUFFICIENT_OPPORTUNITIES',stages=dict(staged),all_neural_frames=total,normal_pre_fault_opportunities=len(before),post_fault_frames=len(after),
 normal_ready_rate=len(receipts)/len(before),normal_final_gate_rate=len(qualified)/len(before),overall_ready_rate=len(receipts)/total,overall_qualified_rate=len(qualified)/total,
 ready_deadline_timeout_count=sum('DEADLINE' in q.get('reason','') for b in frames for q in b['queries']),
 NONE_frames=sum(b['selection']=='NONE' for b in frames),actual_gate_refused_frames=sum(not yes(g['passed']) for g in actual),
 qualified_latency_sample_count=len(qualified),qualified_source_to_ready_p50_p95_max_ms=quant([q['source_to_ready_ms'] for q in qualified]),qualified_source_to_actual_gate_p50_p95_max_ms=quant([q['source_to_gate_ms'] for q in qualified]),
 all_ready_source_to_ready_p50_p95_max_ms=quant([r['source_to_ready_ms'] for r in receipts]),opportunity_min20_pass=len(before)>=20,
 raw_query_source_gap_pass=cadence_result['query']['source_gap_p50_p95_max_ms'][-1]<=100,normal_source_gap_pass=cadence_result['normal_frame_max_gap_ms']<=100,
 input_fidelity_pass=True,native_tracking_pass=native['necessary_control_gates_pass'],full_stop_rows_rechecked=stopchecks,actual_gates=actual,qualified_proposals=qualified,outcomes=dict(outcomes),Query_control=False,first3m_takeover='BLOCKED')
 (OUT/'E_REALTIME_RESULTS.json').write_text(json.dumps(result,indent=2,cls=NumpyEncoder));(OUT/'NETWORK_CONTRIBUTION.json').write_text(json.dumps(dict(proposals=parts,six_query_directions=directions,actual_query_executions=0),indent=2,cls=NumpyEncoder))
 with (OUT/'NEURAL_INPUT_FIELDS.csv').open('x',newline='') as f:
  names=['frame','source_stamp']+['field'+str(i) for i in range(12)]+['error'+str(i) for i in range(12)]
  w=csv.DictWriter(f,fieldnames=names);w.writeheader()
  for r in fields:w.writerow(dict(frame=r['frame'],source_stamp=r['source_stamp'],**{'field'+str(i):v for i,v in enumerate(r['state12'])},**{'error'+str(i):v for i,v in enumerate(r['errors'])}))
 # Failures kept, all phase/control data unchanged.
 keys=[k+str(i) for k in ('p','v','a') for i in range(3)];actualPVA=np.array([fail[k] for k in keys]).reshape(3,3);expected=np.array([fail['r'+k] for k in keys]).reshape(3,3)
 dt=fail['x'];prediction=actualPVA.copy();prediction[0]+=actualPVA[1]*dt+.5*actualPVA[2]*dt*dt;prediction[1]+=actualPVA[2]*dt
 near=odom[(odom['stamp']>=failure-.06)&(odom['stamp']<=failure+.06)]
 failure_result=dict(source='NEW_RUN_NOT_OLD007',first_event42_ros=failure,event_predicted_error=np.linalg.norm(prediction-expected,axis=1).tolist(),event_actual_pva=actualPVA.tolist(),event_reference=expected.tolist(),
 odom_near=[{k:float(r[k]) for k in ('stamp','recv','v0','a0')} for r in near],
 command_gap_p50_p95_max_ms=native['command_gap_p50_p95_max_ms'],wake_late_ms=native['wake_delay_p50_p95_max_ms'],PVA_max_errors=native['all_phase_tracking_error_norm_p50_p95_max'][2],
 confirmed='With own worker queue, max wake1.088ms and publish gap21.007ms, yet acceleration tracking still exceeds0.1. This rules out asserting the old13.76ms wake event as necessary sole cause. Fixed-step/wall-stamp and physical Controller response still not separable without model acceleration/step count.',
 model_acceleration_recorded=False,unique_physical_root_cause_proven=False,backup=native['backups'],dwells=native['stop_dwells'])
 (OUT/'CONTROL_FAILURE_CAUSAL_REVIEW.json').write_text(json.dumps(failure_result,indent=2,cls=NumpyEncoder))
 print(json.dumps(dict(summary={k:v for k,v in result.items() if k not in ('actual_gates','qualified_proposals','outcomes')},cadence=cadence_result,fieldmax=np.max([r['errors'] for r in fields],axis=0).tolist()),indent=2,cls=NumpyEncoder))
if __name__=='__main__':main()
