"""Independent raw checks for execution006, no ROS/physical permissions."""
import collections,csv,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,trim,concatenate
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope,ErrorBudget
from query_minco.inference.candidate_feasibility import contribution
from query_minco.inference.shadow import rotation
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution006';RUN=OUT/'shadow_inputs001'
def quant(a):return np.percentile(a,[50,95,100]).tolist() if len(a) else None
def thaw(s):return Program(np.frombuffer(s.coefficients,dtype='<f8').reshape(-1,3,6),s.durations)
def main():
 for p,h in json.loads((OUT/'REAL_SHADOW_BINDING.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 with (OUT/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:bundles=pickle.load(f)
 cmd=np.genfromtxt(RUN/'COMMAND.csv',delimiter=',',names=True);odom=np.genfromtxt(RUN/'ODOM.csv',delimiter=',',names=True);events=np.genfromtxt(RUN/'EVENT.csv',delimiter=',',names=True)
 keys=[k+str(i) for k in ('p','v','a') for i in range(3)]
 cv=np.stack([cmd[k] for k in keys],1).reshape(-1,3,3);ov=np.stack([odom[k] for k in keys],1).reshape(-1,3,3)
 activ=events[events['code']==11];error=[];command_res=[];windows=[];cur=[]
 def reference(t):
  eligible=activ[activ['y']<=t]
  if not len(eligible):return bundles[0].prefix.pva(0)
  a=eligible[-1];p=bundles[int(a['bundle'])].prefix
  return p.pva(max(0,min(t-a['y'],p.duration)))
 for c,v in zip(cmd,cv):
  a=activ[activ['bundle']==c['bundle']][-1];p=bundles[int(c['bundle'])].prefix
  command_res.append(np.linalg.norm(v-p.pva(max(0,min(c['mono']-a['z'],p.duration))),axis=1))
 for o,v in zip(odom,ov):
  err=np.linalg.norm(v-reference(o['stamp']),axis=1);error.append(err)
  R=rotation([o[x] for x in ('qx','qy','qz','qw')]);goal=R.T@(np.array([3,0,2])-v[0])
  ok=np.linalg.norm(v[1]-[.8,0,0])<=.03 and np.linalg.norm(v[2])<=.05 and np.arccos(np.clip(R[2,2],-1,1))<=np.deg2rad(.5) and goal[0]>0 and np.max(np.abs(goal[1:]/goal[0]))<=.175 and err[0]<=.02 and np.linalg.norm(goal)>=2.4
  if ok and (not cur or o['stamp']-cur[-1]['stamp']<=.04):cur.append(o)
  else:
   if cur:windows.append(cur)
   cur=[o] if ok else []
 if cur:windows.append(cur)
 error=np.array(error);res=json.loads((RUN/'ARBITER_RESULT.json').read_text())
 scene=json.loads((RUN/'SCENE_READY.json').read_text());resources=[json.loads(l) for l in (RUN/'RESOURCE_TRACE.jsonl').read_text().splitlines()]
 he=[json.loads(l) for l in (RUN/'HOST_EVENTS.jsonl').read_text().splitlines()];camera=[json.loads(l) for l in (RUN/'CAMERA_TRACE.jsonl').read_text().splitlines()]
 cleanup=json.loads((RUN/'CLEANUP.json').read_text())['remaining']
 failcodes=events[np.isin(events['code'],[34,40,41,42,43,44,45])]
 native=dict(status='PENDING',scope='THIS_RUN_NOMINAL_REFERENCE_SHADOW_NOT_NEW_WATCHDOG_TRIAL',command_count=len(cmd),command_gap_p50_p95_max_ms=quant(np.diff(cmd['begin'])*1000),command_reference_max_error=np.max(command_res,axis=0).tolist(),tracking_error_norm_p50_p95_max=np.percentile(error,[50,95,100],axis=0).tolist(),measured_speed_max=float(np.linalg.norm(ov[:,1],axis=1).max()),measured_acceleration_max=float(np.linalg.norm(ov[:,2],axis=1).max()),measured_min_z=float(ov[:,0,2].min()),actual_motion_sampled_not_continuous_proof=True,activation_lateness_max_ms=float(events[events['code']==20]['y'].max()*1000),moving_window_s=max(w[-1]['stamp']-w[0]['stamp'] for w in windows),unique_publisher_at_scene_gate=dict(scene['publishers']).get('/so3_control/pos_cmd')==['/f3_reference_arbiter'],continuous_topology_raw_history_recorded=False,fail_event_count=len(failcodes),camera_exit=[x['type'] for x in camera if 'type' in x][-4:],camera_fatal=any(x.get('type')=='FATAL' for x in camera),cleanup_remaining=cleanup,resources=dict(WSL_min_GiB=min(r['MemAvailable_kb'] for r in resources)/1048576,Windows_min_GiB=min(r['available_kb'] for r in he if r['type']=='WINDOWS_AVAILABLE')/1048576,GPU_peak_MiB=max(r['gpu']['used_mib'] for r in resources),swap_peak_kb=max(r['SwapUsed_kb'] for r in resources)),Query_control=False)
 native['passed']=bool(not res['qualification_failed'] and not res['buffer_overflow'] and len(cmd)==res['commands'] and max(np.diff(cmd['begin']))<=.04 and np.all(error<=[.02,.05,.1]) and native['measured_speed_max']<=1 and native['measured_acceleration_max']<=1.5 and native['measured_min_z']>1.9 and native['activation_lateness_max_ms']<=10 and native['moving_window_s']>=.5 and native['unique_publisher_at_scene_gate'] and not len(failcodes) and not native['camera_fatal'] and not cleanup)
 native['status']='PASS_NOMINAL_REFERENCE_SCOPE' if native['passed'] else 'FAIL'
 (OUT/'NATIVE_MEASURED_RESULTS.json').write_text(json.dumps(native,indent=2))
 batches=[json.loads(l) for l in (RUN/'c1/events.jsonl').read_text().splitlines()];result=json.loads((RUN/'c1/RESULT.json').read_text())
 proof=json.loads(bundles[0].prefix_proof);volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],proof['valid_until'],'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR,ErrorBudget(**proof['error_budget']))
 complete=[];gates=result['activation_gates'];allq=[q for b in batches for q in b['queries']]
 for batch in batches:
  for q in batch['queries']:
   if not q['passed']:continue
   with (RUN/('c1/banks_%03d.pkl'%batch['frame'])).open('rb') as f:bank=pickle.load(f)[q['query']]
   prefix=thaw(bank.prefix);z=np.load(RUN/('c1/frame_%03d.npz'%batch['frame']))
   retained=Program(z['q%d_retained_coefficients'%q['query']],z['q%d_retained_durations'%q['query']]);conn=trim(prefix,0,q['connection_s'])
   assert q['ready_ros']<=batch['activation_ros']-.02
   assert q['ready_ros']-batch['source_stamp']<=.25
   assert 0<=q['ready_ros']-q['measurement_stamp'] # saved measurement precedes readiness
   a=activ[activ['bundle']==1][-1];expected=bundles[1].prefix.pva(batch['activation_ros']-a['y'])
   np.testing.assert_allclose(prefix.pva(0),expected,atol=5e-7)
   extrema=[];proofslack=[];headgap=[]
   for stop in bank.stops:
    sp=thaw(stop.polynomial);full=sp if stop.phase==0 else concatenate(trim(prefix,0,stop.phase),sp)
    full.validate(MotionLimits());sp.validate(MotionLimits());ex=sp.extrema();assert ex[2]<=.35+1e-8
    gp=volume.certify(full.coefficients,full.durations,bank.prepared_ros,required_until=bank.valid_until_ros)
    assert gp['status']=='CERTIFIED_FREE'
    saved=json.loads(stop.proof);assert saved['trajectory_sha256']==sp.digest and saved['certified_prefix_sha256']==prefix.digest and saved['status']=='CERTIFIED_FREE'
    assert bank.valid_until_ros<proof['valid_until'] and stop.phase+sp.duration+.06<bank.valid_until_ros-bank.prepared_ros
    np.testing.assert_allclose(sp.pva(0),prefix.pva(stop.phase),atol=1e-9)
    np.testing.assert_allclose(sp.pva(sp.duration)[1:],0,atol=1e-8)
    extrema.append(ex);proofslack.append(gp['minimum_halfspace_slack_m']);headgap.append(float(np.max(np.abs(sp.pva(0)-prefix.pva(stop.phase)))))
   actual=next(g for g in gates if g['frame']==batch['frame'] and g['query']==q['query'])
   assert actual['measurement_stamp']<=actual['actual'] and actual['measurement_received_mono']<=actual['actual_mono'] and actual['passed']
   brake=thaw(bank.stops[-1].polynomial)
   parts={k:contribution(p) for k,p in [('connection',conn),('original_MINCO_retained',retained),('endpoint_brake',brake)]}
   total=sum(p['arc_length_estimate_m'] for p in parts.values());duration=sum(p['duration_s'] for p in parts.values())
   for v in parts.values():v.update(arc_length_fraction=v['arc_length_estimate_m']/total,time_fraction=v['duration_s']/duration)
   complete.append(dict(frame=batch['frame'],query=q['query'],full_bank_rows=len(bank.stops),source_to_complete_ms=q['source_to_complete_ms'],source_to_actual_gate_ms=(actual['actual']-batch['source_stamp'])*1000,actual_gate=actual,full_prefix_extrema=prefix.extrema(),bank_stop_max_extrema=np.max(extrema,axis=0).tolist(),all_phase_swept_min_slack_m=min(proofslack),max_backup_head_pva_gap=max(headgap),prefix_terminal_pva=prefix.pva(prefix.duration).tolist(),endpoint_brake_terminal_pva=brake.pva(brake.duration).tolist(),original_cell_expiry=proof['valid_until'],proposal_expiry=bank.valid_until_ros,parts=parts,scope='EXTERNAL_SUPERVISOR_CONDITIONAL_TRACKING_BUDGET; C1 command not executed'))
 stats={k:quant([b[k] for b in batches]) for k in ('network_ms','decode_export_ms','six_generation_ms','six_policy_processing_ms','post_generation_ms','source_to_decision_ms','async_evidence_IO_ms')}
 stats['complete_bank_ms']=quant([q['complete_bank_ms'] for q in allq if 'complete_bank_ms' in q])
 stats['source_to_full_numerical_proof_ms_including_deadline_failures']=quant([q['source_to_complete_ms'] for q in allq if 'source_to_complete_ms' in q])
 summary=dict(status='NONZERO_READONLY_PROPOSAL_ACHIEVED_REALTIME_RELIABILITY_BLOCKED',frames=len(batches),generated_candidates=6*len(batches),complete_before_deadline=len(complete),causal_actual_activation_gate_pass=sum(g['passed'] for g in gates),qualified_source_to_proposal_p50_p95_max_ms=quant([x['source_to_complete_ms'] for x in complete]),qualified_latency_sample_count=len(complete),all_frame_timing_p50_p95_max_ms=stats,candidate_outcomes=dict(collections.Counter(q.get('reason','COMPLETE_PASS') for q in allq)),joint_input_rejects=result['joint_input_rejects'],complete_proposals=complete,selection_NONE_frames=sum(b['selection']=='NONE' for b in batches),native_reference_scope_pass=native['passed'],Query_control=False,online_depth_body_certification='UNKNOWN; external independent empty-room evidence only',module_split_limit='Actual run recorded network, decode, connection and combined complete-bank wall time. Brake construction, sweep, serialization and CPU-vs-scheduler time NOT separately instrumented; do not infer per-module native timings from CPU profile.',whole_six_complete_certification_timing=None,whole_six_note='All six generated, deterministic first complete-candidate policy only. Other45 were explicitly not certified; no worst-case six-complete latency claim.',new_control_qualification=False,first_3m_Query_takeover='BLOCKED')
 (OUT/'E_REALTIME_RESULTS.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(dict(native=native,summary={k:v for k,v in summary.items() if k not in ('complete_proposals','joint_input_rejects')}),indent=2))
if __name__=='__main__':main()
