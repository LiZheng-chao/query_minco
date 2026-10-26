"""Readonly007/008 causal comparison; derived windows are NOT acceptance data."""
import collections,csv,hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution009'
def quant(v):return np.percentile(v,[50,95,100]).tolist() if len(v) else None
def js(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=js))
def writecsv(path,rows):
 with path.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
 comparisons=[]
 for label in ('f3_execution007','f3_execution008'):
  base=OUT.parent/label;run=base/'shadow_inputs001'
  cmd=np.genfromtxt(run/'COMMAND.csv',delimiter=',',names=True);odom=np.genfromtxt(run/'ODOM.csv',delimiter=',',names=True);events=np.genfromtxt(run/'EVENT.csv',delimiter=',',names=True)
  with (base/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:bundles=pickle.load(f)
  activation=events[events['code']==11];selected=events[events['code']==31];failure=events[events['code']==42][0];tf=float(failure['ros'])
  def reference(t):
   aa=activation[activation['y']<=t]
   if not len(aa):return bundles[0].prefix.pva(0)
   a=aa[-1];b=bundles[int(a['bundle'])];elapsed=t-a['y']
   stops=selected[(selected['bundle']==a['bundle'])&(selected['x']<=elapsed)]
   if len(stops):
    phase=stops[0]['x'];stop=next(s for s in b.stops if abs(s.phase-phase)<1e-8)
    return stop.polynomial.pva(max(0,min(elapsed-phase,stop.polynomial.duration)))
   return b.prefix.pva(max(0,min(elapsed,b.prefix.duration)))
  fields=[k+str(i) for k in ('p','v','a') for i in range(3)]
  state=np.stack([odom[k] for k in fields],1).reshape(-1,3,3);desired=np.array([reference(t) for t in odom['stamp']])
  errors=np.linalg.norm(state-desired,axis=2)
  errA=errors[:,2];bad=(errA>.1);intervals=[];start=None
  for i,flag in enumerate(bad):
   if flag and start is None:start=i
   if start is not None and (not flag or i==len(bad)-1):
    end=i if flag else i-1;intervals.append(dict(first_row=start,last_row=end,count=end-start+1,first_stamp=float(odom['stamp'][start]),last_stamp=float(odom['stamp'][end]),span_s=float(odom['stamp'][end]-odom['stamp'][start]),peak_error=float(errA[start:end+1].max())));start=None
  for name,data,timefield in [('COMMAND',cmd,'begin'),('ODOM',odom,'stamp'),('EVENT',events,'ros')]:
   subset=data[(data[timefield]>=tf-.6)&(data[timefield]<=tf+.6)]
   rows=[]
   for r in subset:
    row={k:float(r[k]) for k in data.dtype.names};row['relative_to_first_failure_s']=float(r[timefield]-tf)
    if name=='COMMAND':
     ix=int(np.where(cmd['begin']==r['begin'])[0][0]);row['previous_publish_gap_ms']=float((cmd['begin'][ix]-cmd['begin'][ix-1])*1000) if ix else 0
     row['wake_late_ms']=float((r['mono']-r['wake'])*1000);row['publish_wall_ms']=float((r['end']-r['begin'])*1000);row['tick_thread_cpu_ms']=float((r['cpu1']-r['cpu0'])*1000)
    if name=='ODOM':
     ix=int(np.where(odom['stamp']==r['stamp'])[0][0]);row['source_dt_s']=float(r['stamp']-odom['stamp'][ix-1]) if ix else 0;row['callback_wall_ms']=float((r['done']-r['recv'])*1000)
     row['continuous_reference_acceleration_x']=float(desired[ix,2,0]);row['original_acceleration_error_norm']=float(errA[ix])
    rows.append(row)
   writecsv(OUT/'analysis'/(label+'_'+name+'_NEAR_FAILURE.csv'),rows)
  windows=[]
  for ix,t in enumerate(odom['stamp']):
   if not tf-.3<=t<=tf+.3:continue
   for k in (1,2,3,5,10,20):
    if ix<k:continue
    j=ix-k;dt=t-odom['stamp'][j];v=(state[ix,1]-state[j,1])/dt
    # Compare trailing mean with trailing reference mean, not an instantaneous reference.
    meanref=(desired[ix,1]-desired[j,1])/dt
    windows.append(dict(mode='DERIVED_TRAILING_WINDOW_NOT_ACCEPTANCE',end_row=ix,start_row=j,end_stamp=float(t),relative_s=float(t-tf),samples=k,actual_window_s=float(dt),a_x=float(v[0]),a_y=float(v[1]),a_z=float(v[2]),reference_mean_a_x=float(meanref[0]),mean_error_norm=float(np.linalg.norm(v-meanref)),original_sample_a_x=float(state[ix,2,0]),original_error_norm=float(errA[ix])))
  writecsv(OUT/'analysis'/(label+'_TRAILING_WINDOWS.csv'),windows)
  # First failing snapshot maps to the last source Odom stamp from recorded age.
  eventSource=tf-float(failure['x']);ix=int(np.argmin(abs(odom['stamp']-eventSource)))
  at_event=[r for r in windows if r['end_row']==ix]
  signals=[str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and ('so3' in p.name.lower() or 'rpm' in p.name.lower())]
  sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
   str(ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution004/runtime/quadrotor_simulator_startup_candidate.cpp'),
   '/home/lizheng/YOPO/Controller/src/so3_control/src/NetworkControl.cpp','/home/lizheng/YOPO/Controller/src/so3_control/src/SO3Control.cpp',
   '/home/lizheng/YOPO/Controller/src/so3_quadrotor_simulator/src/dynamics/Quadrotor.cpp']}
  comparison=dict(run=label,first_failure_ros=tf,original_max_acceleration_error=float(errA.max()),original_A_exceed_samples=int(bad.sum()),original_A_exceed_before_first_failure=int((bad&(odom['stamp']<tf)).sum()),original_A_exceed_at_or_after_first_failure=int((bad&(odom['stamp']>=tf)).sum()),original_exceed_intervals=intervals,
   snapshot_source_stamp=float(odom['stamp'][ix]),snapshot_previous_dt_s=float(odom['stamp'][ix]-odom['stamp'][ix-1]),
   snapshot_velocity_difference=(state[ix,1]-state[ix-1,1]).tolist(),snapshot_original_acceleration=state[ix,2].tolist(),snapshot_reference_acceleration=desired[ix,2].tolist(),
   command_gap_p50_p95_max_ms=quant(np.diff(cmd['begin'])*1000),wake_delay_p50_p95_max_ms=quant((cmd['mono']-cmd['wake'])*1000),
   callback_wall_p50_p95_max_ms=quant((odom['done']-odom['recv'])*1000),trailing_windows_at_first_snapshot=at_event,
   SO3_numeric_assets=signals,SO3_numeric_available=False,SO3_note='Topic connectivity and source code are available, not actual time-series force/RPM. Cannot reconstruct missing Controller receive and SO3 output.',
   integration_steps_available=False,RHS_available=False,sources_sha256=sources)
  gapix=int(np.diff(cmd['begin']).argmax())+1;nearest=int(abs(cmd['begin']-comparison['snapshot_source_stamp']).argmin())
  comparison.update(max_gap_interval_ros=[float(cmd['begin'][gapix-1]),float(cmd['begin'][gapix])],max_gap_end_minus_anomaly_source_ms=float((cmd['begin'][gapix]-comparison['snapshot_source_stamp'])*1000),nearest_command_to_anomaly_source={k:float(cmd[nearest][k]) for k in ('begin','end','wake','mono','cpu0','cpu1')})
  comparisons.append(comparison)
 # E4 real latency/backpressure audit, preserves exact original denominators.
 run=OUT.parent/'f3_execution008/shadow_inputs001';frames=[json.loads(s) for s in (run/'c1/events.jsonl').read_text().splitlines()]
 ledger=[json.loads(s) for s in (run/'c1/INPUT_LEDGER.jsonl').read_text().splitlines()];network=[r for r in ledger if r['kind']=='NETWORK_INPUT']
 latency=[]
 for b in frames:
  l=next(r for r in network if r['frame']==b['frame']);q=next((q for q in b['queries'] if q['passed']),None)
  r=dict(frame=b['frame'],source_stamp=b['source_stamp'],source_age_at_processing_ms=(l['processing_ros']-b['source_stamp'])*1000,receive_to_take_ms=(l['processing_start_mono']-l['received_mono'])*1000,take_to_network_ms=(l['network_start_mono']-l['processing_start_mono'])*1000,network_wall_ms=b['network_ms'],network_cpu_ms=b['network_thread_cpu_ms'],decode_wall_ms=b['decode_export_ms'],decode_cpu_ms=b['decode_thread_cpu_ms'],decision_source_ms=b['source_to_decision_ms'],ready_source_ms=q['source_to_complete_ms'] if q else None,post_generation_ms=b['post_generation_ms'],async_IO_ms=b['async_evidence_IO_ms'],selection=b['selection'],after_fault=b['source_stamp']>=comparisons[1]['first_failure_ros'])
  latency.append(r)
 writecsv(OUT/'analysis/LATENCY_FRAME_DECOMPOSITION.csv',latency)
 latency_summary={k:quant([r[k] for r in latency if r[k] is not None]) for k in latency[0] if k not in ('frame','source_stamp','selection','after_fault')}
 save('CAUSAL_COMPARISON.json',dict(mode='READONLY_REANALYSIS_NO_NEW_ROS',runs=comparisons,
  controller_mass_audit=dict(NetworkControl_h_mass=.98,NetworkControl_constructor_calls_setMass=True,plant_mass=.98,mass_mismatch_evidence=False),
  warning_post_fault='008 has1 pre-fault and20 post-fault exceedances; never report all21 as normal cruise. Threshold and original failure unchanged.',
  confirmed=['Original velocity/stamp finite-difference acceptance exceeded0.1 in both runs; points and failures remain unchanged.',
   '007 has an abrupt source-time/velocity-difference anomaly near33.65ms publisher gap;008 includes smooth multi-sample positive acceleration while command acceleration is0 and max publisher gap21.01ms.',
   'Original EMPTY/nonREADY status follows P/V feedback SO3Control branch, not simply a_ref as physical acceleration. Source has kx/kv contributions, not proof of actual numeric SO3.',
   'Fixed1ms integration and wall-stamped Odom are source facts. Runtime step counts/RHS/SO3/motor series are missing.'],
  not_proven=['Actual plant RHS acceleration tracking error above0.1, as distinct from Odom metrology error','Single unique cause or GIL causality','Absolute physical acceleration limit1.5 exceeded; old Odom-derived observed values remain below it'],
  windows_warning='Trailing estimates are diagnostic-only with original rows preserved. They do not replace acceptance measurements or change thresholds.'))
 save('LATENCY_BACKPRESSURE_AUDIT.json',dict(mode='HISTORICAL_ONLY',frames=latency,quantiles=latency_summary,original_processed=471,terminal_REJECT_rows=452,NETWORK_INPUT_rows=18,missing_processed_terminal_information=1,queue_replaced=2,post_fault_neural_frames=5,source_expired_successes=0,proposals_reused=False,interpretation='Ready latency growth must be decomposed from actual network/decode/queue stages. Raw protocol lacked one terminal on RPC closure and a fault latch before new inference. No new native latency or20-opportunity success claimed.'))
 print(json.dumps(dict(runs=[{k:r[k] for k in ('run','original_max_acceleration_error','original_A_exceed_samples','original_exceed_intervals')} for r in comparisons],latency_quantiles=latency_summary),indent=2,default=js))
if __name__=='__main__':main()
