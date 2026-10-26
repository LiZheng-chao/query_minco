"""Independent raw CSV acceptance, reconstructs exact transported reference."""
import json,pickle,hashlib,csv
from pathlib import Path
import numpy as np
from query_minco.inference.shadow import rotation
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution005';RUN=OUT/'native_ref001'
def read(n):return np.genfromtxt(RUN/n,delimiter=',',names=True)
def quant(a):return np.quantile(a,[.5,.95,1]).tolist() if len(a) else None
def main():
 for p,h in json.loads((OUT/'NATIVE_BINDING_live.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 with (OUT/'preparation_live/TRANSPORT_BUNDLES.pkl').open('rb') as f:bundles=pickle.load(f)
 cmd=read('COMMAND.csv');odom=read('ODOM.csv');events=read('EVENT.csv')
 ids=['p0','p1','p2','v0','v1','v2','a0','a1','a2']
 cv=np.stack([cmd[k] for k in ids],axis=1).reshape(-1,3,3);ov=np.stack([odom[k] for k in ids],axis=1).reshape(-1,3,3)
 activations=events[events['code']==11];physical=events[events['code']==20];selected=events[events['code']==31];backup_actual=events[events['code']==33];gates=events[events['code']==32]
 def planbid(bid):return activations[activations['bundle']==bid][-1]
 def poly_at(bid,elapsed):
  ss=selected[selected['bundle']==bid]
  row=next((s for s in ss if elapsed>=s['x']),None)
  if row is not None:
   entry=next(x for x in bundles[int(bid)].stops if abs(x.phase-row['x'])<1e-8)
   return entry.polynomial.pva(min(elapsed-row['x'],entry.polynomial.duration))
  return bundles[int(bid)].prefix.pva(max(0,min(elapsed,bundles[int(bid)].prefix.duration)))
 # Every actual transmitted PVA must match its immutable reference, evaluated at exact tick mono.
 residual=[]
 for c,pva in zip(cmd,cv):
  a=planbid(c['bundle']);elapsed=c['mono']-a['z']
  residual.append(np.linalg.norm(pva-poly_at(c['bundle'],elapsed),axis=1))
 residual=np.array(residual);assert residual.max()<1e-7
 error=[];per=[];handover=[]
 for o,pva in zip(odom,ov):
  t=o['stamp'];eligible=activations[activations['y']<=t]
  if len(eligible):a=eligible[-1];bid=int(a['bundle']);expected=poly_at(bid,t-a['y'])
  else:bid=0;expected=bundles[0].prefix.pva(0)
  err=np.linalg.norm(pva-expected,axis=1);error.append(err);per.append(bid)
  if bid==1:
   q=np.array([o[x] for x in ['qx','qy','qz','qw']]);R=rotation(q);tilt=np.arccos(np.clip(R[2,2],-1,1));goal=R.T@(np.array([3,0,2])-pva[0])
   phase=t-a['y'];ok=(6.9<=phase<=7.96 and np.linalg.norm(pva[1]-[.8,0,0])<=.03 and np.linalg.norm(pva[2])<=.05 and tilt<=np.deg2rad(.5) and goal[0]>0 and max(abs(goal[1:]/goal[0]))<=.175 and err[0]<=.02 and np.linalg.norm(goal)>=2.4)
   handover.append(dict(stamp=t,ok=bool(ok),pva=pva.tolist(),tilt_deg=float(np.rad2deg(tilt)),position_error=float(err[0]),phase=phase))
 error=np.array(error);per=np.array(per)
 windows=[];cur=[]
 for h in handover:
  if h['ok'] and (not cur or h['stamp']-cur[-1]['stamp']<=.04):cur.append(h)
  else:
   if cur:windows.append(cur)
   cur=[h] if h['ok'] else []
 if cur:windows.append(cur)
 window=max(windows,key=lambda w:w[-1]['stamp']-w[0]['stamp']) if windows else []
 crossing=min(handover,key=lambda x:abs(x['pva'][0][0]))
 dwells=[]
 for e in events[events['code']==53]:
  rr=odom[odom['stamp']<=e['ros']]
  all_a=np.stack([rr[k] for k in ids],axis=1).reshape(-1,3,3)
  good=(np.linalg.norm(all_a[:,1],axis=1)<=.03)&(np.linalg.norm(all_a[:,2],axis=1)<=.05)
  j=len(rr)-1
  while j>0 and good[j-1] and rr['stamp'][j]-rr['stamp'][j-1]<=.04:j-=1
  rr=rr[j:];a=all_a[j:]
  passed=len(rr)>1 and rr['stamp'][-1]-rr['stamp'][0]>=.5 and np.diff(rr['stamp']).max()<=.04 and np.linalg.norm(a[:,1],axis=1).max()<=.03 and np.linalg.norm(a[:,2],axis=1).max()<=.05
  dwells.append(dict(stage=int(e['x']),passed=bool(passed),duration=float(rr['stamp'][-1]-rr['stamp'][0]),last_pva=a[-1].tolist()))
 host=json.loads((RUN/'HOST_RESULT.json').read_text());arb=json.loads((RUN/'ARBITER_RESULT.json').read_text())
 # Source/binary lifetime and all exact backup head connections independently verified.
 backup_info=[]
 for e in backup_actual:
  sel=selected[(selected['bundle']==e['bundle'])&(np.abs(selected['y']-(e['z']-planbid(e['bundle'])['y']+planbid(e['bundle'])['z']))<.01)]
  # Match by planned ROS phase directly.
  a=planbid(e['bundle']);phase=e['z']-a['y'];row=min(bundles[int(e['bundle'])].stops,key=lambda s:abs(s.phase-phase))
  continuous=np.max(np.abs(row.polynomial.pva(0)-bundles[int(e['bundle'])].prefix.pva(row.phase)),axis=1)
  backup_info.append(dict(bundle=int(e['bundle']),phase=row.phase,first_publish_begin_lateness=float(e['x']),first_publish_end_lateness=float(e['y']),head_PVA_gap=continuous.tolist(),true_pva=[e[k] for k in ids]))
 res=[json.loads(l) for l in (RUN/'RESOURCE_TRACE.jsonl').read_text().splitlines()]
 he=[json.loads(l) for l in (RUN/'HOST_EVENTS.jsonl').read_text().splitlines()]
 cam=[json.loads(l) for l in (RUN/'CAMERA_TRACE.jsonl').read_text().splitlines()]
 scene=json.loads((RUN/'SCENE_READY.json').read_text())
 with np.load(RUN/'LOADED_MAP.npz') as z:loaded=hashlib.sha256(z['quantised'].astype('<i4').tobytes()).hexdigest()
 assert loaded==json.loads((OUT/'scene/SCENE_VALIDATION.json').read_text())['quantised_xyz_sha256']
 fail_events=events[np.isin(events['code'],[34,40,41,42,43,44,45])]
 final=dict(status='TO_EVALUATE',Query_control=False,arbiter=arb,
  original_host_receipt=host['status'],host_receipt_race=host.get('exception'),
  command_count=len(cmd),command_gap_p50_p95_max_ms=quant(np.diff(cmd['begin'])*1000),
  wake_delay_p50_p95_max_ms=quant((cmd['mono']-cmd['wake'])*1000),tick_thread_cpu_p50_p95_max_ms=quant((cmd['cpu1']-cmd['cpu0'])*1000),
  publish_p50_p95_max_ms=quant((cmd['end']-cmd['begin'])*1000),
  callback_receive_to_processed_p50_p95_max_ms=quant((odom['done']-odom['recv'])*1000),
  callback_source_to_receive_p50_p95_max_ms=quant((odom['recv']-odom['stamp']+np.median(odom['stamp']-odom['recv']))*1000),
  callback_delay_note='ROS wall/monotonic offset aligned by median; raw original timestamps retained',
  max_command_reference_residual=residual.max(axis=0).tolist(),all_phase_tracking_error_norm_p50_p95_max=np.quantile(error,[.5,.95,1],axis=0).tolist(),
  observed_speed_max=float(np.linalg.norm(ov[:,1],axis=1).max()),observed_acceleration_max=float(np.linalg.norm(ov[:,2],axis=1).max()),
  actual_norm_limits_are_sampled_not_continuous_proof=True,all_phase_min_z=float(ov[:,0,2].min()),
  activations_lateness_ms=physical['y'].tolist(),activation_max_ms=float(physical['y'].max()*1000),
  backup_gate_count=len(gates),backup_gates_pass=bool((gates['y']==1).all()),backups=backup_info,
  stop_dwells=dwells,fail_event_count=len(fail_events),raw_fail_events=[dict(zip(events.dtype.names,map(float,e))) for e in fail_events],
  moving_window_duration=0 if not window else window[-1]['stamp']-window[0]['stamp'],
  moving_window_first=None if not window else window[0],moving_window_last=None if not window else window[-1],
  instant_handover=crossing,instant_position_error=float(np.linalg.norm(np.array(crossing['pva'][0])-[0,0,2])),
  resources=dict(WSL_min_GiB=min(e['MemAvailable_kb'] for e in res)/1048576,Windows_min_GiB=min(e['available_kb'] for e in he if e['type']=='WINDOWS_AVAILABLE')/1048576,
   GPU_peak_MiB=max(e['gpu']['used_mib'] for e in res),swap_peak_kb=max(e['SwapUsed_kb'] for e in res)),
  cleanup_remaining=json.loads((RUN/'CLEANUP.json').read_text())['remaining'],camera_fatal=any(e.get('type')=='FATAL' for e in cam),
  camera_exit=[e['type'] for e in cam if 'type' in e][-4:],scene_verified=True,
  unique_publisher_in_scene=dict(scene['publishers']).get('/so3_control/pos_cmd')==['/f3_reference_arbiter'])
 passed=(arb['status']=='REFERENCE_TRIALS_COMPLETE' and not arb['qualification_failed'] and not arb['buffer_overflow'] and len(cmd)==arb['commands'] and np.diff(cmd['begin']).max()<=.04
  and (error<=[.02,.05,.1]).all() and norm_ok(ov) and ov[:,0,2].min()>1.9 and physical['y'].max()<=.01 and (gates['y']==1).all()
  and all(x['first_publish_end_lateness']<=.01 and max(x['head_PVA_gap'])<.01 for x in backup_info) and len(dwells)==2 and all(x['passed'] for x in dwells)
  and final['moving_window_duration']>=.5 and final['instant_position_error']<=.02 and len(fail_events)==0 and not final['cleanup_remaining'] and not final['camera_fatal'])
 final['dwell_checker_note']='initial fixed0.51s window dropped an eligible preceding sample; final checks complete contiguous measured-stop suffix with unchanged0.5s/40ms thresholds; initial result retained'
 final['necessary_control_gates_pass']=bool(passed);final['status']='CONTROL_GATES_PASS_HOST_RECEIPT_RACE_RETAINED' if passed else 'FAIL_BLOCKED'
 with (OUT/'NATIVE_MEASURED_RESULTS_FINAL.json').open('x') as f:json.dump(final,f,indent=2)
 print(json.dumps({k:v for k,v in final.items() if k not in ['host_receipt_race','moving_window_first','moving_window_last','instant_handover','raw_fail_events']},indent=2))
def norm_ok(a):return np.isfinite(a).all() and np.linalg.norm(a[:,1],axis=1).max()<=1 and np.linalg.norm(a[:,2],axis=1).max()<=1.5
if __name__=='__main__':main()
