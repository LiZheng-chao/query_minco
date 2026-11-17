"""Fail-closed predecessor and runtime provenance contracts."""
import json,math
from pathlib import Path
from query_minco.evaluation.mirror_holdout_v1 import sha
CASES=['request_timeout','depth_loss','odom_rollback','nan_control','reference_exit','controller_exit']
def expected_fault(case,reason):
 if case=='depth_loss':return reason in ('depth_STALE',"REQUEST ValueError('F0_DEPTH_STALE')")
 allowed={'request_timeout':['request_'],'depth_loss':['depth_'],'odom_rollback':['odom_'],'nan_control':['REQUEST '],'reference_exit':['reference_EXIT'],'controller_exit':['controller_EXIT']}
 return any(str(reason).startswith(p) for p in allowed.get(case,[]))
def predecessor(path,stage,binding):
 p=Path(path);d=json.loads(p.read_text())
 if d.get('stage')!=stage or d.get('status')!='PASS' or d.get('source_binding')!=binding:raise PermissionError('PREDECESSOR_STAGE_SHA_OR_STATUS')
 if stage=='C_CAMPAIGN':
  if [x['case'] for x in d['episodes']]!=CASES or not all(x['status']=='PASS' for x in d['episodes']):raise PermissionError('INCOMPLETE_C')
  for x in d['episodes']:
   q=Path(x['result_path'])
   if sha(q)!=x['result_sha256'] or json.loads(q.read_text())['status']!='PASS':raise PermissionError('C_EPISODE_SHA')
 if stage=='D':
  if d.get('runtime_status')!='PASS' or d.get('cleanup_status')!='PASS' or not d.get('physical_feedback'):raise PermissionError('D_NOT_PHYSICAL')
 return d
def legacy_window(path):
 d=json.loads(Path(path).read_text())
 if d.get('scope')!='PHYSICAL_WINDOW_ONLY_NOT_CLEANUP' or d.get('status')!='PASS':raise PermissionError('LEGACY_WINDOW')
 p=Path(d['original_acceptance'])
 if sha(p)!=d['original_acceptance_sha256'] or sha(p.parent/'preparation/EVENTS.jsonl')!=d['events_sha256']:raise PermissionError('LEGACY_B_SHA')
 return d

def depth_loss_causality(events,fault,terminal,stop_time):
 """Post-stop evidence check, never used to defer termination."""
 try:
  if fault['case']!='depth_loss' or fault['armed'] is not True or fault['steps']<=0:return False
  begin=fault['monotonic'];fault_ros=fault['ros_time']
  if not all(math.isfinite(x) for x in (begin,fault_ros,stop_time)):return False
  if not 0<=stop_time-begin<=.35:return False
  if terminal['terminal_state'] not in ('OWNED_PROCESS_TERMINATED','OWNED_PROCESS_ALREADY_EXITED') or terminal['actual_exit_code'] is None or not terminal['pid'] or not terminal['start_ticks']:return False
  plants=[e for e in events if e.get('kind')=='plant' and e.get('accepted') and e.get('mode')=='ARMED' and e['monotonic']<=begin and e['payload'].get('integration_steps',0)>0 and e['payload'].get('state_source')=='DYNAMICS_INTEGRATED']
  if not plants:return False
  faults=[e for e in events if e.get('kind')=='fault' and expected_fault('depth_loss',e['payload'].get('reason')) and begin<=e['monotonic']<=stop_time]
  if not faults:return False
  detected=min(faults,key=lambda e:e['monotonic'])
  depths=[e for e in events if e.get('kind')=='depth' and e.get('accepted') and e['monotonic']<=detected['monotonic']]
  if not depths:return False
  last=max(depths,key=lambda e:e['stamp'])
  if last['payload']['source_odom_stamp']!=last['stamp']:return False
  if last['monotonic']>begin or not 0<=fault_ros-last['stamp']<=.2:return False
  if not all(math.isfinite(x) for x in (detected['ros_now'],detected['monotonic'],last['stamp'],last['monotonic'])):return False
  if not detected['ros_now']-last['stamp']>.2:return False
  if terminal['reason']!=detected['payload']['reason']:return False
  return True
 except (KeyError,TypeError,ValueError,IndexError):return False
