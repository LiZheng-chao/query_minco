"""Fail-closed predecessor and runtime provenance contracts."""
import json,math
from pathlib import Path
from query_minco.evaluation.mirror_holdout_v1 import sha
CASES=['request_timeout','depth_loss','odom_rollback','nan_control','reference_exit','controller_exit']
def expected_fault(case,reason):
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
